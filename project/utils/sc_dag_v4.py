import hashlib
import inspect
import json
import os
import uuid

import networkx as nx
import requests
import scanpy as sc


DEFAULT_STORAGE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "workspace", "sc_dag_v4")
)


class Rule:
    def __init__(self, name, requires, func, virtual=False):
        self.name = name
        self.requires = requires
        self.func = func
        self.virtual = virtual

        sig = inspect.signature(func)
        system_params = {"mgr", "parent_id"}
        self.param_keys = [k for k in sig.parameters if k not in system_params]


class RuleRegistry:
    def __init__(self):
        self.rules = {}

    def register(self, rule):
        self.rules[rule.name] = rule

    def get(self, name):
        if name not in self.rules:
            raise ValueError(f"Rule '{name}' is not registered.")
        return self.rules[name]

    def has(self, name):
        return name in self.rules


def _file_fingerprint(path):
    if not path:
        raise ValueError("Lineage error: data_path is required when loading raw data.")

    abs_path = os.path.abspath(path)
    try:
        if os.path.isdir(abs_path):
            file_names = [
                "matrix.mtx",
                "matrix.mtx.gz",
                "genes.tsv",
                "genes.tsv.gz",
                "features.tsv",
                "features.tsv.gz",
                "barcodes.tsv",
                "barcodes.tsv.gz",
            ]
            files = []
            for file_name in file_names:
                file_path = os.path.join(abs_path, file_name)
                if os.path.exists(file_path):
                    file_stat = os.stat(file_path)
                    files.append(
                        {
                            "name": file_name,
                            "size": file_stat.st_size,
                            "mtime_ns": file_stat.st_mtime_ns,
                        },
                    )
            return {
                "path": abs_path,
                "kind": "directory",
                "files": files,
            }

        stat = os.stat(abs_path)
        identity = {
            "path": abs_path,
            "kind": "file",
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    except OSError:
        identity = {"path": abs_path}
    return identity


def compute_step_hash(mgr, rule_name, parent_hash, all_params):
    if rule_name == "raw":
        data_paths = all_params.get("data_paths")
        data_path = all_params.get("data_path")
        if data_paths:
            sample_ids = all_params.get("sample_ids") or [
                f"sample_{idx + 1}" for idx in range(len(data_paths))
            ]
            data = {
                "rule": "raw",
                "source": [
                    {
                        "sample_id": sample_id,
                        "fingerprint": _file_fingerprint(path),
                    }
                    for path, sample_id in zip(data_paths, sample_ids)
                ],
                "sample_key": all_params.get("sample_key"),
                "join": all_params.get("multi_sample_join"),
            }
        else:
            if not data_path:
                raw_nodes = [n for n, attr in mgr.graph.nodes(data=True) if attr.get("action") == "raw"]
                if len(raw_nodes) == 1:
                    data_path = mgr.graph.nodes[raw_nodes[0]].get("params", {}).get("data_path")
            data = {
                "rule": "raw",
                "source": _file_fingerprint(data_path),
            }
    else:
        rule = mgr.registry.get(rule_name)
        relevant_params = {k: v for k, v in all_params.items() if k in rule.param_keys}
        data = {
            "parent_hash": parent_hash,
            "rule": rule_name,
            "params": relevant_params,
        }

    return hashlib.md5(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()

class SCStateManager:
    def __init__(self, storage_dir=DEFAULT_STORAGE_DIR):
        self.graph = nx.DiGraph()
        self.objects = {}
        self.registry = RuleRegistry()
        self.hash_index = {}
        self.storage_dir = storage_dir
        self.object_dir = os.path.join(self.storage_dir, "objects")
        self.graph_path = os.path.join(self.storage_dir, "graph.json")
        self.load()

    def _new_id(self):
        return str(uuid.uuid4())[:8]

    def _register_node(self, node_id, hash_val, **attr):
        self.graph.add_node(node_id, hash=hash_val, **attr)
        self.hash_index[hash_val] = node_id

    def register_new_object(self, adata, parent_id, action, params, hash_val=None, result_key=None):
        obj_ref = f"obj_{self._new_id()}"
        self.objects[obj_ref] = adata

        node_id = f"node_{self._new_id()}"
        self._register_node(
            node_id,
            hash_val,
            action=action,
            params=params,
            obj_ref=obj_ref,
            is_virtual=False,
            result_key=result_key,
            shape=adata.shape,
        )
        if parent_id:
            self.graph.add_edge(parent_id, node_id)
        self.save()
        return node_id

    def register_virtual_node(self, adata, parent_id, action, params, result_key, hash_val=None):
        parent_obj_ref = self.graph.nodes[parent_id]["obj_ref"]
        self.objects[parent_obj_ref] = adata

        node_id = f"node_{self._new_id()}"
        self._register_node(
            node_id,
            hash_val,
            action=action,
            params=params,
            obj_ref=parent_obj_ref,
            is_virtual=True,
            result_key=result_key,
            shape=adata.shape,
        )
        self.graph.add_edge(parent_id, node_id)
        self.save()
        return node_id

    def get_object(self, node_id):
        if node_id not in self.graph.nodes:
            raise ValueError(f"Node {node_id} not found.")
        obj_ref = self.graph.nodes[node_id].get("obj_ref")
        return self.objects[obj_ref]

    def save(self):
        os.makedirs(self.object_dir, exist_ok=True)

        for obj_ref, adata in self.objects.items():
            object_path = os.path.join(self.object_dir, f"{obj_ref}.h5ad")
            adata.write_h5ad(object_path)

        graph_data = {
            "nodes": [
                {"id": node_id, **self._json_safe_attrs(attr)}
                for node_id, attr in self.graph.nodes(data=True)
            ],
            "edges": [[src, dst] for src, dst in self.graph.edges()],
        }

        tmp_path = f"{self.graph_path}.tmp"
        with open(tmp_path, "w") as f:
            json.dump(graph_data, f, indent=2, sort_keys=True)
        os.replace(tmp_path, self.graph_path)

    def load(self):
        if not os.path.exists(self.graph_path):
            return

        with open(self.graph_path) as f:
            graph_data = json.load(f)

        self.graph.clear()
        self.objects.clear()
        self.hash_index.clear()

        for node in graph_data.get("nodes", []):
            node = dict(node)
            node_id = node.pop("id")
            self.graph.add_node(node_id, **node)
            node_hash = node.get("hash")
            if node_hash is not None:
                self.hash_index[node_hash] = node_id

        self.graph.add_edges_from(graph_data.get("edges", []))

        obj_refs = {
            attr.get("obj_ref")
            for _, attr in self.graph.nodes(data=True)
            if attr.get("obj_ref") is not None
        }
        for obj_ref in obj_refs:
            object_path = os.path.join(self.object_dir, f"{obj_ref}.h5ad")
            if os.path.exists(object_path):
                self.objects[obj_ref] = sc.read_h5ad(object_path)

    def _json_safe_attrs(self, attrs):
        safe_attrs = {}
        for key, value in attrs.items():
            if isinstance(value, tuple):
                safe_attrs[key] = list(value)
            else:
                safe_attrs[key] = value
        return safe_attrs

    def dependency_chain(self, target_stage):
        if target_stage == "raw":
            return ["raw"]

        chain = []
        curr = target_stage
        seen = set()
        while curr != "raw":
            if curr in seen:
                raise ValueError(f"Cycle detected while resolving '{target_stage}'.")
            seen.add(curr)
            chain.insert(0, curr)
            rule = self.registry.get(curr)
            if not rule.requires:
                break
            curr = rule.requires[0]
        chain.insert(0, "raw")
        return chain

    def find_node_smart(self, target_stage, **user_params):
        strict_id = self.find_node_strict(target_stage, **user_params)
        if strict_id:
            return strict_id, "exact_match"

        if not self.registry.has(target_stage):
            return None, "no_match"

        target_rule = self.registry.get(target_stage)
        candidates = []
        for node_id, attr in self.graph.nodes(data=True):
            if attr.get("action") != target_stage:
                continue

            node_params = attr.get("params", {})
            for key in target_rule.param_keys:
                user_val = user_params.get(key)
                if user_val is not None and str(user_val) != str(node_params.get(key)):
                    break
            else:
                candidates.append(node_id)

        if not candidates:
            return None, "no_match"
        if len(candidates) == 1:
            return candidates[0], "fuzzy_match"
        return candidates, "ambiguous"

    def find_node_strict(self, target_stage, **full_params):
        chain = self.dependency_chain(target_stage)
        current_hash = "init"

        for stage in chain:
            current_hash = compute_step_hash(self, stage, current_hash, full_params)

        return self.hash_index.get(current_hash)


def run_rule(mgr, rule_name, parent_id, **params):
    parent_hash = mgr.graph.nodes[parent_id].get("hash", "init") if parent_id else "init"
    hash_val = compute_step_hash(mgr, rule_name, parent_hash, params)

    if hash_val in mgr.hash_index:
        return mgr.hash_index[hash_val]

    rule = mgr.registry.get(rule_name)
    rule_params = {k: v for k, v in params.items() if k in rule.param_keys}
    result = rule.func(mgr, parent_id, **rule_params)

    if result[1] == "new_object":
        result_key = result[2] if len(result) > 2 else None
        return mgr.register_new_object(result[0], parent_id, rule_name, rule_params, hash_val, result_key)
    if result[1] == "virtual":
        return mgr.register_virtual_node(result[0], parent_id, rule_name, rule_params, result[2], hash_val)

    raise ValueError(f"Unknown rule result type: {result[1]}")


def ensure(mgr, target, start_state, **params):
    if target == "raw":
        return start_state

    rule = mgr.registry.get(target)
    current_action = mgr.graph.nodes[start_state]["action"] if start_state else "raw"

    if current_action == target:
        preds = list(mgr.graph.predecessors(start_state))
        return run_rule(mgr, target, preds[0], **params) if preds else start_state

    if current_action in rule.requires:
        return run_rule(mgr, target, start_state, **params)

    parent_stage = rule.requires[0]
    parent_id = ensure(mgr, parent_stage, start_state, **params)
    return run_rule(mgr, target, parent_id, **params)


def register_raw(mgr, adata, path, params=None):
    params = params or {"data_path": path}
    hash_val = compute_step_hash(mgr, "raw", "init", params)
    if hash_val in mgr.hash_index:
        return mgr.hash_index[hash_val]
    return mgr.register_new_object(adata, None, "raw", params, hash_val)


def scrublet_rule(
    mgr,
    parent_id,
    scrublet_batch_key=None,
    scrublet_expected_doublet_rate=0.05,
    scrublet_threshold=None,
    scrublet_n_prin_comps=30,
    scrublet_filter_doublets=False,
    scrublet_skip_on_failure=True,
):
    adata = mgr.get_object(parent_id).copy()

    if scrublet_batch_key is not None and scrublet_batch_key not in adata.obs:
        raise ValueError(f"scrublet_batch_key '{scrublet_batch_key}' not found in adata.obs.")

    effective_n_prin_comps = max(1, min(scrublet_n_prin_comps, adata.n_obs - 1, adata.n_vars - 1))
    try:
        sc.pp.scrublet(
            adata,
            batch_key=scrublet_batch_key,
            expected_doublet_rate=scrublet_expected_doublet_rate,
            threshold=scrublet_threshold,
            n_prin_comps=effective_n_prin_comps,
            random_state=0,
        )
    except Exception as exc:
        if not scrublet_skip_on_failure:
            raise
        adata.uns["scrublet"] = {
            "status": "skipped",
            "error": str(exc),
            "batch_key": scrublet_batch_key,
            "expected_doublet_rate": scrublet_expected_doublet_rate,
            "threshold": scrublet_threshold,
            "n_prin_comps": effective_n_prin_comps,
        }
        return adata, "new_object", "scrublet_skipped"

    if scrublet_filter_doublets:
        if "predicted_doublet" not in adata.obs:
            raise ValueError("Scrublet did not produce adata.obs['predicted_doublet'].")
        adata = adata[~adata.obs["predicted_doublet"].astype(bool)].copy()

    adata.uns["scrublet"] = {
        "status": "completed",
        "batch_key": scrublet_batch_key,
        "expected_doublet_rate": scrublet_expected_doublet_rate,
        "threshold": scrublet_threshold,
        "n_prin_comps": effective_n_prin_comps,
        "filtered_doublets": scrublet_filter_doublets,
    }
    result_key = "scrublet_filtered" if scrublet_filter_doublets else "scrublet"
    return adata, "new_object", result_key


def qc_filter_rule(mgr, parent_id, qc_min_genes=200, qc_max_genes=2500, qc_mt_pct=5, min_cells=3):
    adata = mgr.get_object(parent_id).copy()
    sc.pp.filter_genes(adata, min_cells=min_cells)
    adata.var["mt"] = adata.var_names.str.startswith(("MT-", "mt-"))
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    sc.pp.filter_cells(adata, min_genes=qc_min_genes)
    sc.pp.filter_cells(adata, max_genes=qc_max_genes)
    if qc_mt_pct is not None:
        adata = adata[adata.obs["pct_counts_mt"] < qc_mt_pct, :].copy()
    return adata, "new_object"


def normalize_rule(mgr, parent_id, target_sum=1e4):
    adata = mgr.get_object(parent_id).copy()
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    adata.raw = adata.copy()
    return adata, "new_object"


def hvg_rule(mgr, parent_id, n_hvg=2000, hvg_flavor="seurat", hvg_batch_key=None):
    adata = mgr.get_object(parent_id).copy()
    if hvg_batch_key is not None and hvg_batch_key not in adata.obs:
        raise ValueError(f"hvg_batch_key '{hvg_batch_key}' not found in adata.obs.")
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_hvg,
        flavor=hvg_flavor,
        batch_key=hvg_batch_key,
    )
    adata = adata[:, adata.var["highly_variable"]].copy()
    return adata, "new_object"


def batch_correct_rule(
    mgr,
    parent_id,
    batch_correction_method="none",
    combat_key=None,
    sample_key="sample",
):
    adata = mgr.get_object(parent_id).copy()
    method = (batch_correction_method or "none").lower()

    if method == "none":
        adata.uns["batch_correction"] = {"method": "none"}
        return adata, "new_object", "batch_correction_none"

    if method == "combat":
        key = combat_key or sample_key
        if key not in adata.obs:
            raise ValueError(f"ComBat batch key '{key}' not found in adata.obs.")
        sc.pp.combat(adata, key=key, inplace=True)
        adata.uns["batch_correction"] = {
            "method": "combat",
            "key": key,
        }
        return adata, "new_object", f"combat_{key}"

    raise ValueError("batch_correction_method must be 'none' or 'combat'.")


def scale_rule(mgr, parent_id, max_scale_value=10, regress_out=True):
    adata = mgr.get_object(parent_id).copy()
    if regress_out:
        regressors = [key for key in ["total_counts", "pct_counts_mt"] if key in adata.obs]
        if regressors:
            sc.pp.regress_out(adata, regressors)
    sc.pp.scale(adata, max_value=max_scale_value)
    return adata, "new_object"


def pca_rule(mgr, parent_id, n_comps=50):
    adata = mgr.get_object(parent_id).copy()
    effective_n_comps = max(1, min(n_comps, adata.n_obs - 1, adata.n_vars - 1))
    sc.tl.pca(adata, n_comps=effective_n_comps, svd_solver="arpack")
    return adata, "new_object", "X_pca"


def neighbors_rule(mgr, parent_id, n_neighbors=10, n_pcs=40, use_rep="X_pca"):
    adata = mgr.get_object(parent_id).copy()
    effective_n_pcs = n_pcs
    if use_rep in adata.obsm:
        effective_n_pcs = min(n_pcs, adata.obsm[use_rep].shape[1])
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=effective_n_pcs, use_rep=use_rep)
    return adata, "new_object", "neighbors"


def umap_rule(mgr, parent_id, min_dist=0.5, spread=1.0):
    adata = mgr.get_object(parent_id).copy()
    sc.tl.umap(adata, min_dist=min_dist, spread=spread)
    return adata, "new_object", "X_umap"


def cluster_rule(mgr, parent_id, resolution=0.5, cluster_method="leiden"):
    adata = mgr.get_object(parent_id)
    key_added = f"{cluster_method}_res{resolution}"

    if cluster_method == "leiden":
        sc.tl.leiden(
            adata,
            resolution=resolution,
            key_added=key_added,
            random_state=0,
            flavor="igraph",
            n_iterations=2,
            directed=False,
        )
    elif cluster_method == "louvain":
        sc.tl.louvain(adata, resolution=resolution, key_added=key_added)
    else:
        raise ValueError(f"Unknown clustering method: {cluster_method}")

    return adata, "virtual", key_added


def markers_rule(mgr, parent_id, groupby=None, marker_method="wilcoxon", n_marker_genes=25):
    adata = mgr.get_object(parent_id).copy()
    if groupby is None:
        groupby = mgr.graph.nodes[parent_id].get("result_key")
    if groupby is None or groupby not in adata.obs:
        raise ValueError("A valid groupby key is required for marker detection.")

    result_key = f"markers_{groupby}_{marker_method}_{n_marker_genes}"
    sc.tl.rank_genes_groups(
        adata,
        groupby=groupby,
        method=marker_method,
        n_genes=n_marker_genes,
        key_added=result_key,
        use_raw=adata.raw is not None,
    )
    return adata, "new_object", result_key


def _extract_cluster_markers(adata, groupby, marker_key, n_markers):
    markers = sc.get.rank_genes_groups_df(
        adata,
        group=None,
        key=marker_key,
    )

    cluster_markers = {}
    for cluster in sorted(adata.obs[groupby].astype(str).unique()):
        genes = (
            markers[markers["group"].astype(str) == cluster]["names"]
            .astype(str)
            .head(n_markers)
            .tolist()
        )
        cluster_markers[cluster] = genes

    return cluster_markers


def _parse_annotation_response(content):
    content = content.strip()
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"Annotation model did not return JSON: {content[:200]}")
        parsed = json.loads(content[start : end + 1])

    if "annotations" in parsed and isinstance(parsed["annotations"], dict):
        parsed = parsed["annotations"]
    if not isinstance(parsed, dict):
        raise ValueError("Annotation model response must be a JSON object mapping clusters to cell types.")

    return {str(cluster): str(label) for cluster, label in parsed.items()}


def annotation_rule(
    mgr,
    parent_id,
    groupby=None,
    annotation_model="qwen3.5:122b",
    annotation_api_base="http://localhost:11434/v1",
    annotation_api_key="ollama",
    n_annotation_markers=10,
):
    adata = mgr.get_object(parent_id).copy()
    marker_key = mgr.graph.nodes[parent_id].get("result_key")

    if marker_key is None or marker_key not in adata.uns:
        raise ValueError("A valid markers parent is required for cell-type annotation.")

    if groupby is None:
        marker_params = mgr.graph.nodes[parent_id].get("params", {})
        groupby = marker_params.get("groupby")
        if groupby is None:
            cluster_parent = next(iter(mgr.graph.predecessors(parent_id)), None)
            if cluster_parent is not None:
                groupby = mgr.graph.nodes[cluster_parent].get("result_key")

    if groupby is None or groupby not in adata.obs:
        raise ValueError("A valid groupby key is required for cell-type annotation.")

    cluster_markers = _extract_cluster_markers(
        adata,
        groupby=groupby,
        marker_key=marker_key,
        n_markers=n_annotation_markers,
    )

    prompt = {
        "task": "Annotate single-cell RNA-seq clusters from marker genes.",
        "instructions": (
            "Return JSON only. Use cluster IDs as keys and concise cell-type labels as values. "
            'Example: {"0":"CD4 T cell","1":"B cell"}.'
        ),
        "groupby": groupby,
        "marker_key": marker_key,
        "clusters": cluster_markers,
    }

    response = requests.post(
        f"{annotation_api_base.rstrip('/')}/chat/completions",
        headers={
            "Authorization": f"Bearer {annotation_api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": annotation_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert in single-cell RNA-seq cell-type annotation. "
                        "Use canonical marker-gene knowledge to choose the best label for each cluster."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(prompt),
                },
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        },
        timeout=120,
    )
    response.raise_for_status()

    annotations = _parse_annotation_response(response.json()["choices"][0]["message"]["content"])
    result_key = f"cell_type_annotation_{groupby}_{annotation_model.replace(':', '_')}_{n_annotation_markers}"

    adata.obs["cell_type"] = adata.obs[groupby].astype(str).map(annotations).fillna("Unknown")
    adata.uns[result_key] = {
        "groupby": groupby,
        "model": annotation_model,
        "api_base": annotation_api_base,
        "marker_key": marker_key,
        "n_markers": n_annotation_markers,
        "cluster_markers": cluster_markers,
        "annotations": annotations,
    }
    adata.uns["cell_type_annotation"] = adata.uns[result_key]

    return adata, "new_object", result_key


mgr = SCStateManager()

mgr.registry.register(Rule("scrublet", ["raw"], scrublet_rule))
mgr.registry.register(Rule("qc", ["scrublet"], qc_filter_rule))
mgr.registry.register(Rule("normalize", ["qc"], normalize_rule))
mgr.registry.register(Rule("hvg", ["normalize"], hvg_rule))
mgr.registry.register(Rule("batch_correct", ["hvg"], batch_correct_rule))
mgr.registry.register(Rule("scale", ["batch_correct"], scale_rule))
mgr.registry.register(Rule("pca", ["scale"], pca_rule))
mgr.registry.register(Rule("neighbors", ["pca"], neighbors_rule))
mgr.registry.register(Rule("umap", ["neighbors"], umap_rule))
mgr.registry.register(Rule("cluster", ["umap"], cluster_rule, virtual=True))
mgr.registry.register(Rule("markers", ["cluster"], markers_rule))
mgr.registry.register(Rule("annotation", ["markers"], annotation_rule))


def get_manager():
    return mgr
