"""Deterministic pipeline rules and annotation helpers for the single-cell DAG."""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import requests
import scanpy as sc


@dataclass
class Rule:
    """
    Describe one pipeline stage and its callable implementation.

    Constructed by `create_default_registry()` and consumed by sc_run and sc_dag
    for dependency planning, parameter filtering, cache identity, and execution.
    """

    name: str
    requires: List[str]
    func: Callable[..., Any]
    virtual: bool = False

    def __post_init__(self) -> None:
        sig = inspect.signature(self.func)
        system_params = {"mgr", "parent_id"}
        self.param_keys = [key for key in sig.parameters if key not in system_params]


class RuleRegistry:
    """
    Hold registered pipeline rules keyed by stage name.

    Attached to `SCStateManager` in sc_run and used by the runner, DAG hash
    functions, and dependency resolution.
    """

    def __init__(self) -> None:
        self.rules: Dict[str, Rule] = {}

    def register(self, rule: Rule) -> None:
        """Add or replace a stage rule in the registry."""
        self.rules[rule.name] = rule

    def get(self, name: str) -> Rule:
        """Return a registered rule or raise if the stage is unknown."""
        if name not in self.rules:
            raise ValueError(f"Rule '{name}' is not registered.")
        return self.rules[name]

    def has(self, name: str) -> bool:
        """Check whether a stage name is registered."""
        return name in self.rules


def scrublet_rule(
    mgr: Any,
    parent_id: str,
    scrublet_batch_key: str = None,
    scrublet_expected_doublet_rate: float = 0.05,
    scrublet_threshold: float = None,
    scrublet_n_prin_comps: int = 30,
    scrublet_filter_doublets: bool = False,
    scrublet_skip_on_failure: bool = True,
):
    """
    Run Scrublet on raw counts and optionally remove predicted doublets.

    Called by `run_rule()` in sc_run. It reads the parent AnnData from the state
    manager, returns a new AnnData object, and records Scrublet metadata in
    `adata.uns["scrublet"]`.
    """
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


def qc_filter_rule(
    mgr: Any,
    parent_id: str,
    qc_min_genes: int = 200,
    qc_max_genes: int = 2500,
    qc_mt_pct: float = 5,
    min_cells: int = 3,
):
    """
    Apply QC filtering and compute standard QC metrics.

    Called by `run_rule()` in sc_run. It returns a filtered AnnData copy and
    updates `adata.obs` and `adata.var` with QC annotations.
    """
    adata = mgr.get_object(parent_id).copy()
    sc.pp.filter_genes(adata, min_cells=min_cells)
    adata.var["mt"] = adata.var_names.str.startswith(("MT-", "mt-"))
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    sc.pp.filter_cells(adata, min_genes=qc_min_genes)
    sc.pp.filter_cells(adata, max_genes=qc_max_genes)
    if qc_mt_pct is not None:
        adata = adata[adata.obs["pct_counts_mt"] < qc_mt_pct, :].copy()
    return adata, "new_object"


def normalize_rule(mgr: Any, parent_id: str, target_sum: float = 1e4):
    """
    Normalize counts, log-transform, and preserve the normalized snapshot as `raw`.

    Called by `run_rule()` in sc_run. It returns a new AnnData object with
    normalized expression values.
    """
    adata = mgr.get_object(parent_id).copy()
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    adata.raw = adata.copy()
    return adata, "new_object"


def hvg_rule(
    mgr: Any,
    parent_id: str,
    n_hvg: int = 2000,
    hvg_flavor: str = "seurat",
    hvg_batch_key: str = None,
):
    """
    Select highly variable genes and subset the matrix to them.

    Called by `run_rule()` in sc_run. It returns a new AnnData object restricted
    to the selected HVGs.
    """
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
    mgr: Any,
    parent_id: str,
    batch_correction_method: str = "none",
    combat_key: str = None,
    sample_key: str = "sample",
):
    """
    Apply optional batch correction after HVG selection.

    Called by `run_rule()` in sc_run. It returns a new AnnData object and stores
    batch-correction metadata under `adata.uns["batch_correction"]`.
    """
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
        adata.uns["batch_correction"] = {"method": "combat", "key": key}
        return adata, "new_object", f"combat_{key}"

    raise ValueError("batch_correction_method must be 'none' or 'combat'.")


def scale_rule(mgr: Any, parent_id: str, max_scale_value: int = 10, regress_out: bool = True):
    """
    Regress nuisance covariates and scale the expression matrix.

    Called by `run_rule()` in sc_run. It returns a new scaled AnnData object.
    """
    adata = mgr.get_object(parent_id).copy()
    if regress_out:
        regressors = [key for key in ["total_counts", "pct_counts_mt"] if key in adata.obs]
        if regressors:
            sc.pp.regress_out(adata, regressors)
    sc.pp.scale(adata, max_value=max_scale_value)
    return adata, "new_object"


def pca_rule(mgr: Any, parent_id: str, n_comps: int = 50):
    """
    Compute PCA on the scaled matrix.

    Called by `run_rule()` in sc_run. It returns a new AnnData object with
    `adata.obsm["X_pca"]`.
    """
    adata = mgr.get_object(parent_id).copy()
    effective_n_comps = max(1, min(n_comps, adata.n_obs - 1, adata.n_vars - 1))
    sc.tl.pca(adata, n_comps=effective_n_comps, svd_solver="arpack")
    return adata, "new_object", "X_pca"


def neighbors_rule(
    mgr: Any,
    parent_id: str,
    n_neighbors: int = 10,
    n_pcs: int = 40,
    use_rep: str = "X_pca",
):
    """
    Build the nearest-neighbor graph for downstream embedding and clustering.

    Called by `run_rule()` in sc_run. It returns a new AnnData object with the
    neighbor graph in `adata.uns` and `adata.obsp`.
    """
    adata = mgr.get_object(parent_id).copy()
    effective_n_pcs = n_pcs
    if use_rep in adata.obsm:
        effective_n_pcs = min(n_pcs, adata.obsm[use_rep].shape[1])
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=effective_n_pcs, use_rep=use_rep)
    return adata, "new_object", "neighbors"


def umap_rule(mgr: Any, parent_id: str, min_dist: float = 0.5, spread: float = 1.0):
    """
    Compute a UMAP embedding from the neighbor graph.

    Called by `run_rule()` in sc_run. It returns a new AnnData object with
    `adata.obsm["X_umap"]`.
    """
    adata = mgr.get_object(parent_id).copy()
    sc.tl.umap(adata, min_dist=min_dist, spread=spread)
    return adata, "new_object", "X_umap"


def cluster_rule(mgr: Any, parent_id: str, resolution: float = 0.5, cluster_method: str = "leiden"):
    """
    Cluster cells from the UMAP-ready state without mutating the parent node.

    Called by `run_rule()` in sc_run. It operates on a copy so the parent UMAP
    node remains unchanged, and returns a virtual lineage node keyed by the new
    cluster assignment column.
    """
    adata = mgr.get_object(parent_id).copy()
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


def markers_rule(
    mgr: Any,
    parent_id: str,
    groupby: str = None,
    marker_method: str = "wilcoxon",
    n_marker_genes: int = 25,
):
    """
    Rank marker genes for each cluster or grouping.

    Called by `run_rule()` in sc_run. It returns a new AnnData object with the
    ranking stored under a stage-specific `adata.uns` key.
    """
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


def extract_cluster_markers(adata: Any, groupby: str, marker_key: str, n_markers: int) -> Dict[str, List[str]]:
    """
    Convert rank_genes_groups output into a cluster-to-marker-gene mapping.

    Called by `annotation_rule()` when assembling the prompt payload for the
    annotation backend.
    """
    markers = sc.get.rank_genes_groups_df(adata, group=None, key=marker_key)

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


def parse_annotation_response(content: str) -> Dict[str, str]:
    """
    Parse a JSON or JSON-wrapped model response into cluster annotations.

    Called by `annotation_rule()` after the API response is returned.
    """
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
    mgr: Any,
    parent_id: str,
    groupby: str = None,
    annotation_model: str = "qwen3.5:122b",
    annotation_api_base: str = "http://localhost:11434/v1",
    annotation_api_key: str = "ollama",
    n_annotation_markers: int = 10,
):
    """
    Annotate clusters with a chat-completions backend using top marker genes.

    Called by `run_rule()` in sc_run. It returns a new AnnData object with
    `adata.obs["cell_type"]` and annotation metadata in `adata.uns`, while the
    API key remains runtime-only and is not intended for persistence.
    """
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

    cluster_markers = extract_cluster_markers(
        adata=adata,
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
                {"role": "user", "content": json.dumps(prompt)},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        },
        timeout=120,
    )
    response.raise_for_status()

    annotations = parse_annotation_response(response.json()["choices"][0]["message"]["content"])
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


def create_default_registry() -> RuleRegistry:
    """
    Build the default deterministic single-cell stage registry.

    Called by `get_runner()` in sc_run when the shared pipeline manager is
    constructed.
    """
    registry = RuleRegistry()
    registry.register(Rule("scrublet", ["raw"], scrublet_rule))
    registry.register(Rule("qc", ["scrublet"], qc_filter_rule))
    registry.register(Rule("normalize", ["qc"], normalize_rule))
    registry.register(Rule("hvg", ["normalize"], hvg_rule))
    registry.register(Rule("batch_correct", ["hvg"], batch_correct_rule))
    registry.register(Rule("scale", ["batch_correct"], scale_rule))
    registry.register(Rule("pca", ["scale"], pca_rule))
    registry.register(Rule("neighbors", ["pca"], neighbors_rule))
    registry.register(Rule("umap", ["neighbors"], umap_rule))
    registry.register(Rule("cluster", ["umap"], cluster_rule, virtual=True))
    registry.register(Rule("markers", ["cluster"], markers_rule))
    registry.register(Rule("annotation", ["markers"], annotation_rule))
    return registry
