"""DAG state, cache identity, and persistence for the single-cell pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Sequence
from typing import Any, Dict, Optional

import networkx as nx
import scanpy as sc


DEFAULT_STORAGE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "workspace", "sc_dag_v4")
)
GRAPH_SCHEMA_VERSION = 2

RUNTIME_ONLY_PARAM_KEYS = {"annotation_api_key"}
LEGACY_PARAM_DEFAULTS = {
    "neighbors": {
        "integration_method": "none",
        "integration_batch_key": None,
    }
}


def _json_ready(value: Any) -> Any:
    """Convert values to a stable JSON-friendly representation for hashing and persistence."""
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    return value


def _sanitize_params(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Remove runtime-only data such as API keys before storing or hashing parameters."""
    params = params or {}
    return {key: _json_ready(value) for key, value in params.items() if key not in RUNTIME_ONLY_PARAM_KEYS}


def _saved_param(action: str, params: Dict[str, Any], key: str) -> Any:
    """Read a saved parameter while supplying defaults for older DAG schemas."""
    return params.get(key, LEGACY_PARAM_DEFAULTS.get(action, {}).get(key))


def file_fingerprint(path: Optional[str]) -> Dict[str, Any]:
    """Describe the current on-disk identity of a raw input path for cache reuse decisions."""
    if not path:
        raise ValueError("Lineage error: data_path is required when loading raw data.")

    abs_path = os.path.abspath(path)
    try:
        if os.path.isdir(abs_path):
            expected_files = [
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
            for file_name in expected_files:
                file_path = os.path.join(abs_path, file_name)
                if os.path.exists(file_path):
                    file_stat = os.stat(file_path)
                    files.append(
                        {
                            "name": file_name,
                            "size": file_stat.st_size,
                            "mtime_ns": file_stat.st_mtime_ns,
                        }
                    )
            return {"path": abs_path, "kind": "directory", "files": files}

        stat = os.stat(abs_path)
        return {
            "path": abs_path,
            "kind": "file",
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    except OSError:
        return {"path": abs_path}


def _raw_hash_payload(mgr: "SCStateManager", params: Dict[str, Any]) -> Dict[str, Any]:
    """Build the stable raw-node identity payload."""
    data_paths = params.get("data_paths")
    data_path = params.get("data_path")
    if data_paths:
        sample_ids = params.get("sample_ids") or [f"sample_{idx + 1}" for idx in range(len(data_paths))]
        return {
            "rule": "raw",
            "source": [
                {
                    "sample_id": sample_id,
                    "fingerprint": file_fingerprint(path),
                }
                for path, sample_id in zip(data_paths, sample_ids)
            ],
            "sample_key": params.get("sample_key"),
            "join": params.get("multi_sample_join"),
        }

    if not data_path:
        raw_nodes = [node_id for node_id, attr in mgr.graph.nodes(data=True) if attr.get("action") == "raw"]
        if len(raw_nodes) == 1:
            data_path = mgr.graph.nodes[raw_nodes[0]].get("params", {}).get("data_path")
    payload = {"rule": "raw", "source": file_fingerprint(data_path)}
    if params.get("sample_id") is not None:
        payload.update(
            {
                "sample_id": params.get("sample_id"),
                "sample_key": params.get("sample_key"),
            }
        )
    return payload


def compute_step_hash(
    mgr: "SCStateManager",
    rule_name: str,
    parent_hash: Any,
    all_params: Dict[str, Any],
) -> str:
    """Compute the cache identity for one pipeline step."""
    if rule_name == "raw":
        data = _raw_hash_payload(mgr, _sanitize_params(all_params))
    else:
        if mgr.registry is None:
            raise ValueError("Cannot compute a non-raw hash without an attached rule registry.")
        rule = mgr.registry.get(rule_name)
        relevant_params = {
            key: value
            for key, value in _sanitize_params(all_params).items()
            if key in rule.param_keys
        }
        if isinstance(parent_hash, Sequence) and not isinstance(parent_hash, (str, bytes)):
            data = {
                "parent_hashes": list(parent_hash),
                "rule": rule_name,
                "params": relevant_params,
            }
        else:
            # Keep the legacy single-parent payload stable so existing
            # single-sample cache entries remain reusable.
            data = {"parent_hash": parent_hash, "rule": rule_name, "params": relevant_params}

    return hashlib.md5(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()


class SCStateManager:
    """Own the pipeline DAG, cached AnnData objects, and persistence on disk."""

    def __init__(self, storage_dir: str = DEFAULT_STORAGE_DIR, registry: Any = None):
        """Load the persistent graph and objects used by `SingleCellPipelineRunner`."""
        self.graph = nx.DiGraph()
        self.objects: Dict[str, Any] = {}
        self.registry = registry
        self.hash_index: Dict[str, str] = {}
        self.storage_dir = storage_dir
        self.object_dir = os.path.join(self.storage_dir, "objects")
        self.graph_path = os.path.join(self.storage_dir, "graph.json")
        self.active_node_id: Optional[str] = None
        self._dirty_objects: set[str] = set()
        self.load()

    def _new_id(self) -> str:
        return str(uuid.uuid4())[:8]

    def register_new_object(
        self,
        adata: Any,
        parent_id: Optional[Any],
        action: str,
        params: Dict[str, Any],
        hash_val: Optional[str] = None,
        result_key: Optional[str] = None,
        is_virtual: bool = False,
        parent_ids: Optional[Sequence[str]] = None,
        effective_params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store an immutable snapshot with zero, one, or many ordered parents."""
        if parent_ids is not None and parent_id is not None:
            raise ValueError("Specify parent_id or parent_ids, not both.")
        parent_value = parent_ids if parent_ids is not None else parent_id
        if parent_value is None:
            ordered_parents: list[str] = []
        elif isinstance(parent_value, str):
            ordered_parents = [parent_value]
        else:
            ordered_parents = list(parent_value)
        if len(ordered_parents) != len(set(ordered_parents)):
            raise ValueError("A DAG node cannot list the same parent more than once.")
        missing = [item for item in ordered_parents if item not in self.graph.nodes]
        if missing:
            raise ValueError(f"Parent node(s) not found: {missing}")

        obj_ref = f"obj_{self._new_id()}"
        self.objects[obj_ref] = adata
        self._dirty_objects.add(obj_ref)

        node_id = f"node_{self._new_id()}"
        self.graph.add_node(
            node_id,
            hash=hash_val,
            action=action,
            params=_sanitize_params(params),
            obj_ref=obj_ref,
            is_virtual=is_virtual,
            result_key=result_key,
            shape=list(adata.shape),
            parent_ids=ordered_parents,
            effective_params=_sanitize_params(effective_params) if effective_params else None,
        )
        if hash_val is not None:
            self.hash_index[hash_val] = node_id
        for item in ordered_parents:
            self.graph.add_edge(item, node_id)
        self.save()
        return node_id

    def get_object(self, node_id: str) -> Any:
        """Fetch the AnnData object for a DAG node."""
        if node_id not in self.graph.nodes:
            raise ValueError(f"Node {node_id} not found.")
        obj_ref = self.graph.nodes[node_id].get("obj_ref")
        return self.objects[obj_ref]

    def update_effective_params(self, node_id: str, params: Optional[Dict[str, Any]]) -> None:
        """Record the latest sanitized request that activated a cached node."""
        if not params:
            return
        if node_id not in self.graph.nodes:
            raise ValueError(f"Node {node_id} not found.")
        sanitized = _sanitize_params(params)
        if self.graph.nodes[node_id].get("effective_params") == sanitized:
            return
        self.graph.nodes[node_id]["effective_params"] = sanitized
        self.save()

    def save(self) -> None:
        """Write graph metadata and newly registered objects to the cache directory."""
        os.makedirs(self.object_dir, exist_ok=True)

        for obj_ref in sorted(self._dirty_objects):
            object_path = os.path.join(self.object_dir, f"{obj_ref}.h5ad")
            self.objects[obj_ref].write_h5ad(object_path)
        self._dirty_objects.clear()

        graph_data = {
            "schema_version": GRAPH_SCHEMA_VERSION,
            "nodes": [
                {"id": node_id, **_json_ready(attr)}
                for node_id, attr in self.graph.nodes(data=True)
            ],
            "edges": [[src, dst] for src, dst in self.graph.edges()],
            "active_node_id": self.active_node_id,
        }

        os.makedirs(self.storage_dir, exist_ok=True)
        tmp_path = f"{self.graph_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(graph_data, handle, indent=2, sort_keys=True)
        os.replace(tmp_path, self.graph_path)

    def load(self) -> None:
        """Load the saved graph, including files that predate `active_node_id`."""
        if not os.path.exists(self.graph_path):
            return

        with open(self.graph_path, encoding="utf-8") as handle:
            graph_data = json.load(handle)

        self.graph.clear()
        self.objects.clear()
        self.hash_index.clear()
        self._dirty_objects.clear()
        self.active_node_id = graph_data.get("active_node_id")

        for node in graph_data.get("nodes", []):
            node = dict(node)
            node_id = node.pop("id")
            self.graph.add_node(node_id, **node)
            node_hash = node.get("hash")
            if node_hash is not None:
                self.hash_index[node_hash] = node_id

        self.graph.add_edges_from(graph_data.get("edges", []))

        # Older graphs only persisted edge pairs. Preserve their order as read
        # and materialize parent metadata for the multi-parent-aware API.
        for node_id in self.graph.nodes:
            attr = self.graph.nodes[node_id]
            if "parent_ids" not in attr:
                attr["parent_ids"] = list(self.graph.predecessors(node_id))

        obj_refs = {
            attr.get("obj_ref")
            for _, attr in self.graph.nodes(data=True)
            if attr.get("obj_ref") is not None
        }
        for obj_ref in obj_refs:
            object_path = os.path.join(self.object_dir, f"{obj_ref}.h5ad")
            if os.path.exists(object_path):
                self.objects[obj_ref] = sc.read_h5ad(object_path)

    def dependency_chain(self, target_stage: str) -> list[str]:
        """Return the raw-to-target stages used by cache lookup and the runner."""
        if target_stage == "raw":
            return ["raw"]
        if self.registry is None:
            raise ValueError("Cannot resolve dependencies without an attached rule registry.")

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

    def find_node_smart(self, target_stage: str, **user_params: Any) -> tuple[Any, str]:
        """Find an exact or stage-local cache match for the pipeline runner."""
        strict_id = self.find_node_strict(target_stage, **user_params)
        if strict_id:
            return strict_id, "exact_match"

        if self.registry is None or target_stage not in self.registry.rules:
            return None, "no_match"

        target_rule = self.registry.get(target_stage)
        candidates = []
        sanitized_user_params = _sanitize_params(user_params)
        for node_id, attr in self.graph.nodes(data=True):
            if attr.get("action") != target_stage:
                continue

            node_params = attr.get("params", {})
            for key in target_rule.param_keys:
                user_val = sanitized_user_params.get(key)
                if user_val is not None and str(user_val) != str(_saved_param(target_stage, node_params, key)):
                    break
            else:
                candidates.append(node_id)

        if not candidates:
            return None, "no_match"
        if len(candidates) == 1:
            return candidates[0], "fuzzy_match"
        return candidates, "ambiguous"

    def find_node_strict(self, target_stage: str, **full_params: Any) -> Optional[str]:
        """Find the node whose complete lineage matches the runner's parameters."""
        chain = self.dependency_chain(target_stage)
        current_hash = "init"
        for stage in chain:
            current_hash = compute_step_hash(self, stage, current_hash, full_params)
        exact_node_id = self.hash_index.get(current_hash)
        if exact_node_id:
            return exact_node_id

        # Cache schemas evolve as optional parameters and pipeline stages are
        # added. Fall back to structural matching so missing and null optional
        # values remain compatible without reusing nodes from another source or
        # an obsolete dependency chain.
        sanitized_params = _sanitize_params(full_params)
        compatible_nodes = []
        for node_id, attr in self.graph.nodes(data=True):
            if attr.get("action") != target_stage:
                continue
            try:
                lineage = self.lineage_to_node(node_id)
            except ValueError:
                continue
            if [self.graph.nodes[item].get("action") for item in lineage] != chain:
                continue

            raw_node_id = lineage[0]
            try:
                requested_raw_hash = compute_step_hash(self, "raw", "init", sanitized_params)
            except (OSError, ValueError):
                continue
            if self.graph.nodes[raw_node_id].get("hash") != requested_raw_hash:
                continue

            for lineage_node_id in lineage[1:]:
                node_meta = self.graph.nodes[lineage_node_id]
                rule = self.registry.get(node_meta["action"])
                node_params = node_meta.get("params", {})
                if any(
                    _saved_param(node_meta["action"], node_params, key) != sanitized_params.get(key)
                    for key in rule.param_keys
                ):
                    break
            else:
                compatible_nodes.append(node_id)

        if len(compatible_nodes) == 1:
            return compatible_nodes[0]
        if self.active_node_id in compatible_nodes:
            return self.active_node_id
        return None

    def lineage_to_node(self, node_id: str) -> list[str]:
        """Return the single ordered lineage ending at ``node_id``."""
        lineage = []
        current_id: Optional[str] = node_id
        while current_id is not None:
            lineage.append(current_id)
            predecessors = self.parent_ids(current_id)
            if len(predecessors) > 1:
                raise ValueError(
                    f"Node {current_id} has multiple parents; use ancestry_to_node() instead."
                )
            current_id = predecessors[0] if predecessors else None
        return list(reversed(lineage))

    def parent_ids(self, node_id: str) -> list[str]:
        """Return a node's parents in their semantically significant order."""
        if node_id not in self.graph.nodes:
            raise ValueError(f"Node {node_id} not found.")
        saved = self.graph.nodes[node_id].get("parent_ids")
        if saved is not None:
            return list(saved)
        return list(self.graph.predecessors(node_id))

    def ancestry_to_node(self, node_id: str) -> list[str]:
        """Return deterministic parent-first ancestry for linear or merged DAGs."""
        if node_id not in self.graph.nodes:
            raise ValueError(f"Node {node_id} not found.")
        ordered: list[str] = []
        visited: set[str] = set()

        def visit(current_id: str) -> None:
            if current_id in visited:
                return
            for parent_id in self.parent_ids(current_id):
                visit(parent_id)
            visited.add(current_id)
            ordered.append(current_id)

        visit(node_id)
        return ordered

    def ancestors_including_self(self, node_id: str) -> set[str]:
        """Return the node and every ancestor queried through NetworkX."""
        return self.ancestors(node_id) | {node_id}

    def raw_ancestor(self, node_id: str) -> Optional[str]:
        """Return the sole raw ancestor, or ``None`` for a merged lineage."""
        if node_id not in self.graph.nodes:
            return None
        raw_nodes = self.raw_ancestors(node_id)
        return raw_nodes[0] if len(raw_nodes) == 1 else None

    def raw_ancestors(self, node_id: str) -> list[str]:
        """Return every raw root contributing to a node, in parent order."""
        if node_id not in self.graph.nodes:
            return []
        return [
            item
            for item in self.ancestry_to_node(node_id)
            if self.graph.nodes[item].get("action") == "raw"
        ]

    def ancestors(self, node_id: str) -> set[str]:
        """Return all ancestors using NetworkX."""
        return nx.ancestors(self.graph, node_id)

    def descendants(self, node_id: str) -> set[str]:
        """Return all descendants using NetworkX."""
        return nx.descendants(self.graph, node_id)

    def shortest_path(self, source: str, target: str) -> list[str]:
        """Return the shortest directed path using NetworkX."""
        return nx.shortest_path(self.graph, source=source, target=target)
