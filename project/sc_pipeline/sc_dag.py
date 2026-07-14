"""DAG state, cache identity, and persistence for the single-cell pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections import deque
from typing import Any, Dict, Iterable, Optional

import scanpy as sc


DEFAULT_STORAGE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "workspace", "sc_dag_v4")
)

RUNTIME_ONLY_PARAM_KEYS = {"annotation_api_key"}


def _json_ready(value: Any) -> Any:
    """Convert values to a stable JSON-friendly representation for hashing and persistence."""
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    return value


def _sanitize_params(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Remove runtime-only data such as API keys before storing or hashing parameters."""
    params = params or {}
    return {
        key: _json_ready(value)
        for key, value in params.items()
        if key not in RUNTIME_ONLY_PARAM_KEYS
    }


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
    return {"rule": "raw", "source": file_fingerprint(data_path)}


def compute_step_hash(
    mgr: "SCStateManager",
    rule_name: str,
    parent_hash: str,
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
        data = {"parent_hash": parent_hash, "rule": rule_name, "params": relevant_params}

    return hashlib.md5(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()


class SCStateManager:
    """Own the pipeline DAG, cached AnnData objects, and persistence on disk."""

    def __init__(self, storage_dir: str = DEFAULT_STORAGE_DIR, registry: Any = None):
        """
        Manage the persistent cache for pipeline nodes.

        Called by `SingleCellPipelineRunner` in sc_run. It stores DAG metadata,
        object references, cache hashes, and the active node for the current process.
        """
        self.graph = SimpleDiGraph()
        self.objects: Dict[str, Any] = {}
        self.registry = registry
        self.hash_index: Dict[str, str] = {}
        self.storage_dir = storage_dir
        self.object_dir = os.path.join(self.storage_dir, "objects")
        self.graph_path = os.path.join(self.storage_dir, "graph.json")
        self.active_node_id: Optional[str] = None
        self._dirty_objects: set[str] = set()
        self.load()

    def set_registry(self, registry: Any) -> None:
        """Attach the rule registry after construction to avoid circular imports."""
        self.registry = registry

    def _new_id(self) -> str:
        return str(uuid.uuid4())[:8]

    def _register_node(self, node_id: str, hash_val: Optional[str], **attr: Any) -> None:
        self.graph.add_node(node_id, hash=hash_val, **attr)
        if hash_val is not None:
            self.hash_index[hash_val] = node_id

    def register_new_object(
        self,
        adata: Any,
        parent_id: Optional[str],
        action: str,
        params: Dict[str, Any],
        hash_val: Optional[str] = None,
        result_key: Optional[str] = None,
        is_virtual: bool = False,
    ) -> str:
        """
        Register an immutable node backed by its own AnnData object.

        Called by `run_rule()` and `register_raw()` in sc_run. It creates a new
        object ref, stores sanitized parameters, updates the hash index, and saves
        the DAG state.
        """
        obj_ref = f"obj_{self._new_id()}"
        self.objects[obj_ref] = adata
        self._dirty_objects.add(obj_ref)

        node_id = f"node_{self._new_id()}"
        self._register_node(
            node_id,
            hash_val,
            action=action,
            params=_sanitize_params(params),
            obj_ref=obj_ref,
            is_virtual=is_virtual,
            result_key=result_key,
            shape=list(adata.shape),
        )
        if parent_id:
            self.graph.add_edge(parent_id, node_id)
        self.save()
        return node_id

    def register_virtual_node(
        self,
        adata: Any,
        parent_id: str,
        action: str,
        params: Dict[str, Any],
        result_key: str,
        hash_val: Optional[str] = None,
    ) -> str:
        """
        Register a lineage node that is logically virtual but still keeps its own object snapshot.

        Called by `run_rule()` for rules such as clustering. The `is_virtual`
        metadata is preserved for DAG rendering, while the object copy keeps
        parent results immutable.
        """
        return self.register_new_object(
            adata=adata,
            parent_id=parent_id,
            action=action,
            params=params,
            hash_val=hash_val,
            result_key=result_key,
            is_virtual=True,
        )

    def get_object(self, node_id: str) -> Any:
        """Fetch the AnnData object for a DAG node."""
        if node_id not in self.graph.nodes:
            raise ValueError(f"Node {node_id} not found.")
        obj_ref = self.graph.nodes[node_id].get("obj_ref")
        return self.objects[obj_ref]

    def save(self) -> None:
        """
        Persist the graph and any new or changed AnnData objects to disk.

        Called after node registration and from tests that want an explicit save.
        Object writes are tracked so repeated graph saves do not rewrite every
        cached AnnData file.
        """
        os.makedirs(self.object_dir, exist_ok=True)

        for obj_ref in sorted(self._dirty_objects):
            object_path = os.path.join(self.object_dir, f"{obj_ref}.h5ad")
            self.objects[obj_ref].write_h5ad(object_path)
        self._dirty_objects.clear()

        graph_data = {
            "nodes": [
                {"id": node_id, **self._json_safe_attrs(attr)}
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
        """
        Restore the saved DAG and object cache from disk if present.

        Called from `__init__`. It preserves compatibility with the prior graph
        format by accepting missing `active_node_id` and tuple-like values.
        """
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

        obj_refs = {
            attr.get("obj_ref")
            for _, attr in self.graph.nodes(data=True)
            if attr.get("obj_ref") is not None
        }
        for obj_ref in obj_refs:
            object_path = os.path.join(self.object_dir, f"{obj_ref}.h5ad")
            if os.path.exists(object_path):
                self.objects[obj_ref] = sc.read_h5ad(object_path)

    def _json_safe_attrs(self, attrs: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize node attributes for graph JSON persistence."""
        return {key: _json_ready(value) for key, value in attrs.items()}

    def dependency_chain(self, target_stage: str) -> list[str]:
        """
        Return the linear upstream dependency path for a target stage.

        Called by cache lookup, execution planning, and DAG plotting. It walks
        through the rule registry and returns stages starting at `raw`.
        """
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
        """
        Find an exact or partial cached node for a request.

        Called by sc_run before any new execution. It first checks the full
        lineage hash and then falls back to stage-local fuzzy matching using the
        target rule's parameter surface.
        """
        strict_id = self.find_node_strict(target_stage, **user_params)
        if strict_id:
            return strict_id, "exact_match"

        if self.registry is None or not self.registry.has(target_stage):
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
                if user_val is not None and str(user_val) != str(node_params.get(key)):
                    break
            else:
                candidates.append(node_id)

        if not candidates:
            return None, "no_match"
        if len(candidates) == 1:
            return candidates[0], "fuzzy_match"
        return candidates, "ambiguous"

    def find_node_strict(self, target_stage: str, **full_params: Any) -> Optional[str]:
        """
        Resolve the full lineage hash for a request and return the cached node ID.

        Called by sc_run during exact-match lookup and ancestor selection.
        """
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
            lineage = self.lineage_to_node(node_id)
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
                    node_params.get(key) != sanitized_params.get(key)
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
            predecessors = list(self.graph.predecessors(current_id))
            if len(predecessors) > 1:
                return []
            current_id = predecessors[0] if predecessors else None
        return list(reversed(lineage))

    def ancestors_including_self(self, node_id: str) -> Iterable[str]:
        """Return the node lineage from all ancestors through the node itself."""
        return list(self.ancestors(node_id)) + [node_id]

    def raw_ancestor(self, node_id: str) -> Optional[str]:
        """Return the upstream raw node for a lineage."""
        if node_id not in self.graph.nodes:
            return None
        lineage = self.ancestors_including_self(node_id)
        for ancestor_id in reversed(lineage):
            if self.graph.nodes[ancestor_id].get("action") == "raw":
                return ancestor_id
        return node_id

    def ancestors(self, node_id: str) -> set[str]:
        """Return all ancestors of a node in the DAG."""
        visited = set()
        stack = list(self.graph.predecessors(node_id))
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            stack.extend(self.graph.predecessors(current))
        return visited

    def descendants(self, node_id: str) -> set[str]:
        """Return all descendants of a node in the DAG."""
        visited = set()
        stack = list(self.graph.successors(node_id))
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            stack.extend(self.graph.successors(current))
        return visited

    def shortest_path(self, source: str, target: str) -> list[str]:
        """Return the shortest directed path between two DAG nodes."""
        queue = deque([[source]])
        seen = {source}
        while queue:
            path = queue.popleft()
            node_id = path[-1]
            if node_id == target:
                return path
            for child_id in self.graph.successors(node_id):
                if child_id not in seen:
                    seen.add(child_id)
                    queue.append(path + [child_id])
        raise ValueError(f"No path found from {source} to {target}.")


class NodeView:
    """Minimal node-view API compatible with this pipeline's usage patterns."""

    def __init__(self, graph: "SimpleDiGraph"):
        self._graph = graph

    def __call__(self, data: bool = False):
        if data:
            return list(self._graph._nodes.items())
        return list(self._graph._nodes.keys())

    def __iter__(self):
        return iter(self._graph._nodes.keys())

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._graph._nodes

    def __getitem__(self, node_id: str) -> Dict[str, Any]:
        return self._graph._nodes[node_id]

    def __len__(self) -> int:
        return len(self._graph._nodes)


class SimpleDiGraph:
    """Small directed-graph implementation for cache metadata and plotting."""

    def __init__(self) -> None:
        self._nodes: Dict[str, Dict[str, Any]] = {}
        self._succ: Dict[str, set[str]] = {}
        self._pred: Dict[str, set[str]] = {}
        self.nodes = NodeView(self)

    def add_node(self, node_id: str, **attr: Any) -> None:
        self._nodes.setdefault(node_id, {})
        self._succ.setdefault(node_id, set())
        self._pred.setdefault(node_id, set())
        self._nodes[node_id].update(attr)

    def add_edge(self, src: str, dst: str) -> None:
        self.add_node(src)
        self.add_node(dst)
        self._succ[src].add(dst)
        self._pred[dst].add(src)

    def add_edges_from(self, edges: Iterable[Iterable[str]]) -> None:
        for src, dst in edges:
            self.add_edge(src, dst)

    def clear(self) -> None:
        self._nodes.clear()
        self._succ.clear()
        self._pred.clear()

    def edges(self):
        return [(src, dst) for src, dsts in self._succ.items() for dst in dsts]

    def predecessors(self, node_id: str):
        return list(self._pred.get(node_id, set()))

    def successors(self, node_id: str):
        return list(self._succ.get(node_id, set()))
