"""Pipeline orchestration for the single-cell TaskWeaver plugin."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Dict, Optional

from project.sc_pipeline.sc_dag import DEFAULT_STORAGE_DIR, SCStateManager, compute_step_hash
from project.sc_pipeline.sc_io import SingleCellIO
from project.sc_pipeline.sc_rules import RuleRegistry, create_default_registry


@dataclass
class PipelineRequest:
    """Hold the normalized target and parameters used throughout one run."""

    target_stage: str
    params: Dict[str, Any]


class SingleCellPipelineRunner:
    """Coordinate cache lookup and stage execution for the TaskWeaver plugin."""

    def __init__(
        self,
        ctx: Any,
        config: Optional[Dict[str, Any]] = None,
        storage_dir: Optional[str] = None,
        manager: Optional[SCStateManager] = None,
        registry: Optional[RuleRegistry] = None,
    ):
        self.ctx = ctx
        self.config = config or {}
        self.io = SingleCellIO(ctx)
        self.registry = registry or create_default_registry()
        self.manager = manager or SCStateManager(
            storage_dir=storage_dir or self.config.get("storage_dir", DEFAULT_STORAGE_DIR),
            registry=self.registry,
        )
        self.manager.registry = self.registry

    def execute(self, target_stage: str, **kwargs: Any):
        """Run or reuse a pipeline result, then let `SingleCellIO` publish it."""
        request = self.normalize_request(target_stage=target_stage, **kwargs)
        node_id = self.resolve_request(request)
        return self.io.visualize_result(self.manager, node_id, request.target_stage)

    def normalize_request(self, target_stage: str, **kwargs: Any) -> PipelineRequest:
        """Validate plugin input and build the parameter set used by `execute`."""
        mgr = self.manager
        supplied_data_paths = self.io.coerce_optional_list(kwargs.get("data_paths"))
        if kwargs.get("data_path") is None and not supplied_data_paths:
            inherited_params = self.infer_active_params()
            if inherited_params:
                kwargs = {**inherited_params, **kwargs}

        data_path = kwargs.get("data_path")
        data_paths = self.io.coerce_optional_list(kwargs.get("data_paths"))
        sample_ids = self.io.coerce_optional_list(kwargs.get("sample_ids"))
        sample_key = kwargs.get("sample_key", "sample")
        multi_sample_join = kwargs.get("multi_sample_join", "inner")
        sample_overrides = kwargs.get("sample_overrides") or {}
        if isinstance(sample_overrides, str):
            sample_overrides = json.loads(sample_overrides)
        if not isinstance(sample_overrides, Mapping):
            raise ValueError("sample_overrides must be a mapping keyed by sample ID.")
        sample_overrides = {str(key): value for key, value in sample_overrides.items()}
        scrublet_batch_key = kwargs.get("scrublet_batch_key")
        hvg_batch_key = kwargs.get("hvg_batch_key")
        combat_key = kwargs.get("combat_key")
        integration_method = (kwargs.get("integration_method") or "auto").lower()
        integration_batch_key = kwargs.get("integration_batch_key")
        legacy_batch_method = (kwargs.get("batch_correction_method") or "none").lower()

        if target_stage == "umap" and kwargs.get("resolution", 0.5) != 0.5:
            self.ctx.log(
                "info",
                "sc_pipeline",
                "Interpreting target_stage='umap' with a non-default resolution as target_stage='cluster'.",
            )
            target_stage = "cluster"

        if data_paths:
            if data_path is not None:
                raise ValueError("Specify either data_path or data_paths, not both.")
            if sample_ids is None:
                sample_ids = [f"sample_{idx + 1}" for idx in range(len(data_paths))]
            sample_ids = [str(item) for item in sample_ids]
            if len(sample_ids) != len(data_paths):
                raise ValueError("sample_ids must have the same length as data_paths.")
            if len(sample_ids) != len(set(sample_ids)):
                raise ValueError("sample_ids must be unique.")
            if hvg_batch_key is None:
                hvg_batch_key = sample_key
        elif data_path is None:
            source_params = self.infer_active_source_params()
            if source_params:
                data_path = source_params.get("data_path")
                data_paths = source_params.get("data_paths")
                sample_ids = source_params.get("sample_ids")
                sample_key = source_params.get("sample_key", sample_key)
                multi_sample_join = source_params.get("multi_sample_join", multi_sample_join)
            if data_paths:
                if hvg_batch_key is None:
                    hvg_batch_key = sample_key
        is_multi_sample = bool(data_paths and len(data_paths) > 1)
        if data_paths and len(data_paths) < 2:
            raise ValueError("data_paths requires at least two inputs; use data_path for one sample.")
        if target_stage == "concat" and not is_multi_sample:
            raise ValueError("target_stage='concat' requires at least two data_paths.")
        if sample_overrides and not is_multi_sample:
            raise ValueError("sample_overrides is only supported with data_paths.")
        self.validate_sample_overrides(sample_overrides, sample_ids or [])
        if integration_method not in {"auto", "none", "combat", "bbknn"}:
            raise ValueError("integration_method must be 'auto', 'none', 'combat', or 'bbknn'.")
        if legacy_batch_method not in {"none", "combat"}:
            raise ValueError("batch_correction_method must be 'none' or 'combat'.")
        if legacy_batch_method == "combat":
            if integration_method not in {"auto", "combat"}:
                raise ValueError(
                    "Conflicting integration choices: batch_correction_method='combat' "
                    f"cannot be combined with integration_method='{integration_method}'."
                )
            integration_method = "combat"
        elif integration_method == "auto":
            integration_method = "bbknn" if is_multi_sample else "none"

        if integration_method == "combat":
            if integration_batch_key and combat_key and integration_batch_key != combat_key:
                raise ValueError("integration_batch_key and combat_key must match when both are provided.")
            combat_key = integration_batch_key or combat_key or (sample_key if is_multi_sample else None)
            batch_correction_method = "combat"
            neighbor_integration_method = "none"
        else:
            batch_correction_method = "none"
            neighbor_integration_method = integration_method
            if combat_key is None and is_multi_sample:
                combat_key = sample_key

        if neighbor_integration_method == "bbknn" and integration_batch_key is None and is_multi_sample:
            integration_batch_key = sample_key

        valid_stages = set(mgr.registry.rules.keys()) | {"raw"}
        if target_stage not in valid_stages:
            raise ValueError(f"Unknown target_stage '{target_stage}'. Valid stages: {sorted(valid_stages)}")

        params = {
            "data_path": data_path,
            "data_paths": data_paths,
            "sample_ids": sample_ids,
            "sample_key": sample_key,
            "multi_sample_join": multi_sample_join,
            "sample_overrides": sample_overrides,
            "scrublet_batch_key": scrublet_batch_key,
            "scrublet_expected_doublet_rate": kwargs.get("scrublet_expected_doublet_rate", 0.05),
            "scrublet_threshold": kwargs.get("scrublet_threshold"),
            "scrublet_n_prin_comps": kwargs.get("scrublet_n_prin_comps", 30),
            "scrublet_filter_doublets": kwargs.get("scrublet_filter_doublets", False),
            "scrublet_skip_on_failure": kwargs.get("scrublet_skip_on_failure", True),
            "qc_min_genes": kwargs.get("qc_min_genes", 200),
            "qc_max_genes": kwargs.get("qc_max_genes", 2500),
            "qc_mt_pct": kwargs.get("qc_mt_pct", 5),
            "min_cells": kwargs.get("min_cells", 3),
            "target_sum": kwargs.get("target_sum", 1e4),
            "n_hvg": kwargs.get("n_hvg", 2000),
            "hvg_flavor": kwargs.get("hvg_flavor", "seurat"),
            "hvg_batch_key": hvg_batch_key,
            "batch_correction_method": batch_correction_method,
            "combat_key": combat_key,
            "max_scale_value": kwargs.get("max_scale_value", 10),
            "regress_out": kwargs.get("regress_out", True),
            "n_comps": kwargs.get("n_comps", 50),
            "n_neighbors": kwargs.get("n_neighbors", 10),
            "n_pcs": kwargs.get("n_pcs", 40),
            "use_rep": kwargs.get("use_rep", "X_pca"),
            "integration_method": neighbor_integration_method,
            "integration_batch_key": integration_batch_key,
            "min_dist": kwargs.get("min_dist", 0.5),
            "spread": kwargs.get("spread", 1.0),
            "resolution": kwargs.get("resolution", 0.5),
            "cluster_method": kwargs.get("cluster_method", "leiden"),
            "groupby": kwargs.get("groupby"),
            "marker_method": kwargs.get("marker_method", "wilcoxon"),
            "n_marker_genes": kwargs.get("n_marker_genes", 25),
            "annotation_model": kwargs.get("annotation_model", "qwen3.5:122b"),
            "annotation_api_base": kwargs.get("annotation_api_base", "http://localhost:11434/v1"),
            "annotation_api_key": kwargs.get("annotation_api_key", "ollama"),
            "n_annotation_markers": kwargs.get("n_annotation_markers", 10),
        }

        for key, value in kwargs.items():
            if key not in params:
                params[key] = value

        return PipelineRequest(target_stage=target_stage, params=params)

    @staticmethod
    def validate_sample_overrides(overrides: Dict[str, Any], sample_ids: list[str]) -> None:
        """Validate per-sample parameters before any branch work starts."""
        allowed = {
            "scrublet_batch_key",
            "scrublet_expected_doublet_rate",
            "scrublet_threshold",
            "scrublet_n_prin_comps",
            "scrublet_filter_doublets",
            "scrublet_skip_on_failure",
            "qc_min_genes",
            "qc_max_genes",
            "qc_mt_pct",
        }
        unknown_samples = sorted(set(overrides) - set(sample_ids))
        if unknown_samples:
            raise ValueError(f"sample_overrides contains unknown sample IDs: {unknown_samples}")
        for sample_id, values in overrides.items():
            if not isinstance(values, Mapping):
                raise ValueError(f"sample_overrides['{sample_id}'] must be a mapping.")
            unknown_keys = sorted(set(values) - allowed)
            if unknown_keys:
                raise ValueError(
                    f"sample_overrides['{sample_id}'] contains unsupported keys: {unknown_keys}"
                )

    def infer_active_params(self) -> Dict[str, Any]:
        """Collect saved parameters from the active DAG lineage for a follow-up call."""
        mgr = self.manager
        active_node_id = getattr(mgr, "active_node_id", None)
        if active_node_id not in mgr.graph.nodes:
            return {}

        effective = mgr.graph.nodes[active_node_id].get("effective_params")
        if effective:
            return dict(effective)

        params: Dict[str, Any] = {}
        for node_id in mgr.ancestry_to_node(active_node_id):
            params.update(mgr.graph.nodes[node_id].get("params", {}))
        return params

    def infer_active_source_params(self) -> Optional[Dict[str, Any]]:
        """Find raw input parameters when `normalize_request` receives no path."""
        mgr = self.manager
        active_node_id = getattr(mgr, "active_node_id", None)
        if active_node_id in mgr.graph.nodes:
            effective = mgr.graph.nodes[active_node_id].get("effective_params")
            if effective:
                return {
                    key: effective.get(key)
                    for key in (
                        "data_path",
                        "data_paths",
                        "sample_ids",
                        "sample_key",
                        "multi_sample_join",
                    )
                    if effective.get(key) is not None
                }
            raw_node_id = mgr.raw_ancestor(active_node_id)
            if raw_node_id is not None:
                return mgr.graph.nodes[raw_node_id].get("params", {})

            raw_nodes = mgr.raw_ancestors(active_node_id)
            if raw_nodes:
                return {
                    "data_paths": [mgr.graph.nodes[item].get("params", {}).get("data_path") for item in raw_nodes],
                    "sample_ids": [mgr.graph.nodes[item].get("params", {}).get("sample_id") for item in raw_nodes],
                    "sample_key": mgr.graph.nodes[raw_nodes[0]].get("params", {}).get("sample_key", "sample"),
                }

        raw_nodes = [node_id for node_id, attr in mgr.graph.nodes(data=True) if attr.get("action") == "raw"]
        if len(raw_nodes) == 1:
            return mgr.graph.nodes[raw_nodes[0]].get("params", {})

        return None

    def resolve_request(self, request: PipelineRequest) -> str:
        """Return the exact cached node or build missing stages for `execute`."""
        if request.params.get("data_paths"):
            return self.resolve_multi_sample_request(request)

        mgr = self.manager
        self.ctx.log(
            "info",
            "sc_pipeline",
            f"Smart search: '{request.target_stage}' with params: {request.params}",
        )
        result_node_id, match_type = mgr.find_node_smart(request.target_stage, **request.params)

        if match_type == "exact_match":
            self.ctx.log("info", "sc_pipeline", f"Exact lineage match found: {result_node_id}")
            return result_node_id

        if match_type == "ambiguous":
            self.ctx.log(
                "info",
                "sc_pipeline",
                "Multiple stage-local cache matches found; rebuilding from the nearest exact lineage ancestor.",
            )

        start_node_id = self.select_nearest_compatible_ancestor(request)
        if not start_node_id:
            start_node_id = self.register_or_reuse_raw_node(request)

        final_node_id = self.ensure(
            target=request.target_stage,
            start_state=start_node_id,
            **request.params,
        )
        return final_node_id

    def resolve_multi_sample_request(self, request: PipelineRequest) -> str:
        """Build or reuse independent sample branches and their merge."""
        params = request.params
        sample_ids = list(params["sample_ids"])
        data_paths = list(params["data_paths"])
        self.ctx.log(
            "info",
            "sc_pipeline",
            f"Resolving {len(sample_ids)} per-sample branches for '{request.target_stage}'.",
        )

        raw_nodes = [
            self.register_or_reuse_sample_raw(path, sample_id, params)
            for path, sample_id in zip(data_paths, sample_ids)
        ]
        if request.target_stage == "raw":
            return self.run_concat(raw_nodes, params, source_stage="raw", preview=True)

        scrublet_nodes = []
        for node_id, sample_id in zip(raw_nodes, sample_ids):
            branch_params = self.sample_stage_params(params, sample_id)
            scrublet_nodes.append(
                self.run_rule("scrublet", node_id, effective_params=params, **branch_params)
            )
        if request.target_stage == "scrublet":
            return self.run_concat(scrublet_nodes, params, source_stage="scrublet", preview=True)

        qc_nodes = []
        for node_id, sample_id in zip(scrublet_nodes, sample_ids):
            branch_params = self.sample_stage_params(params, sample_id)
            branch_params.pop("min_cells", None)
            branch_params["_filter_genes"] = False
            qc_node = self.run_rule("qc", node_id, effective_params=params, **branch_params)
            if self.manager.get_object(qc_node).n_obs == 0:
                raise ValueError(f"Sample '{sample_id}' has no cells after QC.")
            qc_nodes.append(qc_node)

        concat_node = self.run_concat(qc_nodes, params, source_stage="qc", preview=False)
        if request.target_stage in {"qc", "concat"}:
            return concat_node

        combined_stages = [
            "normalize",
            "hvg",
            "batch_correct",
            "scale",
            "pca",
            "neighbors",
            "umap",
            "cluster",
            "markers",
            "annotation",
        ]
        current_node = concat_node
        for stage in combined_stages:
            current_node = self.run_rule(stage, current_node, effective_params=params, **params)
            if stage == request.target_stage:
                return current_node
        raise ValueError(f"Unable to resolve multi-sample target '{request.target_stage}'.")

    def sample_stage_params(self, params: Dict[str, Any], sample_id: str) -> Dict[str, Any]:
        """Apply one sample's override mapping on top of global parameters."""
        return {**params, **params.get("sample_overrides", {}).get(sample_id, {})}

    def register_or_reuse_sample_raw(
        self,
        data_path: str,
        sample_id: str,
        effective_params: Dict[str, Any],
    ) -> str:
        """Register one raw sample rather than an already concatenated dataset."""
        raw_params = {
            "data_path": data_path,
            "sample_id": sample_id,
            "sample_key": effective_params.get("sample_key", "sample"),
        }
        raw_hash = compute_step_hash(self.manager, "raw", "init", raw_params)
        if raw_hash in self.manager.hash_index:
            return self.manager.hash_index[raw_hash]
        self.ctx.log("info", "sc_pipeline", f"Loading sample '{sample_id}' from {data_path}.")
        adata = self.io.read_input_data(data_path)
        adata.var_names_make_unique()
        adata.obs[raw_params["sample_key"]] = sample_id
        return self.manager.register_new_object(
            adata=adata,
            parent_id=None,
            action="raw",
            params=raw_params,
            hash_val=raw_hash,
            effective_params=effective_params,
        )

    def run_concat(
        self,
        parent_ids: list[str],
        params: Dict[str, Any],
        source_stage: str,
        preview: bool,
    ) -> str:
        """Create a preview or canonical multi-parent concat node."""
        concat_params = {
            "sample_ids": params["sample_ids"],
            "sample_key": params["sample_key"],
            "multi_sample_join": params["multi_sample_join"],
            "source_stage": source_stage,
            "preview": preview,
        }
        if not preview:
            concat_params["min_cells"] = params["min_cells"]
        return self.run_rule(
            "concat",
            parent_ids,
            effective_params=params,
            **concat_params,
        )

    def select_nearest_compatible_ancestor(self, request: PipelineRequest) -> Optional[str]:
        """Find the nearest exact cached ancestor before `resolve_request` rebuilds."""
        mgr = self.manager
        self.ctx.log("info", "sc_pipeline", "Locating nearest valid cached ancestor.")
        ancestor_chain = reversed(mgr.dependency_chain(request.target_stage)[:-1])
        for stage in ancestor_chain:
            node_id = mgr.find_node_strict(stage, **request.params)
            if node_id:
                self.ctx.log("info", "sc_pipeline", f"Anchor found: {stage.upper()} ({node_id})")
                return node_id
        return None

    def register_or_reuse_raw_node(self, request: PipelineRequest) -> str:
        """Reuse raw data or load it when `resolve_request` has no cached ancestor."""
        mgr = self.manager
        params = request.params
        target_raw_hash = compute_step_hash(mgr, "raw", "init", params)

        if target_raw_hash in mgr.hash_index:
            return mgr.hash_index[target_raw_hash]

        data_path = params.get("data_path")
        if data_path:
            self.ctx.log("info", "sc_pipeline", f"Loading data from {data_path}.")
            adata = self.io.read_input_data(data_path)
            return self.register_raw(adata, path=data_path)

        raise ValueError("No data found and no valid parent state exists. Provide data_path.")

    def build_execution_path(self, target: str, start_stage: str) -> list[str]:
        """List the stage names that `ensure` must run after a cached stage."""
        if target == "raw":
            return []
        dependency_chain = self.manager.dependency_chain(target)
        try:
            start_index = dependency_chain.index(start_stage)
        except ValueError as exc:
            raise ValueError(f"Start stage '{start_stage}' is not compatible with target '{target}'.") from exc
        return dependency_chain[start_index + 1 :]

    def run_rule(
        self,
        rule_name: str,
        parent_id: Any,
        effective_params: Optional[Dict[str, Any]] = None,
        **params: Any,
    ) -> str:
        """Run one rule with one or many ordered parents, or reuse its hash."""
        mgr = self.manager
        if parent_id is None:
            parent_ids: list[str] = []
        elif isinstance(parent_id, str):
            parent_ids = [parent_id]
        elif isinstance(parent_id, Sequence):
            parent_ids = list(parent_id)
        else:
            raise ValueError("parent_id must be a node ID or an ordered sequence of node IDs.")
        if not parent_ids:
            parent_hash: Any = "init"
        elif len(parent_ids) == 1:
            parent_hash = mgr.graph.nodes[parent_ids[0]].get("hash", "init")
        else:
            parent_hash = [mgr.graph.nodes[item].get("hash", "init") for item in parent_ids]
        hash_val = compute_step_hash(mgr, rule_name, parent_hash, params)

        if hash_val in mgr.hash_index:
            node_id = mgr.hash_index[hash_val]
            mgr.update_effective_params(node_id, effective_params)
            return node_id

        rule = mgr.registry.get(rule_name)
        rule_params = {key: value for key, value in params.items() if key in rule.call_param_keys}
        rule_parent = parent_ids[0] if len(parent_ids) == 1 else parent_ids
        adata, result_type, *result_keys = rule.func(mgr, rule_parent, **rule_params)
        result_key = result_keys[0] if result_keys else None

        if result_type == "new_object":
            return mgr.register_new_object(
                adata=adata,
                parent_id=None,
                parent_ids=parent_ids,
                action=rule_name,
                params=rule_params,
                hash_val=hash_val,
                result_key=result_key,
                effective_params=effective_params,
            )
        if result_type == "virtual":
            return mgr.register_new_object(
                adata=adata,
                parent_id=None,
                parent_ids=parent_ids,
                action=rule_name,
                params=rule_params,
                result_key=result_key,
                hash_val=hash_val,
                is_virtual=True,
                effective_params=effective_params,
            )

        raise ValueError(f"Unknown rule result type: {result_type}")

    def ensure(self, target: str, start_state: str, **params: Any) -> str:
        """Run each missing stage after the cached node chosen by `resolve_request`."""
        if target == "raw":
            return start_state

        current_action = self.manager.graph.nodes[start_state]["action"] if start_state else "raw"
        if current_action == target:
            preds = list(self.manager.graph.predecessors(start_state))
            return self.run_rule(target, preds[0], **params) if preds else start_state

        plan = self.build_execution_path(target=target, start_stage=current_action)
        current_node = start_state
        for stage in plan:
            current_node = self.run_rule(stage, current_node, **params)
        return current_node

    def register_raw(self, adata: Any, path: Optional[str], params: Optional[Dict[str, Any]] = None) -> str:
        """Register loaded raw data, including calls from the legacy v4 shim."""
        raw_params = params or {"data_path": path}
        hash_val = compute_step_hash(self.manager, "raw", "init", raw_params)
        if hash_val in self.manager.hash_index:
            return self.manager.hash_index[hash_val]
        return self.manager.register_new_object(
            adata=adata,
            parent_id=None,
            action="raw",
            params=raw_params,
            hash_val=hash_val,
            effective_params=raw_params,
        )


_RUNNER: Optional[SingleCellPipelineRunner] = None


def get_runner(
    ctx: Any = None,
    config: Optional[Dict[str, Any]] = None,
    storage_dir: Optional[str] = None,
    force_new: bool = False,
) -> SingleCellPipelineRunner:
    """Return the shared runner for one TaskWeaver context and storage directory."""
    global _RUNNER

    config = config or {}
    resolved_storage_dir = storage_dir or config.get("storage_dir", DEFAULT_STORAGE_DIR)
    if (
        force_new # Logic to create a new runner 
        or _RUNNER is None
        or _RUNNER.ctx is not ctx
        or _RUNNER.manager.storage_dir != resolved_storage_dir
    ):
        _RUNNER = SingleCellPipelineRunner(
            ctx=ctx,
            config=config,
            storage_dir=resolved_storage_dir,
        )
    return _RUNNER
