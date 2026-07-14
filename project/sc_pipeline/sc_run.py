"""Pipeline orchestration for the single-cell TaskWeaver plugin."""

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
        data_path = kwargs.get("data_path")
        data_paths = self.io.coerce_optional_list(kwargs.get("data_paths"))
        sample_ids = self.io.coerce_optional_list(kwargs.get("sample_ids"))
        sample_key = kwargs.get("sample_key", "sample")
        multi_sample_join = kwargs.get("multi_sample_join", "inner")
        scrublet_batch_key = kwargs.get("scrublet_batch_key")
        hvg_batch_key = kwargs.get("hvg_batch_key")
        combat_key = kwargs.get("combat_key")

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
            if len(sample_ids) != len(data_paths):
                raise ValueError("sample_ids must have the same length as data_paths.")
            if scrublet_batch_key is None:
                scrublet_batch_key = sample_key
            if hvg_batch_key is None:
                hvg_batch_key = sample_key
            if combat_key is None:
                combat_key = sample_key
        elif data_path is None:
            source_params = self.infer_active_source_params()
            if source_params:
                data_path = source_params.get("data_path")
                data_paths = source_params.get("data_paths")
                sample_ids = source_params.get("sample_ids")
                sample_key = source_params.get("sample_key", sample_key)
                multi_sample_join = source_params.get("multi_sample_join", multi_sample_join)
            if data_paths:
                if scrublet_batch_key is None:
                    scrublet_batch_key = sample_key
                if hvg_batch_key is None:
                    hvg_batch_key = sample_key
                if combat_key is None:
                    combat_key = sample_key

        valid_stages = set(mgr.registry.rules.keys()) | {"raw"}
        if target_stage not in valid_stages:
            raise ValueError(f"Unknown target_stage '{target_stage}'. Valid stages: {sorted(valid_stages)}")

        params = {
            "data_path": data_path,
            "data_paths": data_paths,
            "sample_ids": sample_ids,
            "sample_key": sample_key,
            "multi_sample_join": multi_sample_join,
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
            "batch_correction_method": kwargs.get("batch_correction_method", "none"),
            "combat_key": combat_key,
            "max_scale_value": kwargs.get("max_scale_value", 10),
            "regress_out": kwargs.get("regress_out", True),
            "n_comps": kwargs.get("n_comps", 50),
            "n_neighbors": kwargs.get("n_neighbors", 10),
            "n_pcs": kwargs.get("n_pcs", 40),
            "use_rep": kwargs.get("use_rep", "X_pca"),
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

    def infer_active_source_params(self) -> Optional[Dict[str, Any]]:
        """Find raw input parameters when `normalize_request` receives no path."""
        mgr = self.manager
        active_node_id = getattr(mgr, "active_node_id", None)
        if active_node_id in mgr.graph.nodes:
            raw_node_id = mgr.raw_ancestor(active_node_id)
            if raw_node_id is not None:
                return mgr.graph.nodes[raw_node_id].get("params", {})

        raw_nodes = [node_id for node_id, attr in mgr.graph.nodes(data=True) if attr.get("action") == "raw"]
        if len(raw_nodes) == 1:
            return mgr.graph.nodes[raw_nodes[0]].get("params", {})

        return None

    def resolve_request(self, request: PipelineRequest) -> str:
        """Return the exact cached node or build missing stages for `execute`."""
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

        data_paths = params.get("data_paths")
        data_path = params.get("data_path")
        if data_paths:
            self.ctx.log("info", "sc_pipeline", f"Loading {len(data_paths)} samples.")
            adata = self.io.read_multi_input_data(
                data_paths=data_paths,
                sample_ids=params.get("sample_ids"),
                sample_key=params.get("sample_key"),
                join=params.get("multi_sample_join"),
            )
            raw_params = {
                "data_paths": data_paths,
                "sample_ids": params.get("sample_ids"),
                "sample_key": params.get("sample_key"),
                "multi_sample_join": params.get("multi_sample_join"),
            }
            return self.register_raw(adata, path=None, params=raw_params)

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

    def run_rule(self, rule_name: str, parent_id: str, **params: Any) -> str:
        """Run one rule for `ensure`, or reuse the child with the same step hash."""
        mgr = self.manager
        parent_hash = mgr.graph.nodes[parent_id].get("hash", "init") if parent_id else "init"
        hash_val = compute_step_hash(mgr, rule_name, parent_hash, params)

        if hash_val in mgr.hash_index:
            return mgr.hash_index[hash_val]

        rule = mgr.registry.get(rule_name)
        rule_params = {key: value for key, value in params.items() if key in rule.param_keys}
        adata, result_type, *result_keys = rule.func(mgr, parent_id, **rule_params)
        result_key = result_keys[0] if result_keys else None

        if result_type == "new_object":
            return mgr.register_new_object(
                adata=adata,
                parent_id=parent_id,
                action=rule_name,
                params=rule_params,
                hash_val=hash_val,
                result_key=result_key,
            )
        if result_type == "virtual":
            return mgr.register_new_object(
                adata=adata,
                parent_id=parent_id,
                action=rule_name,
                params=rule_params,
                result_key=result_key,
                hash_val=hash_val,
                is_virtual=True,
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
        force_new
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
