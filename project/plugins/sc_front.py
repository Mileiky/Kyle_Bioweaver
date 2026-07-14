from taskweaver.plugin import Plugin, register_plugin

from project.sc_pipeline.sc_run import get_runner


@register_plugin
class SingleCellPipeline(Plugin):
    def __call__(
        self,
        target_stage: str,
        data_path: str = None,
        data_paths=None,
        sample_ids=None,
        sample_key: str = "sample",
        multi_sample_join: str = "inner",
        scrublet_batch_key: str = None,
        scrublet_expected_doublet_rate: float = 0.05,
        scrublet_threshold: float = None,
        scrublet_n_prin_comps: int = 30,
        scrublet_filter_doublets: bool = False,
        scrublet_skip_on_failure: bool = True,
        qc_min_genes: int = 200,
        qc_max_genes: int = 2500,
        qc_mt_pct: float = 5,
        min_cells: int = 3,
        target_sum: float = 1e4,
        n_hvg: int = 2000,
        hvg_flavor: str = "seurat",
        hvg_batch_key: str = None,
        batch_correction_method: str = "none",
        combat_key: str = None,
        max_scale_value: int = 10,
        regress_out: bool = True,
        n_comps: int = 50,
        n_neighbors: int = 10,
        n_pcs: int = 40,
        use_rep: str = "X_pca",
        min_dist: float = 0.5,
        spread: float = 1.0,
        resolution: float = 0.5,
        cluster_method: str = "leiden",
        groupby: str = None,
        marker_method: str = "wilcoxon",
        n_marker_genes: int = 25,
        annotation_model: str = "qwen3.5:122b",
        annotation_api_base: str = "http://localhost:11434/v1",
        annotation_api_key: str = "ollama",
        n_annotation_markers: int = 10,
        **kwargs,
    ):
        """Run the single-cell pipeline when TaskWeaver calls this plugin."""
        try:
            runner = get_runner(ctx=self.ctx, config=self.config)
            return runner.execute(
                target_stage=target_stage,
                data_path=data_path,
                data_paths=data_paths,
                sample_ids=sample_ids,
                sample_key=sample_key,
                multi_sample_join=multi_sample_join, # Look at multi-sample behavior
                scrublet_batch_key=scrublet_batch_key, # Let into scrublet, how it works, and what is done here
                scrublet_expected_doublet_rate=scrublet_expected_doublet_rate,
                scrublet_threshold=scrublet_threshold,
                scrublet_n_prin_comps=scrublet_n_prin_comps,
                scrublet_filter_doublets=scrublet_filter_doublets,
                scrublet_skip_on_failure=scrublet_skip_on_failure,
                qc_min_genes=qc_min_genes,
                qc_max_genes=qc_max_genes,
                qc_mt_pct=qc_mt_pct,
                min_cells=min_cells,
                target_sum=target_sum,
                n_hvg=n_hvg,
                hvg_flavor=hvg_flavor,
                hvg_batch_key=hvg_batch_key,
                batch_correction_method=batch_correction_method,
                combat_key=combat_key, # What is combat key?
                max_scale_value=max_scale_value,
                regress_out=regress_out,
                n_comps=n_comps,
                n_neighbors=n_neighbors,
                n_pcs=n_pcs,
                use_rep=use_rep,
                min_dist=min_dist,
                spread=spread,
                resolution=resolution,
                cluster_method=cluster_method,
                groupby=groupby,
                marker_method=marker_method,
                n_marker_genes=n_marker_genes,
                annotation_model=annotation_model,
                annotation_api_base=annotation_api_base,
                annotation_api_key=annotation_api_key,
                n_annotation_markers=n_annotation_markers,
                **kwargs,
            )
        except Exception as exc:
            raise RuntimeError(f"Single-cell pipeline failed: {exc}") from exc
