import io

import matplotlib.pyplot as plt
import networkx as nx
import scanpy as sc
from taskweaver.plugin import Plugin, register_plugin

from project.utils.sc_dag_v4 import compute_step_hash, ensure, get_manager, register_raw


@register_plugin
class SingleCellPipeline(Plugin):
    def __call__(
        self,
        target_stage: str,
        data_path: str = None,
        qc_min_genes: int = 200,
        qc_max_genes: int = 2500,
        qc_mt_pct: float = 5,
        min_cells: int = 3,
        target_sum: float = 1e4,
        n_hvg: int = 2000,
        hvg_flavor: str = "seurat",
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
        **kwargs,
    ):
        mgr = get_manager()
        valid_stages = set(mgr.registry.rules.keys()) | {"raw"}
        if target_stage not in valid_stages:
            return f"Unknown target_stage '{target_stage}'. Valid stages: {sorted(valid_stages)}"

        all_params = {
            "data_path": data_path,
            "qc_min_genes": qc_min_genes,
            "qc_max_genes": qc_max_genes,
            "qc_mt_pct": qc_mt_pct,
            "min_cells": min_cells,
            "target_sum": target_sum,
            "n_hvg": n_hvg,
            "hvg_flavor": hvg_flavor,
            "max_scale_value": max_scale_value,
            "regress_out": regress_out,
            "n_comps": n_comps,
            "n_neighbors": n_neighbors,
            "n_pcs": n_pcs,
            "use_rep": use_rep,
            "min_dist": min_dist,
            "spread": spread,
            "resolution": resolution,
            "cluster_method": cluster_method,
            "groupby": groupby,
            "marker_method": marker_method,
            "n_marker_genes": n_marker_genes,
            **kwargs,
        }

        self.ctx.log("info", "sc_pipeline", f"Smart search: '{target_stage}' with params: {all_params}")
        try:
            result_node_id, match_type = mgr.find_node_smart(target_stage, **all_params)
        except ValueError as exc:
            return str(exc)

        if match_type == "exact_match":
            self.ctx.log("info", "sc_pipeline", f"Exact lineage match found: {result_node_id}")
            return self._visualize_result(mgr, result_node_id, target_stage)

        if match_type == "ambiguous":
            return "Ambiguous request: found multiple partial matches. Specify upstream parameters to clarify."

        self.ctx.log("info", "sc_pipeline", "Locating nearest valid cached ancestor.")
        start_node_id = None
        try:
            ancestor_chain = list(reversed(mgr.dependency_chain(target_stage)[:-1]))
        except Exception:
            ancestor_chain = []

        for stage in ancestor_chain:
            try:
                node_id = mgr.find_node_strict(stage, **all_params)
            except ValueError:
                node_id = None
            if node_id:
                start_node_id = node_id
                self.ctx.log("info", "sc_pipeline", f"Anchor found: {stage.upper()} ({node_id})")
                break

        if not start_node_id:
            try:
                target_raw_hash = compute_step_hash(mgr, "raw", "init", all_params)
            except ValueError as exc:
                return str(exc)

            if target_raw_hash in mgr.hash_index:
                start_node_id = mgr.hash_index[target_raw_hash]
            elif data_path:
                self.ctx.log("info", "sc_pipeline", f"Loading data from {data_path}.")
                adata = sc.read(data_path)
                start_node_id = register_raw(mgr, adata, data_path)
            else:
                return "No data found and no valid parent state exists. Provide data_path."

        try:
            final_node_id = ensure(
                mgr,
                target=target_stage,
                start_state=start_node_id,
                **all_params,
            )
        except Exception as exc:
            return f"Pipeline failed: {exc}"

        return self._visualize_result(mgr, final_node_id, target_stage)

    def _visualize_result(self, mgr, node_id, stage):
        adata = mgr.get_object(node_id)
        node_meta = mgr.graph.nodes[node_id]

        plt.figure(figsize=(6, 5))
        if stage == "qc":
            sc.pl.violin(adata, ["total_counts", "n_genes_by_counts"], jitter=0.4, show=False)
        elif stage == "normalize":
            sc.pl.violin(adata, ["total_counts", "n_genes_by_counts"], jitter=0.4, show=False)
        elif stage == "hvg":
            sc.pl.highly_variable_genes(adata, show=False)
        elif stage == "pca":
            n_pcs = min(20, adata.obsm["X_pca"].shape[1])
            sc.pl.pca_variance_ratio(adata, n_pcs=n_pcs, show=False)
        elif stage == "umap":
            sc.pl.umap(adata, show=False)
        elif stage == "cluster":
            key = node_meta.get("result_key")
            if key in adata.obs:
                sc.pl.umap(adata, color=key, show=False)
            else:
                plt.text(0.5, 0.5, f"Key {key} not found", ha="center")
        elif stage == "markers":
            key = node_meta.get("result_key")
            if key in adata.uns:
                sc.pl.rank_genes_groups(adata, key=key, n_genes=10, show=False)
            else:
                plt.text(0.5, 0.5, f"Marker key {key} not found", ha="center")
        else:
            plt.text(0.5, 0.5, f"{stage} complete\nshape={adata.shape}", ha="center")

        plt.title(f"{stage.upper()} Result (Node: {node_id})")
        bio_buf = io.StringIO()
        plt.savefig(bio_buf, format="svg", bbox_inches="tight")
        plt.close()

        self.ctx.add_artifact(
            name="Analysis_Result",
            file_name=f"result_{node_id}.svg",
            type="svg",
            val=bio_buf.getvalue(),
            desc=f"Plot for {stage}.",
        )

        self._plot_dag(mgr, node_id)
        return mgr.get_object(node_id), self._summary(mgr, node_id, stage)

    def _summary(self, mgr, node_id, stage):
        adata = mgr.get_object(node_id)
        node_meta = mgr.graph.nodes[node_id]
        result_key = node_meta.get("result_key")
        lineage = list(nx.ancestors(mgr.graph, node_id)) + [node_id]
        return (
            f"Stage '{stage}' complete.\n"
            f"- Active node: {node_id}\n"
            f"- Data shape: {adata.n_obs} cells x {adata.n_vars} genes\n"
            f"- Result key: {result_key or 'None'}\n"
            f"- Cached DAG nodes: {len(mgr.graph.nodes)}\n"
            f"- Lineage nodes considered: {len(lineage)}"
        )

    def _plot_dag(self, mgr, current_node):
        plt.figure(figsize=(9, 6))
        try:
            pos = nx.nx_agraph.graphviz_layout(mgr.graph, prog="dot")
        except ImportError:
            pos = nx.spring_layout(mgr.graph, seed=42)

        node_colors = ["lightgreen" if n == current_node else "lightblue" for n in mgr.graph.nodes()]
        nx.draw(
            mgr.graph,
            pos,
            with_labels=False,
            node_color=node_colors,
            node_size=2200,
            edge_color="#555555",
            arrows=True,
            arrowstyle="-|>",
            arrowsize=20,
        )

        labels = {}
        for node_id, attr in mgr.graph.nodes(data=True):
            action = attr.get("action", "?")
            params = attr.get("params", {})
            details = ""
            if action == "qc":
                details = f"\nmin={params.get('qc_min_genes', '?')}"
            elif action == "hvg":
                details = f"\ntop={params.get('n_hvg', '?')}"
            elif action == "pca":
                details = f"\npc={params.get('n_comps', '?')}"
            elif action == "neighbors":
                details = f"\nk={params.get('n_neighbors', '?')}"
            elif action == "cluster":
                details = f"\nres={params.get('resolution', '?')}"
            elif action == "markers":
                details = f"\nn={params.get('n_marker_genes', '?')}"

            labels[node_id] = f"[{node_id[:4]}]\n{action}{details}"

        nx.draw_networkx_labels(mgr.graph, pos, labels=labels, font_size=8)
        plt.axis("off")
        dag_buf = io.StringIO()
        plt.savefig(dag_buf, format="svg", bbox_inches="tight")
        plt.close()

        self.ctx.add_artifact(
            name="Pipeline_State",
            file_name=f"result_pipeline_dag_{current_node}.svg",
            type="svg",
            val=dag_buf.getvalue(),
            desc="Current pipeline DAG.",
        )
