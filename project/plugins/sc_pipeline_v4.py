import io
import os
import textwrap

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
        model: str = "qwen3.5:122b",
        api_base: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        n_markers: int = 10,
        show_plots: bool = False,
        min_genes: int = None,
        max_genes: int = None,
        pct_mt: float = None,
        n_top_genes: int = None,
        flavor: str = None,
        max_value: int = None,
        method: str = None,
        n_genes: int = None,
        **kwargs,
    ):
        mgr = get_manager()
        target_stage = {
            "cell_type": "annotation",
            "cell_type_annotation": "annotation",
            "annotate": "annotation",
        }.get(target_stage, target_stage)
        valid_stages = set(mgr.registry.rules.keys()) | {"raw"}
        if target_stage not in valid_stages:
            return f"Unknown target_stage '{target_stage}'. Valid stages: {sorted(valid_stages)}"

        qc_min_genes = min_genes if min_genes is not None else qc_min_genes
        qc_max_genes = max_genes if max_genes is not None else qc_max_genes
        qc_mt_pct = pct_mt if pct_mt is not None else qc_mt_pct
        n_hvg = n_top_genes if n_top_genes is not None else n_hvg
        hvg_flavor = flavor if flavor is not None else hvg_flavor
        max_scale_value = max_value if max_value is not None else max_scale_value
        if method is not None:
            if target_stage == "markers":
                marker_method = method
            else:
                cluster_method = method
        n_marker_genes = n_genes if n_genes is not None else n_marker_genes

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
            "model": model,
            "api_base": api_base,
            "api_key": api_key,
            "n_markers": n_markers,
            **kwargs,
        }

        self.ctx.log("info", "sc_pipeline", f"Smart search: '{target_stage}' with params: {all_params}")
        try:
            result_node_id, match_type = mgr.find_node_smart(target_stage, **all_params)
        except ValueError as exc:
            return str(exc)

        if match_type == "exact_match":
            self.ctx.log("info", "sc_pipeline", f"Exact lineage match found: {result_node_id}")
            return self._visualize_result(mgr, result_node_id, target_stage, show_plots=show_plots)

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
                try:
                    adata = self._read_input_data(data_path)
                except Exception as exc:
                    return f"Failed to load data from '{data_path}': {exc}"
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

        return self._visualize_result(mgr, final_node_id, target_stage, show_plots=show_plots)

    def _read_input_data(self, data_path):
        if os.path.isdir(data_path):
            return sc.read_10x_mtx(data_path, var_names="gene_symbols", cache=False)
        return sc.read(data_path)

    def _visualize_result(self, mgr, node_id, stage, show_plots=False):
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
        elif stage == "annotation":
            if "X_umap" in adata.obsm and "cell_type" in adata.obs:
                sc.pl.umap(adata, color="cell_type", show=False)
            elif "cell_type" in adata.obs:
                counts = adata.obs["cell_type"].value_counts(dropna=False)
                counts.plot(kind="bar", ax=plt.gca())
                plt.ylabel("Cells")
                plt.xlabel("Cell type")
            else:
                plt.text(0.5, 0.5, "cell_type annotation not found", ha="center")
        else:
            plt.text(0.5, 0.5, f"{stage} complete\nshape={adata.shape}", ha="center")

        plt.title(f"{stage.upper()} Result (Node: {node_id})")
        bio_buf = io.BytesIO()
        plt.savefig(bio_buf, format="png", bbox_inches="tight", dpi=160)
        if show_plots:
            plt.show()
        plt.close()

        _, bio_path = self.ctx.create_artifact_path(
            name="Analysis_Result",
            file_name=f"result_{node_id}.png",
            type="image",
            desc=f"Plot for {stage}.",
        )
        with open(bio_path, "wb") as f:
            f.write(bio_buf.getvalue())

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
        graph = mgr.graph.copy()
        for node_id in graph.nodes:
            graph.nodes[node_id]["rank"] = self._node_rank(mgr, node_id)

        node_count = max(1, graph.number_of_nodes())
        fig_width = max(10, min(20, 2.0 * len({attr["rank"] for _, attr in graph.nodes(data=True)}) + 6))
        fig_height = max(5, min(16, node_count * 0.55 + 2))
        plt.figure(figsize=(fig_width, fig_height))
        try:
            agraph = nx.nx_agraph.to_agraph(graph)
            agraph.graph_attr.update(rankdir="LR", nodesep="0.85", ranksep="1.25", splines="ortho")
            for node in agraph.nodes():
                node.attr.update(shape="box", style="rounded,filled", width="1.45", height="0.65", margin="0.08")
            pos = nx.nx_agraph.graphviz_layout(nx.nx_agraph.from_agraph(agraph), prog="dot")
        except (ImportError, OSError):
            pos = nx.multipartite_layout(graph, subset_key="rank", align="vertical", scale=3.5)

        node_colors = ["#9BE7A7" if n == current_node else "#D9ECFF" for n in graph.nodes()]
        nx.draw_networkx_edges(
            graph,
            pos,
            node_size=2200,
            edge_color="#6B7280",
            arrows=True,
            arrowstyle="-|>",
            arrowsize=16,
            width=1.4,
            connectionstyle="arc3,rad=0.04",
            min_source_margin=12,
            min_target_margin=16,
        )
        nx.draw_networkx_nodes(
            graph,
            pos,
            node_color=node_colors,
            node_size=1850,
            edgecolors="#374151",
            linewidths=1.0,
            node_shape="s",
        )

        labels = {}
        for node_id, attr in graph.nodes(data=True):
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
            elif action == "annotation":
                details = f"\nn={params.get('n_markers', '?')}"

            labels[node_id] = self._dag_label(node_id, action, details)

        nx.draw_networkx_labels(graph, pos, labels=labels, font_size=7.5, font_family="sans-serif")
        plt.title("Single-Cell Pipeline DAG", fontsize=12, pad=14)
        plt.axis("off")
        plt.margins(x=0.12, y=0.16)
        png_buf = io.BytesIO()
        plt.savefig(png_buf, format="png", bbox_inches="tight", dpi=160)
        plt.close()

        _, png_path = self.ctx.create_artifact_path(
            name="Pipeline_State",
            file_name=f"result_pipeline_dag_{current_node}.png",
            type="image",
            desc="Current pipeline DAG.",
        )
        with open(png_path, "wb") as f:
            f.write(png_buf.getvalue())

    def _node_rank(self, mgr, node_id):
        action = mgr.graph.nodes[node_id].get("action")
        if action == "raw":
            return 0
        try:
            return mgr.dependency_chain(action).index(action)
        except Exception:
            return 0

    def _dag_label(self, node_id, action, details):
        label_action = textwrap.fill(str(action).upper(), width=11)
        return f"{label_action}\n{node_id[:6]}{details}"
