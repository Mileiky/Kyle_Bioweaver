import io
import os

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
        mgr = get_manager()

        if target_stage == "umap" and resolution != 0.5:
            self.ctx.log(
                "info",
                "sc_pipeline",
                "Interpreting target_stage='umap' with a non-default resolution as target_stage='cluster'.",
            )
            target_stage = "cluster"

        if data_path is None:
            data_path = self._infer_active_data_path(mgr)

        valid_stages = set(mgr.registry.rules.keys()) | {"raw"}
        if target_stage not in valid_stages:
            return self._error(f"Unknown target_stage '{target_stage}'. Valid stages: {sorted(valid_stages)}")

        all_params = {
            "data_path": data_path,
            "scrublet_batch_key": scrublet_batch_key,
            "scrublet_expected_doublet_rate": scrublet_expected_doublet_rate,
            "scrublet_threshold": scrublet_threshold,
            "scrublet_n_prin_comps": scrublet_n_prin_comps,
            "scrublet_filter_doublets": scrublet_filter_doublets,
            "scrublet_skip_on_failure": scrublet_skip_on_failure,
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
            "annotation_model": annotation_model,
            "annotation_api_base": annotation_api_base,
            "annotation_api_key": annotation_api_key,
            "n_annotation_markers": n_annotation_markers,
            **kwargs,
        }

        self.ctx.log("info", "sc_pipeline", f"Smart search: '{target_stage}' with params: {all_params}")
        try:
            result_node_id, match_type = mgr.find_node_smart(target_stage, **all_params)
        except ValueError as exc:
            return self._error(str(exc))

        if match_type == "exact_match":
            self.ctx.log("info", "sc_pipeline", f"Exact lineage match found: {result_node_id}")
            return self._visualize_result(mgr, result_node_id, target_stage)

        if match_type == "ambiguous":
            return self._error("Ambiguous request: found multiple partial matches. Specify upstream parameters to clarify.")

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
                return self._error(str(exc))

            if target_raw_hash in mgr.hash_index:
                start_node_id = mgr.hash_index[target_raw_hash]
            elif data_path:
                self.ctx.log("info", "sc_pipeline", f"Loading data from {data_path}.")
                try:
                    adata = self._read_input_data(data_path)
                except Exception as exc:
                    return self._error(f"Failed to load data from '{data_path}': {exc}")
                start_node_id = register_raw(mgr, adata, data_path)
            else:
                return self._error("No data found and no valid parent state exists. Provide data_path.")

        try:
            final_node_id = ensure(
                mgr,
                target=target_stage,
                start_state=start_node_id,
                **all_params,
            )
        except Exception as exc:
            return self._error(f"Pipeline failed: {exc}")

        return self._visualize_result(mgr, final_node_id, target_stage)

    def _error(self, message):
        return None, message

    def _infer_active_data_path(self, mgr):
        active_node_id = getattr(mgr, "active_node_id", None)
        if active_node_id in mgr.graph.nodes:
            data_path = self._data_path_for_lineage(mgr, active_node_id)
            if data_path:
                return data_path

        raw_nodes = [n for n, attr in mgr.graph.nodes(data=True) if attr.get("action") == "raw"]
        if len(raw_nodes) == 1:
            return mgr.graph.nodes[raw_nodes[0]].get("params", {}).get("data_path")

        return None

    def _data_path_for_lineage(self, mgr, node_id):
        lineage = list(nx.ancestors(mgr.graph, node_id)) + [node_id]
        for lineage_node_id in reversed(lineage):
            node_meta = mgr.graph.nodes[lineage_node_id]
            if node_meta.get("action") == "raw":
                return node_meta.get("params", {}).get("data_path")
        return None

    def _read_input_data(self, data_path):
        if os.path.isdir(data_path):
            return sc.read_10x_mtx(data_path, var_names="gene_symbols", cache=False)
        return sc.read(data_path)

    def _visualize_result(self, mgr, node_id, stage):
        mgr.active_node_id = node_id
        adata = mgr.get_object(node_id)
        node_meta = mgr.graph.nodes[node_id]

        plt.figure(figsize=(6, 5))
        if stage == "scrublet":
            if "doublet_score" in adata.obs:
                sc.pl.scrublet_score_distribution(adata, show=False)
            elif adata.uns.get("scrublet", {}).get("status") == "skipped":
                plt.text(
                    0.5,
                    0.5,
                    f"Scrublet skipped\n{adata.uns['scrublet'].get('error', '')}",
                    ha="center",
                    va="center",
                    wrap=True,
                )
            else:
                plt.text(0.5, 0.5, "Scrublet scores not found", ha="center")
        elif stage == "qc":
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
                counts = adata.obs["cell_type"].value_counts()
                counts.plot(kind="barh", ax=plt.gca())
                plt.xlabel("Cells")
            else:
                plt.text(0.5, 0.5, "cell_type annotations not found", ha="center")
        else:
            plt.text(0.5, 0.5, f"{stage} complete\nshape={adata.shape}", ha="center")

        plt.title(f"{stage.upper()} Result (Node: {node_id})")
        bio_buf = io.BytesIO()
        plt.savefig(bio_buf, format="png", bbox_inches="tight", dpi=160)
        plt.close()

        _, bio_path = self.ctx.create_artifact_path(
            name="Analysis_Result",
            file_name=f"result_{node_id}.png",
            type="image",
            desc=f"Plot for {stage}.",
        )
        with open(bio_path, "wb") as f:
            f.write(bio_buf.getvalue())

        self._plot_final_umap(mgr, node_id, stage)
        self._plot_dag(mgr, node_id)
        return mgr.get_object(node_id), self._summary(mgr, node_id, stage)

    def _plot_final_umap(self, mgr, node_id, stage):
        if stage in {"umap", "cluster"}:
            return

        adata = mgr.get_object(node_id)
        if "X_umap" not in adata.obsm:
            return

        color_key = self._get_umap_color_key(mgr, node_id, adata)

        plt.figure(figsize=(6, 5))
        if color_key is not None:
            sc.pl.umap(adata, color=color_key, show=False)
        else:
            sc.pl.umap(adata, show=False)
        plt.title(f"Final UMAP (Node: {node_id})")

        umap_buf = io.BytesIO()
        plt.savefig(umap_buf, format="png", bbox_inches="tight", dpi=160)
        plt.close()

        _, umap_path = self.ctx.create_artifact_path(
            name="Final_UMAP",
            file_name=f"result_final_umap_{node_id}.png",
            type="image",
            desc=f"Final UMAP plot{f' colored by {color_key}' if color_key else ''}.",
        )
        with open(umap_path, "wb") as f:
            f.write(umap_buf.getvalue())

    def _get_umap_color_key(self, mgr, node_id, adata):
        node_meta = mgr.graph.nodes[node_id]
        result_key = node_meta.get("result_key")
        if result_key in adata.obs:
            return result_key

        lineage = list(nx.ancestors(mgr.graph, node_id)) + [node_id]
        for ancestor_id in reversed(lineage):
            ancestor_key = mgr.graph.nodes[ancestor_id].get("result_key")
            if ancestor_key in adata.obs:
                return ancestor_key

        if "cell_type" in adata.obs:
            return "cell_type"

        for prefix in ("leiden_res", "louvain_res"):
            matching_keys = [key for key in adata.obs.keys() if str(key).startswith(prefix)]
            if matching_keys:
                return matching_keys[-1]

        return None

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
        if not mgr.graph.nodes:
            return

        pos = self._pipeline_dag_layout(mgr, current_node)
        active_lineage = self._active_lineage(mgr, current_node)
        stage_counts = {}
        for _, attr in mgr.graph.nodes(data=True):
            stage = attr.get("action", "?")
            stage_counts[stage] = stage_counts.get(stage, 0) + 1

        width = min(14, max(10, len({x for x, _ in pos.values()}) * 1.05))
        height = min(7, max(4.5, max(stage_counts.values(), default=1) * 0.9 + 2))
        plt.figure(figsize=(width, height))

        active_edges = {
            (u, v)
            for u, v in mgr.graph.edges()
            if u in active_lineage and v in active_lineage
        }
        edge_colors = ["#2f6f4e" if edge in active_edges else "#bac4d0" for edge in mgr.graph.edges()]
        edge_widths = [2.4 if edge in active_edges else 1.1 for edge in mgr.graph.edges()]
        nx.draw_networkx_edges(
            mgr.graph,
            pos,
            edge_color=edge_colors,
            width=edge_widths,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=16,
            min_source_margin=18,
            min_target_margin=18,
            connectionstyle="arc3,rad=0.02",
        )

        node_colors = []
        edgecolors = []
        linewidths = []
        for node_id, attr in mgr.graph.nodes(data=True):
            if node_id == current_node:
                node_colors.append("#2fb344")
                edgecolors.append("#14532d")
                linewidths.append(2.6)
            elif node_id in active_lineage:
                node_colors.append("#b7e4c7")
                edgecolors.append("#2f6f4e")
                linewidths.append(2.0)
            elif attr.get("is_virtual"):
                node_colors.append("#fde68a")
                edgecolors.append("#b45309")
                linewidths.append(1.4)
            else:
                node_colors.append("#dbeafe")
                edgecolors.append("#3b82f6")
                linewidths.append(1.2)

        nx.draw_networkx_nodes(
            mgr.graph,
            pos,
            node_color=node_colors,
            node_size=1900,
            edgecolors=edgecolors,
            linewidths=linewidths,
        )

        labels = {}
        for node_id, attr in mgr.graph.nodes(data=True):
            action = attr.get("action", "?")
            params = attr.get("params", {})
            details = ""
            if action == "qc":
                details = f"\nmin={params.get('qc_min_genes', '?')}"
            elif action == "scrublet":
                details = f"\nrate={params.get('scrublet_expected_doublet_rate', '?')}"
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
                details = f"\nn={params.get('n_annotation_markers', '?')}"

            labels[node_id] = f"[{node_id[:4]}]\n{action}{details}"

        nx.draw_networkx_labels(mgr.graph, pos, labels=labels, font_size=7, font_weight="bold")
        stage_y = max(y for _, y in pos.values()) + 0.55
        for stage, x in self._stage_columns(mgr).items():
            if any(attr.get("action", "?") == stage for _, attr in mgr.graph.nodes(data=True)):
                plt.text(
                    x,
                    stage_y,
                    stage.upper(),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                    color="#334155",
                )

        plt.axis("off")
        xs = [x for x, _ in pos.values()]
        ys = [y for _, y in pos.values()]
        plt.xlim(min(xs) - 0.9, max(xs) + 0.9)
        plt.ylim(min(ys) - 0.75, stage_y + 0.35)
        png_buf = io.BytesIO()
        plt.savefig(png_buf, format="png", dpi=150)
        plt.close()

        _, png_path = self.ctx.create_artifact_path(
            name="Pipeline_State",
            file_name=f"result_pipeline_dag_{current_node}.png",
            type="image",
            desc="Current pipeline DAG.",
        )
        with open(png_path, "wb") as f:
            f.write(png_buf.getvalue())

    def _pipeline_dag_layout(self, mgr, current_node):
        active_lineage = self._active_lineage(mgr, current_node)
        stage_columns = self._stage_columns(mgr)
        stage_nodes = {}
        for node_id, attr in mgr.graph.nodes(data=True):
            stage = attr.get("action", "?")
            stage_nodes.setdefault(stage, []).append(node_id)

        pos = {}
        for stage, nodes in stage_nodes.items():
            nodes = sorted(
                nodes,
                key=lambda node_id: (
                    node_id not in active_lineage,
                    self._lineage_branch_sort_key(mgr, node_id),
                    node_id,
                ),
            )
            x = stage_columns.get(stage, len(stage_columns))
            for idx, node_id in enumerate(nodes):
                pos[node_id] = (x, -idx * 1.1)

        return pos

    def _stage_columns(self, mgr):
        known_order = ["raw"] + list(mgr.dependency_chain("annotation")[1:])
        seen = set()
        ordered_stages = []
        for stage in known_order:
            if stage not in seen:
                ordered_stages.append(stage)
                seen.add(stage)

        extra_stages = sorted(
            {
                attr.get("action", "?")
                for _, attr in mgr.graph.nodes(data=True)
                if attr.get("action", "?") not in seen
            },
        )
        ordered_stages.extend(extra_stages)
        return {stage: idx * 1.25 for idx, stage in enumerate(ordered_stages)}

    def _active_lineage(self, mgr, current_node):
        if current_node not in mgr.graph.nodes:
            return set()
        return set(nx.ancestors(mgr.graph, current_node)) | {current_node}

    def _lineage_branch_sort_key(self, mgr, node_id):
        descendants = nx.descendants(mgr.graph, node_id)
        return -len(descendants)
