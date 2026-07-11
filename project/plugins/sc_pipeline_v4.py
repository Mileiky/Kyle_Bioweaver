import io
import json
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
        mgr = get_manager()
        data_paths = self._coerce_optional_list(data_paths)
        sample_ids = self._coerce_optional_list(sample_ids)

        if target_stage == "umap" and resolution != 0.5:
            self.ctx.log(
                "info",
                "sc_pipeline",
                "Interpreting target_stage='umap' with a non-default resolution as target_stage='cluster'.",
            )
            target_stage = "cluster"

        if data_paths:
            if data_path is not None:
                return self._error("Specify either data_path or data_paths, not both.")
            if sample_ids is None:
                sample_ids = [f"sample_{idx + 1}" for idx in range(len(data_paths))]
            if len(sample_ids) != len(data_paths):
                return self._error("sample_ids must have the same length as data_paths.")
            if scrublet_batch_key is None:
                scrublet_batch_key = sample_key
            if hvg_batch_key is None:
                hvg_batch_key = sample_key
            if combat_key is None:
                combat_key = sample_key
        elif data_path is None:
            source_params = self._infer_active_source_params(mgr)
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
            return self._error(f"Unknown target_stage '{target_stage}'. Valid stages: {sorted(valid_stages)}")

        all_params = {
            "data_path": data_path,
            "data_paths": data_paths,
            "sample_ids": sample_ids,
            "sample_key": sample_key,
            "multi_sample_join": multi_sample_join,
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
            "hvg_batch_key": hvg_batch_key,
            "batch_correction_method": batch_correction_method,
            "combat_key": combat_key,
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
            elif data_paths:
                self.ctx.log("info", "sc_pipeline", f"Loading {len(data_paths)} samples.")
                try:
                    adata = self._read_multi_input_data(
                        data_paths=data_paths,
                        sample_ids=sample_ids,
                        sample_key=sample_key,
                        join=multi_sample_join,
                    )
                except Exception as exc:
                    return self._error(f"Failed to load multi-sample data: {exc}")
                start_node_id = register_raw(
                    mgr,
                    adata,
                    None,
                    params={
                        "data_paths": data_paths,
                        "sample_ids": sample_ids,
                        "sample_key": sample_key,
                        "multi_sample_join": multi_sample_join,
                    },
                )
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
        source_params = self._infer_active_source_params(mgr)
        return source_params.get("data_path") if source_params else None

    def _infer_active_source_params(self, mgr):
        active_node_id = getattr(mgr, "active_node_id", None)
        if active_node_id in mgr.graph.nodes:
            source_params = self._source_params_for_lineage(mgr, active_node_id)
            if source_params:
                return source_params

        raw_nodes = [n for n, attr in mgr.graph.nodes(data=True) if attr.get("action") == "raw"]
        if len(raw_nodes) == 1:
            return mgr.graph.nodes[raw_nodes[0]].get("params", {})

        return None

    def _data_path_for_lineage(self, mgr, node_id):
        source_params = self._source_params_for_lineage(mgr, node_id)
        return source_params.get("data_path") if source_params else None

    def _source_params_for_lineage(self, mgr, node_id):
        lineage = list(nx.ancestors(mgr.graph, node_id)) + [node_id]
        for lineage_node_id in reversed(lineage):
            node_meta = mgr.graph.nodes[lineage_node_id]
            if node_meta.get("action") == "raw":
                return node_meta.get("params", {})
        return None

    def _read_input_data(self, data_path):
        if os.path.isdir(data_path):
            return sc.read_10x_mtx(data_path, var_names="gene_symbols", cache=False)
        return sc.read(data_path)

    def _read_multi_input_data(self, data_paths, sample_ids, sample_key, join):
        adatas = {}
        for data_path, sample_id in zip(data_paths, sample_ids):
            adata = self._read_input_data(data_path)
            adata.var_names_make_unique()
            adata.obs[sample_key] = sample_id
            adatas[sample_id] = adata

        combined = sc.concat(
            adatas,
            label=sample_key,
            index_unique="-",
            join=join,
            merge="same",
        )
        combined.uns["multi_sample"] = {
            "sample_key": sample_key,
            "sample_ids": sample_ids,
            "data_paths": data_paths,
            "join": join,
        }
        return combined

    def _coerce_optional_list(self, value):
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            if stripped == "":
                return None
            if stripped.startswith("["):
                return list(json.loads(stripped))
            separator = ";" if ";" in stripped else ","
            return [item.strip() for item in stripped.split(separator) if item.strip()]
        return list(value)

    def _visualize_result(self, mgr, node_id, stage):
        mgr.active_node_id = node_id
        adata = mgr.get_object(node_id)
        node_meta = mgr.graph.nodes[node_id]

        fig = plt.figure(figsize=(6, 5))
        if stage == "scrublet":
            if "doublet_score" in adata.obs:
                sc.pl.scrublet_score_distribution(adata, show=False)
            elif adata.uns.get("scrublet", {}).get("status") == "skipped":
                self._draw_center_message(
                    plt.gca(),
                    f"Scrublet skipped\n{adata.uns['scrublet'].get('error', '')}",
                )
            else:
                self._draw_center_message(plt.gca(), "Scrublet scores not found")
            plt.title(f"{stage.upper()} Result (Node: {node_id})")
        elif stage == "qc":
            sc.pl.violin(adata, ["total_counts", "n_genes_by_counts"], jitter=0.4, show=False)
            plt.title(f"{stage.upper()} Result (Node: {node_id})")
        elif stage == "normalize":
            sc.pl.violin(adata, ["total_counts", "n_genes_by_counts"], jitter=0.4, show=False)
            plt.title(f"{stage.upper()} Result (Node: {node_id})")
        elif stage == "hvg":
            sc.pl.highly_variable_genes(adata, show=False)
            plt.title(f"{stage.upper()} Result (Node: {node_id})")
        elif stage == "batch_correct":
            fig = self._plot_batch_correction_result(mgr, node_id)
        elif stage == "pca":
            n_pcs = min(20, adata.obsm["X_pca"].shape[1])
            sc.pl.pca_variance_ratio(adata, n_pcs=n_pcs, show=False)
            plt.title(f"{stage.upper()} Result (Node: {node_id})")
        elif stage == "umap":
            sc.pl.umap(adata, show=False)
            plt.title(f"{stage.upper()} Result (Node: {node_id})")
        elif stage == "cluster":
            key = node_meta.get("result_key")
            if key in adata.obs:
                sc.pl.umap(adata, color=key, show=False)
            else:
                self._draw_center_message(plt.gca(), f"Key {key} not found")
            plt.title(f"{stage.upper()} Result (Node: {node_id})")
        elif stage == "markers":
            key = node_meta.get("result_key")
            if key in adata.uns:
                sc.pl.rank_genes_groups(adata, key=key, n_genes=10, show=False)
            else:
                self._draw_center_message(plt.gca(), f"Marker key {key} not found")
            plt.title(f"{stage.upper()} Result (Node: {node_id})")
        elif stage == "annotation":
            if "X_umap" in adata.obsm and "cell_type" in adata.obs:
                sc.pl.umap(adata, color="cell_type", show=False)
            elif "cell_type" in adata.obs:
                counts = adata.obs["cell_type"].value_counts()
                counts.plot(kind="barh", ax=plt.gca())
                plt.xlabel("Cells")
            else:
                self._draw_center_message(plt.gca(), "cell_type annotations not found")
            plt.title(f"{stage.upper()} Result (Node: {node_id})")
        else:
            self._draw_center_message(plt.gca(), f"{stage} complete\nshape={adata.shape}")
            plt.title(f"{stage.upper()} Result (Node: {node_id})")

        bio_buf = io.BytesIO()
        fig.savefig(bio_buf, format="png", bbox_inches="tight", dpi=160)
        plt.close(fig)

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

    def _draw_center_message(self, ax, message):
        ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
        ax.set_axis_off()

    def _plot_batch_correction_result(self, mgr, node_id):
        node_meta = mgr.graph.nodes[node_id]
        adata_after = mgr.get_object(node_id)
        parent_ids = list(mgr.graph.predecessors(node_id))
        adata_before = mgr.get_object(parent_ids[0]) if parent_ids else None
        info = adata_after.uns.get("batch_correction", {})
        params = node_meta.get("params", {})
        batch_key = info.get("key") or params.get("combat_key") or params.get("sample_key")

        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        fig.suptitle(f"BATCH_CORRECT Result (Node: {node_id})", fontsize=12, fontweight="bold")

        before_err = self._plot_pca_panel(
            axes[0],
            adata_before,
            batch_key=batch_key,
            title="Before correction",
        )
        after_err = self._plot_pca_panel(
            axes[1],
            adata_after,
            batch_key=batch_key,
            title=f"After correction ({info.get('method', 'unknown')})",
        )

        subtitle = f"shape={adata_after.shape}"
        if batch_key:
            subtitle += f" | color={batch_key}"
        fig.text(0.5, 0.02, subtitle, ha="center", va="bottom", fontsize=9, color="#475569")

        if before_err or after_err:
            details = "\n".join(err for err in [before_err, after_err] if err)
            fig.text(0.5, 0.06, details, ha="center", va="bottom", fontsize=8, color="#64748b")

        fig.tight_layout(rect=[0, 0.08, 1, 0.95])
        return fig

    def _plot_pca_panel(self, ax, adata, batch_key, title):
        if adata is None:
            self._draw_center_message(ax, "Reference state not available")
            ax.set_title(title)
            return "Reference state not available for before/after comparison."

        plot_adata, err = self._prepare_pca_for_plot(adata)
        if plot_adata is None:
            self._draw_center_message(ax, err)
            ax.set_title(title)
            return f"{title}: {err}"

        color_key = batch_key if batch_key in plot_adata.obs else None
        plot_title = title if color_key else f"{title}\n(batch key unavailable)"

        if color_key:
            sc.pl.pca(plot_adata, color=color_key, ax=ax, show=False, title=plot_title)
        else:
            sc.pl.pca(plot_adata, ax=ax, show=False, title=plot_title)
        return None

    def _prepare_pca_for_plot(self, adata):
        plot_adata = adata.copy()
        if "X_pca" in plot_adata.obsm and plot_adata.obsm["X_pca"].shape[1] >= 2:
            return plot_adata, None

        if plot_adata.n_obs < 2 or plot_adata.n_vars < 2:
            return None, "Not enough cells or genes for PCA."

        try:
            max_comps = min(10, plot_adata.n_obs - 1, plot_adata.n_vars - 1)
            if max_comps < 2:
                return None, "Not enough dimensions for PCA."
            sc.tl.pca(plot_adata, n_comps=max_comps, svd_solver="arpack")
            return plot_adata, None
        except Exception as exc:
            return None, f"PCA plot unavailable: {exc}"

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
        max_secondary_depth = max((self._secondary_depth(y) for _, y in pos.values()), default=0)

        width = min(15, max(10, len({x for x, _ in pos.values()}) * 1.15))
        height = min(8, max(4.8, 3.2 + max_secondary_depth * 0.9))
        plt.figure(figsize=(width, height))

        active_edges = {
            (u, v)
            for u, v in mgr.graph.edges()
            if u in active_lineage and v in active_lineage
        }
        edge_colors = ["#2f6f4e" if edge in active_edges else "#d6dee8" for edge in mgr.graph.edges()]
        edge_widths = [2.6 if edge in active_edges else 0.9 for edge in mgr.graph.edges()]
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
                node_colors.append("#fef3c7")
                edgecolors.append("#d97706")
                linewidths.append(1.2)
            else:
                node_colors.append("#eef2f7")
                edgecolors.append("#94a3b8")
                linewidths.append(1.0)

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
            details = self._dag_label_details(action, params)
            if node_id in active_lineage or node_id == current_node:
                labels[node_id] = f"[{node_id[:4]}]\n{action}{details}"
            else:
                labels[node_id] = f"[{node_id[:4]}]\n{action}"

        nx.draw_networkx_labels(mgr.graph, pos, labels=labels, font_size=7, font_weight="bold")
        stage_y = max(y for _, y in pos.values()) + 0.65
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
        plt.ylim(min(ys) - 0.75, stage_y + 0.45)
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
            x = stage_columns.get(stage, len(stage_columns))
            active_nodes = [node_id for node_id in nodes if node_id in active_lineage]
            inactive_nodes = [node_id for node_id in nodes if node_id not in active_lineage]

            active_nodes = sorted(
                active_nodes,
                key=lambda node_id: (
                    0 if node_id == current_node else 1,
                    self._active_lineage_rank(mgr, current_node, node_id),
                    node_id,
                ),
            )
            inactive_nodes = sorted(
                inactive_nodes,
                key=lambda node_id: (
                    self._same_source_priority(mgr, current_node, node_id),
                    self._lineage_branch_sort_key(mgr, node_id),
                    node_id,
                ),
            )

            for idx, node_id in enumerate(active_nodes):
                pos[node_id] = (x, idx * 0.85)

            for idx, node_id in enumerate(inactive_nodes):
                row = idx + 1
                pos[node_id] = (x, -1.45 - (row - 1) * 1.0)

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

    def _active_lineage_rank(self, mgr, current_node, node_id):
        lineage = list(nx.shortest_path(mgr.graph, source=self._raw_ancestor(mgr, current_node), target=current_node))
        try:
            return lineage.index(node_id)
        except ValueError:
            return len(lineage)

    def _same_source_priority(self, mgr, current_node, node_id):
        current_raw = self._raw_ancestor(mgr, current_node)
        node_raw = self._raw_ancestor(mgr, node_id)
        return 0 if current_raw == node_raw else 1

    def _raw_ancestor(self, mgr, node_id):
        if node_id not in mgr.graph.nodes:
            return None
        lineage = list(nx.ancestors(mgr.graph, node_id)) + [node_id]
        for ancestor_id in reversed(lineage):
            if mgr.graph.nodes[ancestor_id].get("action") == "raw":
                return ancestor_id
        return node_id

    def _secondary_depth(self, y_value):
        if y_value >= 0:
            return 0
        return int(round(abs(y_value + 1.45) / 1.0)) + 1

    def _dag_label_details(self, action, params):
        if action == "qc":
            return f"\nmin={params.get('qc_min_genes', '?')}"
        if action == "scrublet":
            return f"\nrate={params.get('scrublet_expected_doublet_rate', '?')}"
        if action == "hvg":
            return f"\ntop={params.get('n_hvg', '?')}"
        if action == "batch_correct":
            return f"\n{params.get('batch_correction_method', 'none')}"
        if action == "pca":
            return f"\npc={params.get('n_comps', '?')}"
        if action == "neighbors":
            return f"\nk={params.get('n_neighbors', '?')}"
        if action == "cluster":
            return f"\nres={params.get('resolution', '?')}"
        if action == "markers":
            return f"\nn={params.get('n_marker_genes', '?')}"
        if action == "annotation":
            return f"\nn={params.get('n_annotation_markers', '?')}"
        return ""
