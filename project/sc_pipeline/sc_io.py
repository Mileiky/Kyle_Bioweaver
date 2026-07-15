"""Input loading, plotting, summaries, and TaskWeaver artifact helpers."""

from __future__ import annotations

import io
import json
import os
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import scanpy as sc


class SingleCellIO:
    """Load input data and publish plots and artifacts for the pipeline runner."""

    def __init__(self, ctx: Any):
        self.ctx = ctx

    def coerce_optional_list(self, value: Any):
        """Turn TaskWeaver list input into a list for runner normalization."""
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

    def read_input_data(self, data_path: str):
        """Load one file or 10x directory for the pipeline runner."""
        if os.path.isdir(data_path):
            return sc.read_10x_mtx(data_path, var_names="gene_symbols", cache=False)
        return sc.read(data_path)

    def visualize_result(self, mgr: Any, node_id: str, stage: str):
        """Publish plots and a summary after the runner selects its result node."""
        mgr.active_node_id = node_id
        mgr.save()
        adata = mgr.get_object(node_id)
        node_meta = mgr.graph.nodes[node_id]

        initial_fig = fig = plt.figure(figsize=(6, 5))
        if stage == "scrublet":
            multi_info = adata.uns.get("multi_sample", {})
            sample_key = multi_info.get("sample_key")
            if "doublet_score" in adata.obs and sample_key in adata.obs:
                for sample_id, values in adata.obs.groupby(sample_key, observed=True)["doublet_score"]:
                    plt.hist(values, bins=30, alpha=0.45, label=str(sample_id))
                plt.xlabel("Doublet score")
                plt.ylabel("Cells")
                plt.legend(title=sample_key)
            elif "doublet_score" in adata.obs:
                sc.pl.scrublet_score_distribution(adata, show=False)
            elif adata.uns.get("scrublet", {}).get("status") == "skipped":
                self.draw_center_message(
                    plt.gca(),
                    f"Scrublet skipped\n{adata.uns['scrublet'].get('error', '')}",
                )
            else:
                self.draw_center_message(plt.gca(), "Scrublet scores not found")
            plt.title(f"{stage.upper()} Result (Node: {node_id})")
        elif stage in {"qc", "normalize"}:
            sc.pl.violin(adata, ["total_counts", "n_genes_by_counts"], jitter=0.4, show=False)
            plt.title(f"{stage.upper()} Result (Node: {node_id})")
        elif stage == "hvg":
            sc.pl.highly_variable_genes(adata, show=False)
            plt.title(f"{stage.upper()} Result (Node: {node_id})")
        elif stage == "batch_correct":
            fig = self.plot_batch_correction_result(mgr, node_id)
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
                self.draw_center_message(plt.gca(), f"Key {key} not found")
            plt.title(f"{stage.upper()} Result (Node: {node_id})")
        elif stage == "markers":
            key = node_meta.get("result_key")
            if key in adata.uns:
                sc.pl.rank_genes_groups(adata, key=key, n_genes=10, show=False)
            else:
                self.draw_center_message(plt.gca(), f"Marker key {key} not found")
            plt.title(f"{stage.upper()} Result (Node: {node_id})")
        elif stage == "annotation":
            if "X_umap" in adata.obsm and "cell_type" in adata.obs:
                sc.pl.umap(adata, color="cell_type", show=False)
            elif "cell_type" in adata.obs:
                counts = adata.obs["cell_type"].value_counts()
                counts.plot(kind="barh", ax=plt.gca())
                plt.xlabel("Cells")
            else:
                self.draw_center_message(plt.gca(), "cell_type annotations not found")
            plt.title(f"{stage.upper()} Result (Node: {node_id})")
        else:
            self.draw_center_message(plt.gca(), f"{stage} complete\nshape={adata.shape}")
            plt.title(f"{stage.upper()} Result (Node: {node_id})")

        fig = plt.gcf()
        if fig is not initial_fig:
            plt.close(initial_fig)
        self.create_artifact_from_figure(
            fig=fig,
            name="Analysis_Result",
            file_name=f"result_{node_id}.png",
            desc=f"Plot for {stage}.",
        )

        self.plot_final_umap(mgr, node_id, stage)
        self.plot_dag(mgr, node_id)
        return mgr.get_object(node_id), self.summary(mgr, node_id, stage)

    def create_artifact_from_figure(self, fig: Any, name: str, file_name: str, desc: str) -> None:
        """Serialize a matplotlib figure to a TaskWeaver artifact path."""
        bio_buf = io.BytesIO()
        fig.savefig(bio_buf, format="png", bbox_inches="tight", dpi=160)
        plt.close(fig)
        _, artifact_path = self.ctx.create_artifact_path(
            name=name,
            file_name=file_name,
            type="image",
            desc=desc,
        )
        with open(artifact_path, "wb") as handle:
            handle.write(bio_buf.getvalue())

    def draw_center_message(self, ax: Any, message: str) -> None:
        """Render a simple fallback message into a plot axis."""
        ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
        ax.set_axis_off()

    def plot_batch_correction_result(self, mgr: Any, node_id: str):
        """Plot before/after PCA panels for `visualize_result`."""
        node_meta = mgr.graph.nodes[node_id]
        adata_after = mgr.get_object(node_id)
        parent_ids = list(mgr.graph.predecessors(node_id))
        adata_before = mgr.get_object(parent_ids[0]) if parent_ids else None
        info = adata_after.uns.get("batch_correction", {})
        params = node_meta.get("params", {})
        batch_key = info.get("key") or params.get("combat_key") or params.get("sample_key")

        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        fig.suptitle(f"BATCH_CORRECT Result (Node: {node_id})", fontsize=12, fontweight="bold")

        before_err = self.plot_pca_panel(
            axes[0],
            adata_before,
            batch_key=batch_key,
            title="Before correction",
        )
        after_err = self.plot_pca_panel(
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

    def plot_pca_panel(self, ax: Any, adata: Any, batch_key: Optional[str], title: str) -> Optional[str]:
        """Draw a PCA scatter panel and return an error string if plotting is unavailable."""
        if adata is None:
            self.draw_center_message(ax, "Reference state not available")
            ax.set_title(title)
            return "Reference state not available for before/after comparison."

        plot_adata, err = self.prepare_pca_for_plot(adata)
        if plot_adata is None:
            self.draw_center_message(ax, err)
            ax.set_title(title)
            return f"{title}: {err}"

        color_key = batch_key if batch_key in plot_adata.obs else None
        plot_title = title if color_key else f"{title}\n(batch key unavailable)"

        if color_key:
            sc.pl.pca(plot_adata, color=color_key, ax=ax, show=False, title=plot_title)
        else:
            sc.pl.pca(plot_adata, ax=ax, show=False, title=plot_title)
        return None

    def prepare_pca_for_plot(self, adata: Any):
        """Ensure an AnnData object has enough PCA coordinates for comparison plotting."""
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

    def plot_final_umap(self, mgr: Any, node_id: str, stage: str) -> None:
        """Publish a final UMAP for downstream stages that already have an embedding."""
        if stage in {"umap", "cluster"}:
            return

        adata = mgr.get_object(node_id)
        if "X_umap" not in adata.obsm:
            return

        color_key = self.get_umap_color_key(mgr, node_id, adata)

        initial_fig = plt.figure(figsize=(6, 5))
        if color_key is not None:
            sc.pl.umap(adata, color=color_key, show=False)
        else:
            sc.pl.umap(adata, show=False)
        plt.title(f"Final UMAP (Node: {node_id})")

        fig = plt.gcf()
        if fig is not initial_fig:
            plt.close(initial_fig)
        self.create_artifact_from_figure(
            fig=fig,
            name="Final_UMAP",
            file_name=f"result_final_umap_{node_id}.png",
            desc=f"Final UMAP plot{f' colored by {color_key}' if color_key else ''}.",
        )

    def get_umap_color_key(self, mgr: Any, node_id: str, adata: Any) -> Optional[str]:
        """Choose the best available annotation or cluster key for UMAP coloring."""
        node_meta = mgr.graph.nodes[node_id]
        result_key = node_meta.get("result_key")
        if result_key in adata.obs:
            return result_key

        lineage = mgr.ancestry_to_node(node_id)
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

    def summary(self, mgr: Any, node_id: str, stage: str) -> str:
        """Build the result summary returned by the pipeline runner."""
        adata = mgr.get_object(node_id)
        node_meta = mgr.graph.nodes[node_id]
        result_key = node_meta.get("result_key")
        lineage = mgr.ancestry_to_node(node_id)
        integration = "none"
        for lineage_node_id in lineage:
            lineage_meta = mgr.graph.nodes[lineage_node_id]
            params = lineage_meta.get("params", {})
            if lineage_meta.get("action") == "batch_correct" and params.get("batch_correction_method") == "combat":
                integration = f"ComBat on expression (batch key: {params.get('combat_key')})"
            elif lineage_meta.get("action") == "neighbors" and params.get("integration_method") == "bbknn":
                integration = f"BBKNN neighbor graph (batch key: {params.get('integration_batch_key')})"
        return (
            f"Stage '{stage}' complete.\n"
            f"- Active node: {node_id}\n"
            f"- Data shape: {adata.n_obs} cells x {adata.n_vars} genes\n"
            f"- Integration: {integration}\n"
            f"- Result key: {result_key or 'None'}\n"
            f"- Cached DAG nodes: {len(mgr.graph.nodes)}\n"
            f"- Lineage nodes considered: {len(lineage)}"
        )

    def plot_dag(self, mgr: Any, current_node: str) -> None:
        """Publish the cached DAG after `visualize_result` draws the main plot."""
        if not mgr.graph.nodes:
            return

        pos = self.pipeline_dag_layout(mgr, current_node)
        active_lineage = mgr.ancestors_including_self(current_node)
        branch_rows = len({y for _, y in pos.values() if y < 0})

        width = min(15, max(10, len({x for x, _ in pos.values()}) * 1.15))
        height = min(12, max(4.8, 3.4 + branch_rows * 1.05))
        fig, ax = plt.subplots(figsize=(width, height))

        stages_in_graph = {attr.get("action", "?") for _, attr in mgr.graph.nodes(data=True)}
        for stage, x in self.stage_columns(mgr).items():
            if stage in stages_in_graph:
                ax.axvline(x, color="#edf1f5", linewidth=0.7, zorder=0)

        active_edges = {
            (u, v)
            for u, v in mgr.graph.edges()
            if u in active_lineage and v in active_lineage
        }
        for edge in mgr.graph.edges():
            (x0, y0), (x1, y1) = pos[edge[0]], pos[edge[1]]
            edge_color = "#2f6f4e" if edge in active_edges else "#d6dee8"
            edge_width = 2.6 if edge in active_edges else 0.9
            ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": edge_color,
                    "lw": edge_width,
                    "shrinkA": 18,
                    "shrinkB": 18,
                },
                zorder=1,
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

        xs = [pos[node_id][0] for node_id in mgr.graph.nodes]
        ys = [pos[node_id][1] for node_id in mgr.graph.nodes]
        ax.scatter(
            xs,
            ys,
            s=1900,
            c=node_colors,
            marker="o",
            edgecolors=edgecolors,
            linewidths=linewidths,
            zorder=2,
        )

        labels = {}
        for node_id, attr in mgr.graph.nodes(data=True):
            action = attr.get("action", "?")
            short_id = node_id.removeprefix("node_")[:4]
            display_action = action.replace("_", "\n")
            params = attr.get("params", {})
            details = self.dag_label_details(action, params)
            raw_nodes = mgr.raw_ancestors(node_id)
            if action in {"raw", "scrublet", "qc"} and len(raw_nodes) == 1:
                sample_id = mgr.graph.nodes[raw_nodes[0]].get("params", {}).get("sample_id")
                if sample_id:
                    details += f"\n{sample_id}"
            if node_id in active_lineage or node_id == current_node:
                labels[node_id] = f"[{short_id}]\n{display_action}{details}"
            else:
                labels[node_id] = f"[{short_id}]\n{display_action}"

        stage_y = max(y for _, y in pos.values()) + 0.65
        for node_id, label in labels.items():
            ax.text(
                pos[node_id][0],
                pos[node_id][1],
                label,
                ha="center",
                va="center",
                fontsize=7,
                fontweight="bold",
                zorder=3,
            )
        for stage, x in self.stage_columns(mgr).items():
            if any(attr.get("action", "?") == stage for _, attr in mgr.graph.nodes(data=True)):
                ax.text(
                    x,
                    stage_y,
                    stage.upper(),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                    color="#334155",
                )

        ax.axis("off")
        x_values = [x for x, _ in pos.values()]
        y_values = [y for _, y in pos.values()]
        ax.set_xlim(min(x_values) - 0.9, max(x_values) + 0.9)
        ax.set_ylim(min(y_values) - 0.75, stage_y + 0.45)

        self.create_artifact_from_figure(
            fig=fig,
            name="Pipeline_State",
            file_name=f"result_pipeline_dag_{current_node}.png",
            desc="Current pipeline DAG.",
        )

    def pipeline_dag_layout(self, mgr: Any, current_node: str) -> Dict[str, tuple[float, float]]:
        """Align the active path and keep each cached branch on a stable row."""
        active_lineage = mgr.ancestors_including_self(current_node)
        stage_columns = self.stage_columns(mgr)
        active_raws = [item for item in mgr.raw_ancestors(current_node) if item in active_lineage]
        lane_gap = 1.35
        active_lanes = {
            raw_id: ((len(active_raws) - 1) / 2 - index) * lane_gap
            for index, raw_id in enumerate(active_raws)
        }
        active_concats = [
            item
            for item in active_lineage
            if mgr.graph.nodes[item].get("action") == "concat"
        ]
        inactive_nodes = set(mgr.graph.nodes) - active_lineage
        branch_ends = [
            node_id
            for node_id in inactive_nodes
            if not any(child_id in inactive_nodes for child_id in mgr.graph.successors(node_id))
        ]
        branch_ends.sort(
            key=lambda node_id: (
                self.same_source_priority(mgr, current_node, node_id),
                tuple(mgr.raw_ancestors(node_id)),
                tuple(mgr.ancestry_to_node(node_id)),
            )
        )
        lowest_active_lane = min(active_lanes.values(), default=0.0)
        branch_rows = {
            node_id: lowest_active_lane - 1.6 - index * 1.15
            for index, node_id in enumerate(branch_ends)
        }

        pos = {}
        for node_id, attr in mgr.graph.nodes(data=True):
            stage = attr.get("action", "?")
            x = stage_columns.get(stage, len(stage_columns) * 1.25)
            if node_id in active_lineage:
                is_after_concat = any(
                    concat_id == node_id or concat_id in mgr.ancestors(node_id)
                    for concat_id in active_concats
                )
                if is_after_concat or len(active_raws) <= 1:
                    pos[node_id] = (x, 0.0)
                else:
                    contributing_raws = [
                        raw_id
                        for raw_id in active_raws
                        if raw_id == node_id or raw_id in mgr.ancestors(node_id)
                    ]
                    pos[node_id] = (x, active_lanes.get(contributing_raws[0], 0.0))
                continue

            descendants = mgr.descendants(node_id)
            branch_end = next(
                (
                    end_id
                    for end_id in branch_ends
                    if end_id == node_id or end_id in descendants
                ),
                None,
            )
            pos[node_id] = (x, branch_rows.get(branch_end, -1.4))

        return pos

    def stage_columns(self, mgr: Any) -> Dict[str, float]:
        """Assign x-positions to known and discovered DAG stages."""
        known_order = [
            "raw",
            "scrublet",
            "qc",
            "concat",
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
            }
        )
        ordered_stages.extend(extra_stages)
        return {stage: idx * 1.25 for idx, stage in enumerate(ordered_stages)}

    def same_source_priority(self, mgr: Any, current_node: str, node_id: str) -> int:
        """Prefer DAG branches rooted in the same raw dataset as the active node."""
        current_raws = frozenset(mgr.raw_ancestors(current_node))
        node_raws = frozenset(mgr.raw_ancestors(node_id))
        return 0 if current_raws == node_raws else 1

    def dag_label_details(self, action: str, params: Dict[str, Any]) -> str:
        """Add concise stage-specific parameter hints to highlighted DAG node labels."""
        if action == "qc":
            return f"\nmin={params.get('qc_min_genes', '?')}"
        if action == "concat":
            preview = "preview" if params.get("preview") else params.get("multi_sample_join", "inner")
            return f"\n{preview}\nn={len(params.get('sample_ids') or [])}"
        if action == "scrublet":
            return f"\nrate={params.get('scrublet_expected_doublet_rate', '?')}"
        if action == "hvg":
            return f"\ntop={params.get('n_hvg', '?')}"
        if action == "batch_correct":
            return f"\n{params.get('batch_correction_method', 'none')}"
        if action == "pca":
            return f"\npc={params.get('n_comps', '?')}"
        if action == "neighbors":
            if params.get("integration_method") == "bbknn":
                return f"\nBBKNN\n{params.get('integration_batch_key', '?')}"
            return f"\nk={params.get('n_neighbors', '?')}"
        if action == "cluster":
            return f"\nres={params.get('resolution', '?')}"
        if action == "markers":
            return f"\nn={params.get('n_marker_genes', '?')}"
        if action == "annotation":
            return f"\nn={params.get('n_annotation_markers', '?')}"
        return ""
