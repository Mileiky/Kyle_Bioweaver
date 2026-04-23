from taskweaver.plugin import Plugin, register_plugin
from project.utils.sc_dag import get_manager, ensure, register_raw
import scanpy as sc
import networkx as nx
import matplotlib.pyplot as plt
import os
import io  # <--- Added to handle SVG text buffers

@register_plugin
class SingleCellPipeline(Plugin):
    def __call__(self, 
                 target_stage: str, 
                 data_path: str = None, 
                 resolution: float = 0.5, 
                 n_hvg: int = 2000, 
                 min_genes: int = 200):
        """
        Manages the single-cell analysis pipeline using a DAG with caching.
        It can load data or continue from a previous state to reach the 'target_stage'.
        """
        # 1. Get the Persistent Manager
        mgr = get_manager()
        
        # 2. Handle Data Source
        start_node_id = None

        # This version initial raw multiple times
        # start
        # # Check if we already have a 'raw' node in the graph history
        existing_raw_nodes = [n for n, attr in mgr.graph.nodes(data=True) if attr.get('action') == 'raw']
        

        
        if existing_raw_nodes:
            # Case B: Continue from the existing loaded data
            start_node_id = existing_raw_nodes[0]

        elif data_path:
            # Case A: User provided a new file. Load and register it.
            # (Note: In a real app, you might check if this specific path was already loaded)
            self.ctx.log("info", "sc_pipeline", f"Loading data from {data_path}...")
            adata = sc.read(data_path)
            start_node_id = register_raw(mgr, adata)
            
        else:
            # Case C: No data in memory and no path provided
            return "❌ Error: No data loaded yet. Please provide 'data_path' for the first run."

        # 3. Run the DAG Logic
        # The 'ensure' function handles caching and dependency resolution automatically.
        try:
            final_node_id = ensure(
                mgr,
                target=target_stage,
                start_state=start_node_id,
                # Parameters map to the arguments in your Rule functions (qc_filter_rule, etc.)
                qc_min_genes=min_genes, 
                qc_mt_pct=5, # Hardcoded for now, or add to YAML to expose it
                n_hvg=n_hvg,
                resolution=resolution
            )
        except Exception as e:
            return f"❌ Pipeline Failed: {str(e)}"

        # 4. Generate Context-Specific Biological Plot
        # This helps the user decide what to do next based on the result.
        adata = mgr.states[final_node_id]
        
        # CHANGE: Use .svg extension
        bio_plot_svg = f"result_{target_stage}_{final_node_id[:4]}.svg"
        bio_plot_png = f"result_{target_stage}_{final_node_id[:4]}.png"
        plt.figure(figsize=(6, 5))
        
        if target_stage == "qc":
            sc.pl.violin(adata, ['total_counts', 'n_genes_by_counts'], jitter=0.4, show=False)
            plt.title(f"QC Metrics (Filtered: >{min_genes} genes)")
            
        elif target_stage == "hvg":
            sc.pl.highly_variable_genes(adata, show=False)
            plt.title(f"HVG Selection (Top {n_hvg})")
            
        elif target_stage == "cluster":
            sc.pl.umap(adata, color='leiden', show=False)
            plt.title(f"Clustering (Resolution {resolution})")
            
        # CHANGE: Save to memory buffer (SVG Text) instead of disk
        bio_buf = io.StringIO()
        plt.savefig(bio_buf, format="svg", bbox_inches='tight')
        bio_svg_content = bio_buf.getvalue()

        # For display to be catched by executor
        plt.show()
        plt.close()
        
        # Register the plot as an artifact for the LLM
        # the goal of add_artifact is to register and persist execution outputs so they 
        # are accessible to both the User (UI) and the LLM Agent.
        # ArtifactType = Literal["chart", "image", "df", "file", "txt", "svg", "html"]
        # The image type is not allowed by add_artifact.
        self.ctx.add_artifact(
            name="Analysis_Result",
            file_name=bio_plot_svg,
            type="svg",
            val=bio_svg_content,
            desc=f"SVG Plot showing {target_stage} results."
        )

        # 5. Generate DAG Plot
        dag_plot_filename = "pipeline_dag.svg" # CHANGE: Use .svg

        # CHANGE: Use buffer logic
        dag_buf = io.StringIO()
        self._plot_dag(mgr, final_node_id, dag_buf)
        dag_svg_content = dag_buf.getvalue()
        
        # add_artifact currently not support type = 'image'
        self.ctx.add_artifact(
            name="Pipeline_State",
            file_name=dag_plot_filename,
            type="svg",          
            val=dag_svg_content,
            desc = "Flowchart of the current pipeline execution state."
        )

        description = (
            f"✅ **Stage '{target_stage}' Complete.**\n"
            f"- **Current State ID:** `{final_node_id}`\n"
            f"- **Data Shape:** {adata.shape[0]} cells × {adata.shape[1]} genes\n"
            f"- **Action:** Generated {bio_plot_svg} and updated pipeline graph.\n"
            f"If this result looks good, you can proceed to the next stage. "
            f"If not, ask me to re-run this step with different parameters."
        )

        # 6. Return Summary
        return adata, description

    def _plot_dag(self, mgr, current_node, output_destination):
        """Helper to draw the pipeline state into the provided file/buffer."""
        plt.figure(figsize=(8, 5))
        pos = nx.spring_layout(mgr.graph, seed=42)
        
        # Color the active node differently
        node_colors = ['lightgreen' if n == current_node else 'lightblue' for n in mgr.graph.nodes()]
        
        nx.draw(mgr.graph, pos, with_labels=False, node_color=node_colors, node_size=1500, edge_color='gray')
        
        # Create readable labels
        labels = {}
        for n, attr in mgr.graph.nodes(data=True):
            action = attr.get('action', '?')
            if action == 'cluster':
                lbl = f"Cluster\n(res={attr['params'].get('resolution')})"
            elif action == 'hvg':
                lbl = f"HVG\n(n={attr['params'].get('n_hvg')})"
            else:
                lbl = action
            labels[n] = lbl
            
        nx.draw_networkx_labels(mgr.graph, pos, labels=labels, font_size=8)
        plt.title(f"Pipeline History (Active: {current_node[:6]})")
        
        # CHANGE: Save to the destination (buffer) in SVG format
        plt.savefig(output_destination, format="svg")
        plt.show()
        plt.close()