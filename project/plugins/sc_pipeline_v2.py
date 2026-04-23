from taskweaver.plugin import Plugin, register_plugin
from project.utils.sc_dag_v2 import get_manager, ensure, register_raw
import scanpy as sc
import networkx as nx
import matplotlib.pyplot as plt
import io

@register_plugin
class SingleCellPipeline(Plugin):
    def __call__(self, 
                 target_stage: str, 
                 data_path: str = None, 
                 resolution: float = 0.5, 
                 n_hvg: int = 2000, 
                 min_genes: int = 200):
        
        mgr = get_manager()
        
        # --- 1. INTELLIGENT SEARCH (Goal 1: Locate existing) ---
        # Before doing anything, check if this EXACT request exists anywhere in history.
        # This handles the "I want to plot res=0.5 again" scenario without re-running.
        
        search_params = {}
        if target_stage == "cluster":
            search_params = {"resolution": resolution}
        elif target_stage == "qc":
            search_params = {"qc_min_genes": min_genes}
        elif target_stage == "hvg":
            search_params = {"n_hvg": n_hvg}

        self.ctx.log("info", "sc_pipeline", f"✅ search_params: {search_params}")
        found_node_id = mgr.find_node(target_stage, search_params)
        
        if found_node_id:
            self.ctx.log("info", "sc_pipeline", f"✅ Found existing analysis matching request: {found_node_id}")
            final_node_id = found_node_id
            # We skip 'ensure' and jump straight to plotting/reporting
        else:
            # --- 2. SETUP START NODE ---
            start_node_id = None
            
            # Smart Parent Search: If we need to cluster, find the best HVG/QC parent
            # that matches our 'min_genes' preference.
            if target_stage == "cluster" or target_stage == "hvg":
                # Find a QC node that matches the requested min_genes
                potential_parent = mgr.find_node("qc", {"qc_min_genes": min_genes})
                if potential_parent:
                     start_node_id = potential_parent
                     self.ctx.log("info", "sc_pipeline", f"Using existing QC parent: {start_node_id}")

            # Fallback: Load Raw if needed
            if not start_node_id:
                raw_nodes = [n for n, attr in mgr.graph.nodes(data=True) if attr.get('action') == 'raw']
                if raw_nodes:
                    start_node_id = raw_nodes[0]
                elif data_path:
                    self.ctx.log("info", "sc_pipeline", f"Loading new data from {data_path}...")
                    adata = sc.read(data_path)
                    start_node_id = register_raw(mgr, adata, data_path)
                else:
                    return "❌ Error: No data found. Please provide 'data_path'."

            # --- 3. RUN PIPELINE ---
            try:
                final_node_id = ensure(
                    mgr,
                    target=target_stage,
                    start_state=start_node_id,
                    qc_min_genes=min_genes, 
                    qc_mt_pct=5, 
                    n_hvg=n_hvg,
                    resolution=resolution
                )
            except Exception as e:
                return f"❌ Pipeline Failed: {str(e)}"

        # --- 4. VISUALIZATION (Handle Hybrid Keys) ---
        # for debug
        png_filename = f"result_{target_stage}_{final_node_id}.png"
        dag_filename = f"dag_{final_node_id}.png"

        # Get the object (Physical)
        adata = mgr.get_object(final_node_id)
        # Get the Node Metadata (Logical)
        node_meta = mgr.graph.nodes[final_node_id]
        
        plt.figure(figsize=(6, 5))
        
        if target_stage == "qc":
            sc.pl.violin(adata, ['total_counts', 'n_genes_by_counts'], jitter=0.4, show=False)
            plt.title(f"QC Metrics (Node: {final_node_id})")
            
        elif target_stage == "hvg":
            sc.pl.highly_variable_genes(adata, show=False)
            plt.title(f"HVG Selection (Node: {final_node_id})")
            
        elif target_stage == "cluster":
            # CRITICAL: Use the 'result_key' from metadata to plot the specific resolution
            # This allows Node A (res0.5) and Node B (res0.6) to use the same object correctly
            plot_key = node_meta.get("result_key", "leiden") 
            
            sc.pl.umap(adata, color=plot_key, show=False)
            plt.title(f"Clustering: {plot_key} (Node: {final_node_id})")
            
        # Capture Plot
        bio_buf = io.StringIO()
        plt.savefig(bio_buf, format="svg", bbox_inches='tight')
        
        self.ctx.add_artifact(
            name="Analysis_Result",
            file_name=f"result_{final_node_id}.svg",
            type="svg",
            val=bio_buf.getvalue(),
            desc=f"Plot for {target_stage}."
        )
        plt.show()
        #plt.savefig(png_filename, format="png", bbox_inches='tight', dpi=150)
        plt.close()

        # --- 5. DAG PLOT ---
        dag_buf = io.StringIO()
        self._plot_dag(mgr, final_node_id, dag_filename)
        
        self.ctx.add_artifact(
            name="Pipeline_State",
            file_name=f"result_pipeline_dag_{final_node_id}.svg",
            type="svg",          
            val=dag_buf.getvalue(),
            desc="Current Pipeline Graph"
        )

        return adata, f"✅ Stage '{target_stage}' Complete. Active Node: {final_node_id}"

    def _plot_dag(self, mgr, current_node, output_destination):
            plt.figure(figsize=(8, 6))
            
            # 1. Try Hierarchical Layout
            try:
                pos = nx.nx_agraph.graphviz_layout(mgr.graph, prog="dot")
            except ImportError:
                print("Graphviz not found. Falling back to spring layout.")
                pos = nx.spring_layout(mgr.graph, seed=42)

            # 2. Styling
            node_colors = ['lightgreen' if n == current_node else 'lightblue' for n in mgr.graph.nodes()]
            
            nx.draw(mgr.graph, pos, with_labels=False, node_color=node_colors, 
                    node_size=2000, edge_color='#555555', arrows=True, arrowstyle='-|>', arrowsize=20)
            
            # 3. Enhanced Labels (NOW WITH NODE IDs)
            labels = {}
            for n, attr in mgr.graph.nodes(data=True):
                action = attr.get('action', '?')
                short_id = n[:8]  # Taking first 4 chars for brevity
                
                if action == 'cluster':
                    res = attr['params'].get('resolution')
                    # Format: [ID] Action Params
                    labels[n] = f"[{short_id}]\nCluster\nres={res}"
                elif action == 'qc':
                    ming = attr['params'].get('qc_min_genes')
                    labels[n] = f"[{short_id}]\nQC\nmin={ming}"
                elif action == 'hvg':
                    n_hvg = attr['params'].get('n_hvg')
                    labels[n] = f"[{short_id}]\nHVG\ntop={n_hvg}"
                else:
                    labels[n] = f"[{short_id}]\n{action}"
                
            nx.draw_networkx_labels(mgr.graph, pos, labels=labels, font_size=9, font_weight='bold')
            
            plt.title(f"Pipeline History (Active: {current_node[:6]})")
            plt.axis('off')
            
            # Check if output_destination is a path (str) or a buffer
            if isinstance(output_destination, str):
                 plt.savefig(output_destination, format="png", dpi=150)
            else:
                 plt.savefig(output_destination, format="svg")
                 
            plt.show() # Commented out for headless execution
            plt.close()