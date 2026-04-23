from taskweaver.plugin import Plugin, register_plugin
from project.utils.sc_dag_v4 import get_manager, ensure, register_raw, compute_step_hash
import scanpy as sc
import networkx as nx
import matplotlib.pyplot as plt
import io

@register_plugin
class SingleCellPipeline(Plugin):
    def __call__(self, 
                 target_stage: str, 
                 data_path: str = None, 
                 # Explicit parameters for clarity
                 resolution: float = 0.5, 
                 n_hvg: int = 2000, 
                 min_genes: int = 200,
                 **kwargs):
        
        mgr = get_manager()
        
        # 0. PARAMETER COLLECTION
        all_params = {
            "data_path": data_path,
            "resolution": resolution,
            "n_hvg": n_hvg,
            "qc_min_genes": min_genes, 
            "qc_mt_pct": 5,
            **kwargs
        }
        
        # --- 1. SMART SEARCH (User Intent) ---
        # Goal: Check if the final result already exists
        self.ctx.log("info", "sc_pipeline", f"🔍 Smart Search: '{target_stage}' with params: {all_params}")
        result_node_id, match_type = mgr.find_node_smart(target_stage, **all_params)
        
        if match_type == "exact_match":
            self.ctx.log("info", "sc_pipeline", f"✅ Exact Lineage Match Found: {result_node_id}")
            return self._visualize_result(mgr, result_node_id, target_stage)

        elif match_type == "fuzzy_match":
            self.ctx.log("warning", "sc_pipeline", f"⚠️ Fuzzy Match Found ({result_node_id}), but lineage differs. Starting new branch to ensure correctness.")
            # We DO NOT return here. We proceed to execution to generate the CORRECT branch.

        elif match_type == "ambiguous":
            return f"❌ Ambiguous Request! Found multiple partial matches. Please specify upstream parameters to clarify."

        # --- 2. EXECUTION FALLBACK (Deep Ancestor Search) ---
        self.ctx.log("info", "sc_pipeline", "⚙️ Locating valid anchor point in history...")
        
        start_node_id = None
        
        # A. Walk Backwards to find the Deepest Valid Ancestor
        # We need to find the node closest to our target that matches our STRICT lineage requirements.
        curr_stage = target_stage
        ancestor_chain = []
        
        # Build dependency chain: e.g. ["cluster", "hvg", "qc"]
        # (We skip the target itself because we already know it doesn't exist)
        try:
            while curr_stage != "raw":
                rule = mgr.registry.get(curr_stage)
                if not rule.requires: break
                parent = rule.requires[0] # Linear assumption
                ancestor_chain.append(parent)
                curr_stage = parent
        except Exception:
            pass # Handle root or errors gracefully

        # Check ancestors in order (Nearest -> Farthest)
        for stage in ancestor_chain:
            # STRICT SEARCH ONLY!
            # We forbid fuzzy matches here. If min_genes changed, QC strict search MUST fail.
            node_id = mgr.find_node_strict(stage, **all_params)
            
            if node_id:
                start_node_id = node_id
                self.ctx.log("info", "sc_pipeline", f"   -> Anchor Found: {stage.upper()} (Node {node_id})")
                break # Found the nearest valid parent!
        
        # B. Fallback to Raw Data (If no ancestors matched)
        if not start_node_id:
            self.ctx.log("info", "sc_pipeline", "   -> No valid ancestors found. Falling back to RAW data.")
            # 1. Calculate the hash we expect for this specific file
            target_raw_hash = compute_step_hash(mgr, "raw", "init", all_params)
            # 2. Check if we already have a node with this hash
            if target_raw_hash in mgr.hash_index:
                start_node_id = mgr.hash_index[target_raw_hash]
                self.ctx.log("info", "sc_pipeline", f"   -> Found existing Raw Data node for {data_path}: {start_node_id}")

            elif data_path:
                self.ctx.log("info", "sc_pipeline", f"Loading new data from {data_path}...")
                adata = sc.read(data_path)
                start_node_id = register_raw(mgr, adata, data_path)
            else:
                return "❌ Error: No data found and no valid parent state exists. Please provide 'data_path'."

        # C. Run Pipeline
        try:
            final_node_id = ensure(
                mgr,
                target=target_stage,
                start_state=start_node_id,
                **all_params 
            )
        except Exception as e:
            return f"❌ Pipeline Failed: {str(e)}"

        # --- 3. VISUALIZATION ---
        return self._visualize_result(mgr, final_node_id, target_stage)

    def _visualize_result(self, mgr, node_id, stage):
        """Helper to handle plotting and artifact generation."""
        adata = mgr.get_object(node_id)
        node_meta = mgr.graph.nodes[node_id]
        
        plt.figure(figsize=(6, 5))
        
        # Check action, not params, for robust plotting logic
        if stage == "qc":
            sc.pl.violin(adata, ['total_counts', 'n_genes_by_counts'], jitter=0.4, show=False)
        elif stage == "hvg":
            sc.pl.highly_variable_genes(adata, show=False)
        elif stage == "cluster":
            # Use the specific key stored in metadata, not a guess
            key = node_meta.get("result_key", "leiden")
            if key in adata.obs:
                sc.pl.umap(adata, color=key, show=False)
            else:
                plt.text(0.5, 0.5, f"Key {key} not found", ha='center')
        
        plt.title(f"{stage.upper()} Result (Node: {node_id})")
        plt.show()
        # Save Plot
        bio_buf = io.StringIO()
        plt.savefig(bio_buf, format="svg", bbox_inches='tight')
        plt.close()
        
        self.ctx.add_artifact(
            name="Analysis_Result",
            file_name=f"result_{node_id}.svg",
            type="svg",
            val=bio_buf.getvalue(),
            desc=f"Plot for {stage}."
        )

        # Plot DAG
        self._plot_dag(mgr, node_id)
        return mgr.get_object(node_id), f"✅ Stage '{stage}' Complete. Active Node: {node_id}"

    def _plot_dag(self, mgr, current_node):
        plt.figure(figsize=(8, 6))
        
        try:
            pos = nx.nx_agraph.graphviz_layout(mgr.graph, prog="dot")
        except ImportError:
            pos = nx.spring_layout(mgr.graph, seed=42)

        node_colors = ['lightgreen' if n == current_node else 'lightblue' for n in mgr.graph.nodes()]
        
        nx.draw(mgr.graph, pos, with_labels=False, node_color=node_colors, 
                node_size=2000, edge_color='#555555', arrows=True, arrowstyle='-|>', arrowsize=20)
        
        # Robust Labels (Fixing the previous issue)
        labels = {}
        for n, attr in mgr.graph.nodes(data=True):
            action = attr.get('action', '?')
            short_id = n[:4]
            params = attr.get('params', {})
            details = ""
            
            if action == "cluster":
                details = f"\nres={params.get('resolution', '?')}"
            elif action == "hvg":
                details = f"\ntop={params.get('n_hvg', '?')}"
            elif action == "qc":
                # Handle mapped names
                val = params.get('qc_min_genes', params.get('min_genes', '?'))
                details = f"\nmin={val}"
            
            labels[n] = f"[{short_id}]\n{action}{details}"
            
        nx.draw_networkx_labels(mgr.graph, pos, labels=labels, font_size=8)
        plt.show()
        dag_buf = io.StringIO()
        plt.savefig(dag_buf, format="svg")
        plt.close()
        
        self.ctx.add_artifact(
            name="Pipeline_State",
            file_name=f"result_pipeline_dag_{current_node}.svg",
            type="svg",          
            val=dag_buf.getvalue(),
            desc="Current Pipeline Graph"
        )