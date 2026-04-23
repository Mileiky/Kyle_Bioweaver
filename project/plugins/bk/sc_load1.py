from taskweaver.plugin import Plugin, register_plugin
import scanpy as sc
import os

# --- SERVER FIX: Prevent Matplotlib from hanging ---
import matplotlib
matplotlib.use('Agg')
# ---------------------------------------------------

# Import our helper modules
from project.utils.monitor import report_changes
from project.utils.state_graph import graph_manager

@register_plugin
class LoadDataPlugin(Plugin):
    def __call__(self, file_path: str):
        # 1. Validation
        if not os.path.exists(file_path):
             return None, f"❌ Error: File not found at {file_path}"

        # 2. Logic: Use read_h5ad explicitly as requested
        try:
            if file_path.endswith('.h5ad'):
                adata = sc.read_h5ad(file_path)
            else:
                # Fallback for other formats (csv, mtx)
                adata = sc.read(file_path)
            
            adata.var_names_make_unique()
        except Exception as e:
            return None, f"❌ Error loading file: {str(e)}"
        
        # 3. Graph Logic
        # We initialize the graph here because this is the start of the workflow
        graph_manager._initialize()
        graph_manager.add_step("LoadData", f"Raw Data\n{adata.n_obs} cells")
        
        # 4. Artifacts (Save the graph image)
        img_path = graph_manager.visualize()
        self.ctx.add_artifact("Workflow Graph", img_path, "image")
        
        return adata, f"✅ Data Loaded. Shape: {adata.shape}."