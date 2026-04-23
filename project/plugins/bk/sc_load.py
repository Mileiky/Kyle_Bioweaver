from taskweaver.plugin import Plugin, register_plugin
import scanpy as sc
import os
from project.utils.monitor import report_changes
from project.utils.state_graph import graph_manager

@register_plugin
class LoadData(Plugin):
    def __call__(self, file_path: str):
        # Ensure path is valid
        if not os.path.exists(file_path):
             return None, f"❌ Error: File not found at {file_path}"

        # Load
        print("Start loading the data")
        adata = sc.read_h5ad(file_path)
        adata.var_names_make_unique()
        
        # Initialize Graph, not working. Will need to dev later
        # graph_manager._initialize()
        # graph_manager.add_step("LoadData", f"Raw Data\n{adata.n_obs} cells")
        
        # img_path = graph_manager.visualize()
        # self.ctx.add_artifact("Workflow Graph", img_path, "image")
        
        return adata, f"✅ Data Loaded. Shape: {adata.shape}."
