import scanpy as sc
from taskweaver.plugin import Plugin, register_plugin

@register_plugin
class run_clustering(Plugin):
    def __call__(self, adata, resolution=0.7, method='leiden', show_plots=False):
        if 'neighbors' not in adata.uns:
            # Changed the error message to match your specific function name logic
            raise ValueError("Neighbors graph not found. Run 'run_umap' (which computes neighbors) first.")

        if method == 'leiden':
            key_added = f"leiden_res{resolution}"
            # Only one call needed with the correct flavor and settings
            sc.tl.leiden(
                adata, 
                resolution=resolution, 
                key_added=key_added,
                random_state=0,     
                flavor="igraph",    
                n_iterations=2,
                directed=False
            )

        elif method == 'louvain':
            key_added = f"louvain_res{resolution}"
            sc.tl.louvain(adata, resolution=resolution, key_added=key_added)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        if show_plots:
            sc.pl.umap(adata, color=[key_added])
            
        return adata