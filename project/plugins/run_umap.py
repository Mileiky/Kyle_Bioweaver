import scanpy as sc
from taskweaver.plugin import Plugin, register_plugin  

@register_plugin
class run_umap(Plugin):
    def __call__(self, adata, min_dist=0.5, spread=1.0, n_neighbors=10, n_pcs=40, use_rep='X_pca', show_plots=False):
        # Note: changed use_rep default to 'X_pca' from None
        # The parameters are the same as the defaults
        # Add graphing capability later
        # Combined run_neighbors and run_umap for simplicity
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, use_rep=use_rep)    
        
        sc.tl.umap(adata, min_dist=min_dist, spread=spread)
        if show_plots:
            # can change the colors later
            sc.pl.umap(adata, color=['CST3', 'NKG7', 'PPBP'])
        return adata