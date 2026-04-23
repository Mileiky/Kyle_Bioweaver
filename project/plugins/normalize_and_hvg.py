import scanpy as sc
from taskweaver.plugin import Plugin, register_plugin

@register_plugin
class normalize_and_hvg(Plugin):
    def __call__(self, adata, n_top_genes=2000, flavor='seurat', show_plots=False):
        #adata.raw = adata
        adata = adata.copy()
        # Store raw counts for DE analysis later 
        
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor=flavor)
        if show_plots:
            sc.pl.highly_variable_genes(adata)
        
        adata.raw = adata;
        adata = adata[:, adata.var.highly_variable]

        return adata