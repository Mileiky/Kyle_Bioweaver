import scanpy as sc
from taskweaver.plugin import Plugin, register_plugin

@register_plugin
class run_pca(Plugin):
    def __call__(self, adata, n_comps=50, show_plots=False):
        sc.tl.pca(adata, n_comps=n_comps, svd_solver='arpack')
        if show_plots:
            sc.pl.pca(adata, annotate_var_explained=True, color=None) # removed color to avoid error when "CST3" is not in the dataset
            sc.pl.pca_variance_ratio(adata, n_pcs=20)
            sc.pl.pca_loadings(adata, components=(1, 2), include_lowest=True)
        return adata