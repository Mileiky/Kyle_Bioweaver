import scanpy as sc
from taskweaver.plugin import Plugin, register_plugin

@register_plugin
class find_marker_genes(Plugin):
    def __call__(self, adata, groupby='leiden_res0.7', method='t-test', n_genes=25, show_plots=False):
        sc.tl.rank_genes_groups(adata, groupby, mask_var="highly_variable", method=method)

        if show_plots:
            # Standard ranking plot
            sc.pl.rank_genes_groups(adata, n_genes=n_genes, sharey=False)
            sc.pl.rank_genes_groups_heatmap(adata, n_genes=n_genes, groupby=groupby, show_gene_labels=True)

        return adata