import scanpy as sc
from taskweaver.plugin import Plugin, register_plugin

# 1. PRE-PROCESSING (QC, Normalization, Scaling)
@register_plugin
class qc_filter(Plugin):
    def __call__(self, adata, min_genes=200, max_genes=2500, pct_mt=15, min_cells=3, show_plots=False):
        adata = adata.copy()

        sc.pp.filter_genes(adata, min_cells=min_cells)

        # annotate the group of mitochondrial genes as "mt"
        adata.var["mt"] = adata.var_names.str.startswith(("MT-", "mt-")) # only human and mouse mt
        sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)


        sc.pp.filter_cells(adata, min_genes=min_genes)
        sc.pp.filter_cells(adata, max_genes=max_genes)

        # Filter by mitochondrial percentage
        if pct_mt is not None:
            adata = adata[adata.obs.pct_counts_mt < pct_mt, :].copy()

        # shows the size
        print(f"Post-filtering: {adata.n_obs} cells, {adata.n_vars} genes.")

        if show_plots:
            sc.pl.violin(adata, ["n_genes_by_counts", "total_counts", "pct_counts_mt"], jitter=0.4, multi_panel=True)
            sc.pl.scatter(adata, x="total_counts", y="pct_counts_mt")
            sc.pl.scatter(adata, x="total_counts", y="n_genes_by_counts")

        return adata