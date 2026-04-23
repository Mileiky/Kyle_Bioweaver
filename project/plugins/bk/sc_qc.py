from taskweaver.plugin import Plugin, register_plugin
import scanpy as sc
#from project.utils.monitor import report_changes

@register_plugin
class RunQC(Plugin):
    #@report_changes  # <--- The Monitor is active
    def __call__(self, adata: sc.AnnData, min_genes: int = 200, max_mito: float = 0.05):
        """
        Filters cells by gene count and mitochondrial percentage.
        """
        # 1. Calc Mito
        adata.var['mt'] = adata.var_names.str.startswith('MT-')
        sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

        # 2. Filter
        sc.pp.filter_cells(adata, min_genes=min_genes)
        adata = adata[adata.obs['pct_counts_mt'] < (max_mito * 100), :].copy()
        
        # 3. Normalize (Standard Workflow)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        
        return "QC Complete" # The decorator will replace this with the detailed report
