import scanpy as sc
from taskweaver.plugin import Plugin, register_plugin

@register_plugin
class scale_data(Plugin):
    def __call__(self, adata, max_value=10):
        sc.pp.regress_out(adata, ["total_counts", "pct_counts_mt"])
        sc.pp.scale(adata, max_value=max_value)
        return adata