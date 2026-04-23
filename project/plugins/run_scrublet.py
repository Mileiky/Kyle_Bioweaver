import scanpy as sc
from taskweaver.plugin import Plugin, register_plugin

@register_plugin
class run_scrublet(Plugin):
    def __call__(self, adata, batch_key=None):
        adata = adata.copy()

        # Can add additional functionality
        sc.pp.scrublet(adata, batch_key=batch_key)
        return adata