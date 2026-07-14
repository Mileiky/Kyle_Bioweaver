import json
import requests
import scanpy as sc
from taskweaver.plugin import Plugin, register_plugin


@register_plugin
class cell_type_ann(Plugin):
    def __call__(
        self,
        adata,
        groupby="leiden_res0.7",
        model="qwen3.5:122b",
        api_base="http://localhost:11434/v1",
        api_key="ollama",
        n_markers=10,
    ):
        if groupby not in adata.obs.columns:
            raise ValueError(f"'{groupby}' not found in adata.obs")

        if adata.raw is None:
            raise ValueError(
                "adata.raw is None. Save log-normalized data to adata.raw before scaling."
            )

        result_key = f"markers_{groupby}"
        marker_adata = adata.raw.to_adata()[:, adata.var_names].copy()
        marker_adata.obs = adata.obs.copy()

        # Find marker genes from raw log-normalized data
        sc.tl.rank_genes_groups(
            marker_adata,
            groupby=groupby,
            method="wilcoxon",
            n_genes=n_markers,
            key_added=result_key,
            use_raw=False,
        )

        markers = sc.get.rank_genes_groups_df(
            marker_adata,
            group=None,
            key=result_key,
        )

        cluster_markers = {}
        for cluster in sorted(adata.obs[groupby].astype(str).unique()):
            genes = (
                markers[markers["group"].astype(str) == cluster]["names"]
                .astype(str)
                .head(n_markers)
                .tolist()
            )
            cluster_markers[cluster] = genes

        prompt = {
            "task": "Annotate scRNA-seq clusters from marker genes.",
            "instructions": 'Return JSON only in this format: {"0":"cell type","1":"cell type"}',
            "clusters": cluster_markers,
        }

        response = requests.post(
            f"{api_base.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert in single-cell RNA-seq cell-type annotation.",
                    },
                    {
                        "role": "user",
                        "content": json.dumps(prompt),
                    },
                ],
                "temperature": 0,
                "response_format": {"type": "json_object"},
            },
            timeout=120,
        )
        response.raise_for_status()

        annotations = json.loads(response.json()["choices"][0]["message"]["content"])

        adata.obs["cell_type"] = adata.obs[groupby].astype(str).map(annotations)

        # Save metadata so results persist inside adata
        adata.uns["cell_type_annotation"] = {
            "groupby": groupby,
            "model": model,
            "cluster_markers": cluster_markers,
            "annotations": annotations,
            "result_key": result_key,
        }

        return adata
