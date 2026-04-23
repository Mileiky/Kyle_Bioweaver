import argparse
import json
import os
import sys

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import scanpy as sc

from project.plugins.cell_type_ann import cell_type_ann
from project.plugins.normalize_and_hvg import normalize_and_hvg
from project.plugins.qc_filter import qc_filter
from project.plugins.run_clustering import run_clustering
from project.plugins.run_pca import run_pca
from project.plugins.run_umap import run_umap
from project.plugins.scale_data import scale_data
from taskweaver.plugin.context import temp_context


DEFAULT_DATA_PATH = os.path.expanduser(
    "~/data/pbmc3k_filtered_gene_bc_matrices/filtered_gene_bc_matrices/hg19"
)


def load_adata(data_path: str):
    expanded = os.path.expanduser(data_path)
    if os.path.isdir(expanded):
        return sc.read_10x_mtx(expanded, var_names="gene_symbols", make_unique=True)
    if os.path.isfile(expanded) and expanded.endswith(".h5ad"):
        return sc.read_h5ad(expanded)
    raise ValueError(
        f"Unsupported data path: {expanded}. Expected a 10x directory or an .h5ad file."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Smoke test for the run_cell_type_annotation plugin."
    )
    parser.add_argument(
        "--data-path",
        default=DEFAULT_DATA_PATH,
        help="Path to a 10x directory or .h5ad file.",
    )
    parser.add_argument(
        "--groupby",
        default="leiden_res0.7",
        help="Cluster column to annotate.",
    )
    parser.add_argument(
        "--model",
        default="qwen3.5:122b",
        help="Model name for the annotation backend.",
    )
    parser.add_argument(
        "--api-base",
        default="http://localhost:11434/v1",
        help="OpenAI-compatible API base.",
    )
    parser.add_argument(
        "--api-key",
        default="ollama",
        help="API key for the annotation backend.",
    )
    parser.add_argument(
        "--n-markers",
        type=int,
        default=10,
        help="Top marker genes per cluster to send to the model.",
    )
    args = parser.parse_args()

    with temp_context() as ctx:
        qc = qc_filter("qc_filter", ctx, {})
        norm = normalize_and_hvg("normalize_and_hvg", ctx, {})
        scale = scale_data("scale_data", ctx, {})
        pca = run_pca("run_pca", ctx, {})
        umap = run_umap("run_umap", ctx, {})
        cluster = run_clustering("run_clustering", ctx, {})
        annotate = cell_type_ann("run_cell_type_annotation", ctx, {})

        print(f"Loading data from {os.path.expanduser(args.data_path)}")
        adata = load_adata(args.data_path)
        print(f"Loaded {adata.n_obs} cells x {adata.n_vars} genes")

        adata = qc(adata)
        adata = norm(adata)
        adata = scale(adata)
        adata = pca(adata)
        adata = umap(adata)
        adata = cluster(adata)
        adata = annotate(
            adata,
            groupby=args.groupby,
            model=args.model,
            api_base=args.api_base,
            api_key=args.api_key,
            n_markers=args.n_markers,
        )

        summary = {
            "data_path": os.path.expanduser(args.data_path),
            "groupby": args.groupby,
            "cell_type_counts": adata.obs["cell_type"].value_counts(dropna=False).to_dict(),
            "annotation_metadata": adata.uns["cell_type_annotation"],
        }
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
