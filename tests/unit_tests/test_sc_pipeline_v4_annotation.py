import json
import os
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
from anndata import AnnData

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from project.plugins import sc_pipeline_v4
from project.plugins.sc_pipeline_v4 import SingleCellPipeline
from project.utils.sc_dag_v4 import (
    Rule,
    SCStateManager,
    cluster_rule,
    compute_step_hash,
    hvg_rule,
    markers_rule,
    neighbors_rule,
    normalize_rule,
    pca_rule,
    qc_filter_rule,
    scale_rule,
    umap_rule,
    annotation_rule,
)
from taskweaver.plugin.context import temp_context


def _register_rules(mgr):
    mgr.registry.register(Rule("qc", ["raw"], qc_filter_rule))
    mgr.registry.register(Rule("normalize", ["qc"], normalize_rule))
    mgr.registry.register(Rule("hvg", ["normalize"], hvg_rule))
    mgr.registry.register(Rule("scale", ["hvg"], scale_rule))
    mgr.registry.register(Rule("pca", ["scale"], pca_rule))
    mgr.registry.register(Rule("neighbors", ["pca"], neighbors_rule))
    mgr.registry.register(Rule("umap", ["neighbors"], umap_rule))
    mgr.registry.register(Rule("cluster", ["umap"], cluster_rule, virtual=True))
    mgr.registry.register(Rule("markers", ["cluster"], markers_rule))
    mgr.registry.register(Rule("annotation", ["cluster"], annotation_rule))


def _seed_cluster_lineage(mgr, adata, params):
    parent_id = None
    parent_hash = "init"
    for stage in ["raw", "qc", "normalize", "hvg", "scale", "pca", "neighbors", "umap", "cluster"]:
        hash_val = compute_step_hash(mgr, stage, parent_hash, params)
        result_key = None
        if stage == "pca":
            result_key = "X_pca"
        elif stage == "neighbors":
            result_key = "neighbors"
        elif stage == "umap":
            result_key = "X_umap"
        elif stage == "cluster":
            result_key = "leiden_res0.5"

        node_params = {"data_path": params["data_path"]} if stage == "raw" else {}
        parent_id = mgr.register_new_object(
            adata.copy(),
            parent_id,
            stage,
            node_params,
            hash_val,
            result_key,
        )
        parent_hash = hash_val
    return parent_id


def test_sc_pipeline_v4_annotation_stage_annotates_cached_cluster(tmp_path):
    obs = pd.DataFrame({"leiden_res0.5": pd.Categorical(["0", "0", "1", "1"])})
    var = pd.DataFrame(index=["g1", "g2", "g3", "g4"])
    adata = AnnData(
        X=np.array(
            [
                [10.0, 8.0, 1.0, 1.0],
                [9.0, 7.0, 1.0, 1.0],
                [1.0, 1.0, 9.0, 8.0],
                [1.0, 1.0, 8.0, 9.0],
            ],
        ),
        obs=obs,
        var=var,
    )
    adata.raw = adata.copy()
    adata.obsm["X_umap"] = np.array([[0.0, 0.0], [0.1, 0.0], [1.0, 1.0], [1.1, 1.0]])
    adata = adata[:, ["g1", "g3"]].copy()

    params = {
        "data_path": str(tmp_path / "input.h5ad"),
        "qc_min_genes": 200,
        "qc_max_genes": 2500,
        "qc_mt_pct": 5,
        "min_cells": 3,
        "target_sum": 1e4,
        "n_hvg": 2000,
        "hvg_flavor": "seurat",
        "max_scale_value": 10,
        "regress_out": True,
        "n_comps": 50,
        "n_neighbors": 10,
        "n_pcs": 40,
        "use_rep": "X_pca",
        "min_dist": 0.5,
        "spread": 1.0,
        "resolution": 0.5,
        "cluster_method": "leiden",
        "groupby": None,
        "marker_method": "wilcoxon",
        "n_marker_genes": 25,
        "model": "qwen3.5:122b",
        "api_base": "http://localhost:11434/v1",
        "api_key": "ollama",
        "n_markers": 2,
    }

    mgr = SCStateManager(storage_dir=str(tmp_path / "sc_dag_v4"))
    _register_rules(mgr)
    _seed_cluster_lineage(mgr, adata, params)

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps({"0": "type_0", "1": "type_1"}),
                        },
                    },
                ],
            }

    with temp_context() as ctx:
        plugin = SingleCellPipeline("sc_pipeline_v4", ctx, {})
        with patch.object(sc_pipeline_v4, "get_manager", return_value=mgr), patch(
            "project.utils.sc_dag_v4.requests.post",
            return_value=FakeResponse(),
        ):
            result, summary = plugin(target_stage="annotation", n_markers=2)

    assert result.obs["cell_type"].tolist() == ["type_0", "type_0", "type_1", "type_1"]
    assert result.uns["cell_type_annotation"]["groupby"] == "leiden_res0.5"
    assert result.uns["cell_type_annotation"]["annotations"] == {"0": "type_0", "1": "type_1"}
    assert "Stage 'annotation' complete" in summary
