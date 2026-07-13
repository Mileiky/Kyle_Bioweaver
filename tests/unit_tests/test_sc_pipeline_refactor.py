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

from project.plugins.sc_front import SingleCellPipeline
from project.sc_pipeline.sc_dag import SCStateManager
from project.sc_pipeline.sc_rules import create_default_registry
from project.sc_pipeline.sc_run import get_runner
from taskweaver.plugin.context import temp_context


def _make_test_adata():
    obs = pd.DataFrame(index=[f"cell_{idx}" for idx in range(30)])
    var = pd.DataFrame(index=["MT-1", "MT-2"] + [f"g{idx}" for idx in range(18)])

    group_a = np.tile(np.array([20, 15, 12, 10, 9, 8, 7, 6, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), (15, 1))
    group_b = np.tile(np.array([1, 1, 1, 1, 1, 2, 6, 7, 8, 9, 10, 12, 15, 18, 16, 14, 13, 12, 11, 10]), (15, 1))
    X = np.vstack([group_a, group_b]).astype(float)
    return AnnData(X=X, obs=obs, var=var)


def _base_params(data_path):
    return {
        "data_path": data_path,
        "qc_min_genes": 1,
        "qc_max_genes": 1000,
        "qc_mt_pct": None,
        "min_cells": 1,
        "n_hvg": 12,
        "n_comps": 5,
        "n_neighbors": 5,
        "n_pcs": 5,
        "resolution": 0.4,
    }


class FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({"0": "type_0", "1": "type_1", "2": "type_2"}),
                    },
                },
            ],
        }


def test_sc_pipeline_refactor_smoke(tmp_path):
    data_path = tmp_path / "synthetic.h5ad"
    dag_dir = tmp_path / "dag_cache"
    _make_test_adata().write_h5ad(data_path)

    with temp_context(str(tmp_path)) as ctx:
        runner = get_runner(ctx=ctx, config={"storage_dir": str(dag_dir)}, force_new=True)
        params = _base_params(str(data_path))

        adata1, summary1 = runner.execute(target_stage="cluster", **params)
        node1 = runner.manager.active_node_id
        graph_size_1 = len(runner.manager.graph.nodes)

        assert adata1 is not None
        assert "Stage 'cluster' complete." in summary1
        assert node1 in runner.manager.graph.nodes

        umap_node = next(iter(runner.manager.graph.predecessors(node1)), None)
        assert umap_node is not None
        assert runner.manager.graph.nodes[umap_node]["action"] == "umap"
        assert "leiden_res0.4" not in runner.manager.get_object(umap_node).obs
        assert "leiden_res0.4" in adata1.obs

        adata2, summary2 = runner.execute(target_stage="cluster", **params)
        node2 = runner.manager.active_node_id
        assert adata2 is not None
        assert "Stage 'cluster' complete." in summary2
        assert node2 == node1
        assert len(runner.manager.graph.nodes) == graph_size_1

        params_branch = dict(params, resolution=0.8)
        adata3, summary3 = runner.execute(target_stage="cluster", **params_branch)
        node3 = runner.manager.active_node_id
        assert adata3 is not None
        assert "Stage 'cluster' complete." in summary3
        assert node3 != node1
        assert len(runner.manager.graph.nodes) == graph_size_1 + 1
        assert "leiden_res0.8" in adata3.obs

        runner.manager.save()
        reloaded = SCStateManager(storage_dir=str(dag_dir), registry=create_default_registry())
        assert len(reloaded.graph.nodes) == len(runner.manager.graph.nodes)
        assert reloaded.get_object(node1).shape == adata1.shape
        assert reloaded.get_object(node3).shape == adata3.shape

        artifact_names = [artifact["name"] for artifact in ctx._artifacts]
        assert "Analysis_Result" in artifact_names
        assert "Pipeline_State" in artifact_names

    with temp_context(str(tmp_path)) as ctx:
        plugin = SingleCellPipeline(name="sc_pipeline_v4", ctx=ctx, config={"storage_dir": str(dag_dir)})
        with patch("project.sc_pipeline.sc_rules.requests.post", return_value=FakeResponse()):
            adata4, summary4 = plugin(
                target_stage="annotation",
                data_path=str(data_path),
                qc_min_genes=1,
                qc_max_genes=1000,
                qc_mt_pct=None,
                min_cells=1,
                n_hvg=12,
                n_comps=5,
                n_neighbors=5,
                n_pcs=5,
                resolution=0.4,
                n_marker_genes=5,
                n_annotation_markers=3,
                annotation_api_key="top-secret-token",
            )

        assert adata4 is not None
        assert "Stage 'annotation' complete." in summary4
        assert "cell_type" in adata4.obs
        assert any(artifact["name"] == "Final_UMAP" for artifact in ctx._artifacts)

        graph_text = (dag_dir / "graph.json").read_text(encoding="utf-8")
        assert "annotation_api_key" not in graph_text
        assert "top-secret-token" not in graph_text
