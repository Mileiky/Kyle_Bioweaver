import json
import os
import sys
from unittest.mock import MagicMock, call, patch

import matplotlib.image as mpimg
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from project.plugins.sc_front import SingleCellPipeline
from project.sc_pipeline.sc_dag import SCStateManager, compute_step_hash
from project.sc_pipeline.sc_rules import create_default_registry
from project.sc_pipeline.sc_run import PipelineRequest, SingleCellPipelineRunner, get_runner
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


def _assert_artifact_is_not_blank(ctx, name):
    artifact = next(item for item in reversed(ctx._artifacts) if item["name"] == name)
    image = mpimg.imread(os.path.join(ctx._temp_dir, artifact["file_name"]))
    assert float(image[..., :3].std()) > 0.01


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
        assert isinstance(runner.manager.graph, nx.DiGraph)

        adata1, summary1 = runner.execute(target_stage="cluster", **params)
        node1 = runner.manager.active_node_id
        graph_size_1 = len(runner.manager.graph.nodes)

        assert adata1 is not None
        assert "Stage 'cluster' complete." in summary1
        assert node1 in runner.manager.graph.nodes
        assert json.loads((dag_dir / "graph.json").read_text(encoding="utf-8"))["active_node_id"] == node1

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
        assert "leiden_res0.4" not in runner.manager.get_object(umap_node).obs
        assert "leiden_res0.8" not in runner.manager.get_object(umap_node).obs

        runner.manager.save()
        reloaded = SCStateManager(storage_dir=str(dag_dir), registry=create_default_registry())
        assert len(reloaded.graph.nodes) == len(runner.manager.graph.nodes)
        assert reloaded.get_object(node1).shape == adata1.shape
        assert reloaded.get_object(node3).shape == adata3.shape

        artifact_names = [artifact["name"] for artifact in ctx._artifacts]
        assert "Analysis_Result" in artifact_names
        assert "Pipeline_State" in artifact_names
        _assert_artifact_is_not_blank(ctx, "Analysis_Result")

    with temp_context(str(tmp_path)) as ctx:
        plugin = SingleCellPipeline(name="sc_front", ctx=ctx, config={"storage_dir": str(dag_dir)})
        session_dag_dir = dag_dir / ctx.session_id
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
        _assert_artifact_is_not_blank(ctx, "Analysis_Result")
        _assert_artifact_is_not_blank(ctx, "Final_UMAP")

        graph_text = (session_dag_dir / "graph.json").read_text(encoding="utf-8")
        assert "annotation_api_key" not in graph_text
        assert "top-secret-token" not in graph_text


def test_multi_sample_loading_and_runner_reuse(tmp_path):
    paths = [tmp_path / "sample_a.h5ad", tmp_path / "sample_b.h5ad"]
    for path in paths:
        _make_test_adata().write_h5ad(path)

    dag_dir = tmp_path / "dag_cache"
    config = {"storage_dir": str(dag_dir)}
    with temp_context(str(tmp_path)) as ctx:
        runner = get_runner(ctx=ctx, config=config, force_new=True)
        assert get_runner(ctx=ctx, config=config) is runner

        adata, summary = runner.execute(
            target_stage="raw",
            data_paths=[str(path) for path in paths],
            sample_ids=["sample_a", "sample_b"],
        )

        assert adata.n_obs == 60
        assert set(adata.obs["sample"].astype(str)) == {"sample_a", "sample_b"}
        assert adata.uns["multi_sample"]["sample_ids"] == ["sample_a", "sample_b"]
        assert "Stage 'raw' complete." in summary


def test_strict_lookup_accepts_legacy_missing_none_param(tmp_path):
    data_path = tmp_path / "source.h5ad"
    data_path.write_text("source fingerprint", encoding="utf-8")
    manager = SCStateManager(storage_dir=str(tmp_path / "dag"), registry=create_default_registry())
    full_params = {
        "data_path": str(data_path),
        "scrublet_batch_key": None,
        "scrublet_expected_doublet_rate": 0.05,
        "scrublet_threshold": None,
        "scrublet_n_prin_comps": 30,
        "scrublet_filter_doublets": False,
        "scrublet_skip_on_failure": True,
        "qc_min_genes": 200,
        "qc_max_genes": 2500,
        "qc_mt_pct": 5,
        "min_cells": 3,
        "target_sum": 10000.0,
        "n_hvg": 2000,
        "hvg_flavor": "seurat",
        "hvg_batch_key": None,
    }
    raw_hash = compute_step_hash(manager, "raw", "init", full_params)
    nodes = [
        ("raw", "raw", {"data_path": str(data_path)}),
        ("scrublet", "scrublet", {key: full_params[key] for key in manager.registry.get("scrublet").param_keys}),
        ("qc", "qc", {key: full_params[key] for key in manager.registry.get("qc").param_keys}),
        ("normalize", "normalize", {"target_sum": 10000.0}),
        # This legacy node predates hvg_batch_key and therefore has no exact current hash.
        ("hvg", "hvg", {"n_hvg": 2000, "hvg_flavor": "seurat"}),
    ]
    for index, (node_id, action, params) in enumerate(nodes):
        manager.graph.add_node(node_id, action=action, params=params, hash=raw_hash if action == "raw" else None)
        if index:
            manager.graph.add_edge(nodes[index - 1][0], node_id)

    assert manager.find_node_strict("hvg", **full_params) == "hvg"


def test_ambiguous_cache_match_rebuilds_from_exact_ancestor():
    runner = object.__new__(SingleCellPipelineRunner)
    runner.ctx = MagicMock()
    runner.manager = MagicMock()
    runner.manager.find_node_smart.return_value = (["stale-a", "stale-b"], "ambiguous")
    runner.select_nearest_compatible_ancestor = MagicMock(return_value="exact-ancestor")
    runner.register_or_reuse_raw_node = MagicMock()
    runner.ensure = MagicMock(return_value="rebuilt-target")
    request = PipelineRequest(target_stage="umap", params={"data_path": "/data/source"})

    result = runner.resolve_request(request)

    assert result == "rebuilt-target"
    runner.ensure.assert_called_once_with(
        target="umap",
        start_state="exact-ancestor",
        data_path="/data/source",
    )
    runner.register_or_reuse_raw_node.assert_not_called()


def test_plugin_raises_pipeline_failures():
    ctx = MagicMock()
    ctx.session_id = "test-session"
    plugin = SingleCellPipeline(name="sc_front", ctx=ctx, config={})
    runner = MagicMock()
    runner.execute.side_effect = ValueError("invalid pipeline request")

    with patch("project.plugins.sc_front.get_runner", return_value=runner):
        with pytest.raises(RuntimeError, match="Single-cell pipeline failed: invalid pipeline request"):
            plugin(target_stage="umap", data_path="/data/source")


def test_plugin_uses_session_specific_storage(tmp_path):
    storage_root = tmp_path / "dag_cache"
    config = {"storage_dir": str(storage_root)}
    contexts = [MagicMock(), MagicMock()]
    contexts[0].session_id = "chat-a"
    contexts[1].session_id = "chat-b"
    runner = MagicMock()
    runner.execute.return_value = (None, "complete")

    with patch("project.plugins.sc_front.get_runner", return_value=runner) as get_runner_mock:
        plugin_a = SingleCellPipeline(name="sc_front", ctx=contexts[0], config=config)
        plugin_b = SingleCellPipeline(name="sc_front", ctx=contexts[1], config=config)
        plugin_a(target_stage="raw", data_path="/data/source")
        plugin_a(target_stage="raw", data_path="/data/source")
        plugin_b(target_stage="raw", data_path="/data/source")

    assert get_runner_mock.call_args_list == [
        call(ctx=contexts[0], config=config, storage_dir=str(storage_root / "chat-a")),
        call(ctx=contexts[0], config=config, storage_dir=str(storage_root / "chat-a")),
        call(ctx=contexts[1], config=config, storage_dir=str(storage_root / "chat-b")),
    ]
