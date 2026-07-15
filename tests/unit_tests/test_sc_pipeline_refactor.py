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
from project.sc_pipeline.sc_rules import (
    batch_correct_rule,
    concat_rule,
    create_default_registry,
    neighbors_rule,
    scrublet_rule,
)
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


def test_scrublet_rule_preserves_scanpy_results_and_parent():
    parent = _make_test_adata()
    manager = MagicMock()
    manager.get_object.return_value = parent

    def add_scrublet_results(adata, **kwargs):
        adata.obs["doublet_score"] = np.linspace(0, 1, adata.n_obs)
        adata.obs["predicted_doublet"] = [True] + [False] * (adata.n_obs - 1)
        adata.uns["scrublet"] = {
            "doublet_scores_sim": np.array([0.2, 0.8]),
            "parameters": {"expected_doublet_rate": kwargs["expected_doublet_rate"]},
        }

    with patch("project.sc_pipeline.sc_rules.sc.pp.scrublet", side_effect=add_scrublet_results) as scrublet:
        result, result_type, result_key = scrublet_rule(
            manager,
            "raw-node",
            scrublet_filter_doublets=True,
        )

    assert "doublet_score" not in parent.obs
    assert result.n_obs == parent.n_obs - 1
    assert result.uns["scrublet"]["doublet_scores_sim"].tolist() == [0.2, 0.8]
    assert result_type == "new_object"
    assert result_key == "scrublet_filtered"
    assert scrublet.call_args.kwargs["n_prin_comps"] == parent.n_vars - 1


def test_scrublet_rule_can_skip_or_raise_failures():
    manager = MagicMock()
    manager.get_object.return_value = _make_test_adata()

    with patch("project.sc_pipeline.sc_rules.sc.pp.scrublet", side_effect=RuntimeError("cannot score")):
        result, _, result_key = scrublet_rule(manager, "raw-node")
        assert result.uns["scrublet"] == {"status": "skipped", "error": "cannot score"}
        assert result_key == "scrublet_skipped"

        with pytest.raises(RuntimeError, match="cannot score"):
            scrublet_rule(manager, "raw-node", scrublet_skip_on_failure=False)


def test_neighbors_rule_uses_bbknn_without_ordinary_neighbors():
    parent = _make_test_adata()
    parent.obs["sample"] = ["a"] * 15 + ["b"] * 15
    parent.obsm["X_pca"] = np.ones((parent.n_obs, 5))
    manager = MagicMock()
    manager.get_object.return_value = parent

    with (
        patch("project.sc_pipeline.sc_rules.sc.external.pp.bbknn") as bbknn,
        patch("project.sc_pipeline.sc_rules.sc.pp.neighbors") as ordinary_neighbors,
    ):
        result, result_type, result_key = neighbors_rule(
            manager,
            "pca-node",
            integration_method="bbknn",
            integration_batch_key="sample",
        )

    bbknn.assert_called_once_with(result, batch_key="sample")
    ordinary_neighbors.assert_not_called()
    assert result is not parent
    np.testing.assert_array_equal(result.X, parent.X)
    assert result_type == "new_object"
    assert result_key == "neighbors"


def test_neighbors_rule_validates_bbknn_batch_key():
    parent = _make_test_adata()
    parent.obsm["X_pca"] = np.ones((parent.n_obs, 5))
    manager = MagicMock()
    manager.get_object.return_value = parent

    with pytest.raises(ValueError, match="BBKNN batch key 'sample' not found"):
        neighbors_rule(
            manager,
            "pca-node",
            integration_method="bbknn",
            integration_batch_key="sample",
        )


def test_combat_still_changes_expression_before_pca():
    parent = _make_test_adata()
    parent.obs["sample"] = ["a"] * 15 + ["b"] * 15
    manager = MagicMock()
    manager.get_object.return_value = parent

    def shift_expression(adata, key, inplace):
        assert key == "sample"
        assert inplace is True
        adata.X += 1

    with patch("project.sc_pipeline.sc_rules.sc.pp.combat", side_effect=shift_expression) as combat:
        result, result_type, result_key = batch_correct_rule(
            manager,
            "hvg-node",
            batch_correction_method="combat",
            combat_key="sample",
        )

    combat.assert_called_once()
    assert not np.array_equal(result.X, parent.X)
    np.testing.assert_array_equal(parent.X, _make_test_adata().X)
    assert result.uns["batch_correction"] == {"method": "combat", "key": "sample"}
    assert result_type == "new_object"
    assert result_key == "combat_sample"
    registry = create_default_registry()
    assert registry.get("scale").requires == ["batch_correct"]
    assert registry.get("pca").requires == ["scale"]


def test_runner_resolves_integration_methods(tmp_path):
    with temp_context(str(tmp_path)) as ctx:
        runner = get_runner(ctx=ctx, config={"storage_dir": str(tmp_path / "dag")}, force_new=True)

        single = runner.normalize_request(target_stage="neighbors", data_path="single.h5ad")
        assert single.params["integration_method"] == "none"
        assert single.params["batch_correction_method"] == "none"

        multi = runner.normalize_request(
            target_stage="neighbors",
            data_paths=["a.h5ad", "b.h5ad"],
        )
        assert multi.params["integration_method"] == "bbknn"
        assert multi.params["integration_batch_key"] == "sample"
        assert multi.params["batch_correction_method"] == "none"

        explicit_none = runner.normalize_request(
            target_stage="neighbors",
            data_paths=["a.h5ad", "b.h5ad"],
            integration_method="none",
        )
        assert explicit_none.params["integration_method"] == "none"

        explicit_bbknn = runner.normalize_request(
            target_stage="neighbors",
            data_path="single.h5ad",
            integration_method="bbknn",
            integration_batch_key="donor",
        )
        assert explicit_bbknn.params["integration_method"] == "bbknn"
        assert explicit_bbknn.params["integration_batch_key"] == "donor"

        combat = runner.normalize_request(
            target_stage="neighbors",
            data_path="single.h5ad",
            integration_method="combat",
            integration_batch_key="donor",
        )
        assert combat.params["integration_method"] == "none"
        assert combat.params["batch_correction_method"] == "combat"
        assert combat.params["combat_key"] == "donor"

        legacy_combat = runner.normalize_request(
            target_stage="neighbors",
            data_paths=["a.h5ad", "b.h5ad"],
            batch_correction_method="combat",
        )
        assert legacy_combat.params["integration_method"] == "none"
        assert legacy_combat.params["batch_correction_method"] == "combat"
        assert legacy_combat.params["combat_key"] == "sample"

        with pytest.raises(ValueError, match="Conflicting integration choices"):
            runner.normalize_request(
                target_stage="neighbors",
                data_paths=["a.h5ad", "b.h5ad"],
                integration_method="bbknn",
                batch_correction_method="combat",
            )


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
        assert "Integration: none" in summary1
        assert node1 in runner.manager.graph.nodes
        assert json.loads((dag_dir / "graph.json").read_text(encoding="utf-8"))["active_node_id"] == node1

        umap_node = next(iter(runner.manager.graph.predecessors(node1)), None)
        assert umap_node is not None
        assert runner.manager.graph.nodes[umap_node]["action"] == "umap"
        assert "leiden_res0.4" not in runner.manager.get_object(umap_node).obs
        assert "leiden_res0.4" in adata1.obs

        _, marker_summary = runner.execute(target_stage="markers")
        marker_node = runner.manager.active_node_id
        assert "Stage 'markers' complete." in marker_summary
        assert list(runner.manager.graph.predecessors(marker_node)) == [node1]
        assert len(runner.manager.graph.nodes) == graph_size_1 + 1

        adata2, summary2 = runner.execute(target_stage="cluster", **params)
        node2 = runner.manager.active_node_id
        assert adata2 is not None
        assert "Stage 'cluster' complete." in summary2
        assert node2 == node1
        assert len(runner.manager.graph.nodes) == graph_size_1 + 1

        params_branch = {"resolution": 0.8}
        adata3, summary3 = runner.execute(target_stage="cluster", **params_branch)
        node3 = runner.manager.active_node_id
        assert adata3 is not None
        assert "Stage 'cluster' complete." in summary3
        assert node3 != node1
        assert len(runner.manager.graph.nodes) == graph_size_1 + 2
        assert "leiden_res0.8" in adata3.obs
        assert "leiden_res0.4" not in runner.manager.get_object(umap_node).obs
        assert "leiden_res0.8" not in runner.manager.get_object(umap_node).obs

        runner.manager.save()
        reloaded = SCStateManager(storage_dir=str(dag_dir), registry=create_default_registry())
        assert len(reloaded.graph.nodes) == len(runner.manager.graph.nodes)
        assert reloaded.get_object(node1).shape == adata1.shape
        assert reloaded.get_object(node3).shape == adata3.shape
        reloaded_runner = SingleCellPipelineRunner(ctx=ctx, manager=reloaded, registry=reloaded.registry)
        continued = reloaded_runner.normalize_request(target_stage="markers")
        assert continued.params["qc_max_genes"] == params["qc_max_genes"]
        assert continued.params["qc_mt_pct"] == params["qc_mt_pct"]
        assert continued.params["resolution"] == 0.8

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
        assert adata4.uns["cell_type_annotation"]["model"] == "qwen3.5:122b"
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
        assert "Stage 'raw' complete." in summary
        concat_node = runner.manager.active_node_id
        assert runner.manager.graph.nodes[concat_node]["action"] == "concat"
        parents = runner.manager.parent_ids(concat_node)
        assert [runner.manager.graph.nodes[item]["action"] for item in parents] == ["raw", "raw"]


def test_manager_persists_ordered_multiple_parents(tmp_path):
    manager = SCStateManager(storage_dir=str(tmp_path / "dag"), registry=create_default_registry())
    left = manager.register_new_object(
        _make_test_adata(), parent_id=None, action="raw", params={"data_path": "left"}, hash_val="left"
    )
    right = manager.register_new_object(
        _make_test_adata(), parent_id=None, action="raw", params={"data_path": "right"}, hash_val="right"
    )
    merged = manager.register_new_object(
        _make_test_adata(),
        parent_id=None,
        parent_ids=[right, left],
        action="concat",
        params={"sample_ids": ["right", "left"]},
        hash_val="merged",
    )

    assert manager.parent_ids(merged) == [right, left]
    assert manager.raw_ancestors(merged) == [right, left]
    assert manager.ancestry_to_node(merged) == [right, left, merged]
    with pytest.raises(ValueError, match="multiple parents"):
        manager.lineage_to_node(merged)

    reloaded = SCStateManager(storage_dir=str(tmp_path / "dag"), registry=create_default_registry())
    assert reloaded.parent_ids(merged) == [right, left]
    graph_json = json.loads((tmp_path / "dag" / "graph.json").read_text(encoding="utf-8"))
    assert graph_json["schema_version"] == 2

    forward_hash = compute_step_hash(
        manager,
        "concat",
        ["right", "left"],
        {"sample_ids": ["right", "left"]},
    )
    reverse_hash = compute_step_hash(
        manager,
        "concat",
        ["left", "right"],
        {"sample_ids": ["right", "left"]},
    )
    assert forward_hash != reverse_hash


def test_concat_preserves_genes_and_does_not_mutate_samples():
    obs = pd.DataFrame(index=[f"cell_{idx}" for idx in range(4)])
    var = pd.DataFrame(index=["common", "rare"])
    sample_a = AnnData(X=np.array([[1, 1], [1, 0], [1, 0], [1, 0]]), obs=obs.copy(), var=var.copy())
    sample_b = AnnData(X=np.array([[1, 0], [1, 0], [1, 0], [1, 0]]), obs=obs.copy(), var=var.copy())
    manager = MagicMock()
    manager.get_object.side_effect = lambda node_id: {"a": sample_a, "b": sample_b}[node_id]

    combined, result_type, result_key = concat_rule(
        manager,
        ["a", "b"],
        sample_ids=["a", "b"],
    )

    assert combined.shape == (8, 2)
    assert "rare" in combined.var_names
    assert set(combined.obs["sample"].astype(str)) == {"a", "b"}
    assert result_type == "new_object"
    assert result_key == "concat"
    assert "sample" not in sample_a.obs
    assert "sample" not in sample_b.obs


def test_concat_validates_parent_mapping_and_non_empty_samples():
    sample = _make_test_adata()
    empty_sample = sample[:0].copy()
    manager = MagicMock()
    manager.get_object.side_effect = lambda node_id: {"full": sample, "empty": empty_sample}[node_id]

    with pytest.raises(ValueError, match="one ordered parent"):
        concat_rule(manager, ["full"], sample_ids=["a", "b"])
    with pytest.raises(ValueError, match="unique sample_ids"):
        concat_rule(manager, ["full", "full"], sample_ids=["a", "a"])
    with pytest.raises(ValueError, match="Sample 'b' has no cells"):
        concat_rule(manager, ["full", "empty"], sample_ids=["a", "b"])


def test_multi_sample_branches_merge_after_qc_and_reuse_unaffected_sample(tmp_path):
    paths = [tmp_path / "sample_a.h5ad", tmp_path / "sample_b.h5ad"]
    for path in paths:
        _make_test_adata().write_h5ad(path)

    def fake_scrublet(adata, **kwargs):
        adata.obs["doublet_score"] = np.linspace(0, 1, adata.n_obs)
        adata.obs["predicted_doublet"] = False
        adata.uns["scrublet"] = {"parameters": kwargs}

    params = {
        "data_paths": [str(path) for path in paths],
        "sample_ids": ["a", "b"],
        "qc_min_genes": 1,
        "qc_max_genes": 1000,
        "qc_mt_pct": None,
        "min_cells": 3,
        "sample_overrides": {"b": {"qc_max_genes": 900}},
    }
    with temp_context(str(tmp_path)) as ctx:
        runner = get_runner(
            ctx=ctx,
            config={"storage_dir": str(tmp_path / "dag")},
            force_new=True,
        )
        with patch("project.sc_pipeline.sc_rules.sc.pp.scrublet", side_effect=fake_scrublet):
            adata, summary = runner.execute(target_stage="normalize", **params)

        normalize_node = runner.manager.active_node_id
        concat_node = runner.manager.parent_ids(normalize_node)[0]
        qc_nodes = runner.manager.parent_ids(concat_node)
        first_qc_by_sample = {
            runner.manager.graph.nodes[runner.manager.raw_ancestors(node)[0]]["params"]["sample_id"]: node
            for node in qc_nodes
        }

        assert adata.n_obs == 60
        assert "Stage 'normalize' complete." in summary
        assert runner.manager.graph.nodes[concat_node]["action"] == "concat"
        assert [runner.manager.graph.nodes[item]["action"] for item in qc_nodes] == ["qc", "qc"]
        assert runner.manager.graph.nodes[qc_nodes[0]]["params"]["qc_max_genes"] == 1000
        assert runner.manager.graph.nodes[qc_nodes[1]]["params"]["qc_max_genes"] == 900
        layout = runner.io.pipeline_dag_layout(runner.manager, normalize_node)
        raw_nodes = runner.manager.raw_ancestors(normalize_node)
        assert layout[raw_nodes[0]][1] != layout[raw_nodes[1]][1]
        assert layout[concat_node][1] == 0

        changed = {**params, "sample_overrides": {"b": {"qc_max_genes": 800}}}
        with patch("project.sc_pipeline.sc_rules.sc.pp.scrublet", side_effect=fake_scrublet):
            runner.execute(target_stage="concat", **changed)
        changed_qc_nodes = runner.manager.parent_ids(runner.manager.active_node_id)
        changed_qc_by_sample = {
            runner.manager.graph.nodes[runner.manager.raw_ancestors(node)[0]]["params"]["sample_id"]: node
            for node in changed_qc_nodes
        }
        assert changed_qc_by_sample["a"] == first_qc_by_sample["a"]
        assert changed_qc_by_sample["b"] != first_qc_by_sample["b"]


def test_multi_sample_request_validation(tmp_path):
    with temp_context(str(tmp_path)) as ctx:
        runner = get_runner(ctx=ctx, config={"storage_dir": str(tmp_path / "dag")}, force_new=True)
        with pytest.raises(ValueError, match="unique"):
            runner.normalize_request(
                target_stage="concat",
                data_paths=["a", "b"],
                sample_ids=["same", "same"],
            )
        with pytest.raises(ValueError, match="unknown sample IDs"):
            runner.normalize_request(
                target_stage="concat",
                data_paths=["a", "b"],
                sample_ids=["a", "b"],
                sample_overrides={"c": {"qc_mt_pct": 10}},
            )
        with pytest.raises(ValueError, match="requires at least two"):
            runner.normalize_request(target_stage="concat", data_path="one.h5ad")


def test_multi_sample_pipeline_runs_from_merged_qc_through_cluster(tmp_path):
    paths = [tmp_path / "sample_a.h5ad", tmp_path / "sample_b.h5ad"]
    for path in paths:
        _make_test_adata().write_h5ad(path)

    def fake_scrublet(adata, **kwargs):
        adata.obs["doublet_score"] = 0.1
        adata.obs["predicted_doublet"] = False
        adata.uns["scrublet"] = {"parameters": kwargs}

    with temp_context(str(tmp_path)) as ctx:
        runner = get_runner(ctx=ctx, config={"storage_dir": str(tmp_path / "dag")}, force_new=True)
        with patch("project.sc_pipeline.sc_rules.sc.pp.scrublet", side_effect=fake_scrublet):
            adata, summary = runner.execute(
                target_stage="cluster",
                data_paths=[str(path) for path in paths],
                sample_ids=["a", "b"],
                qc_min_genes=1,
                qc_max_genes=1000,
                qc_mt_pct=None,
                min_cells=1,
                n_hvg=12,
                regress_out=False,
                n_comps=5,
                n_pcs=5,
                resolution=0.4,
                integration_method="none",
            )

        actions = [
            runner.manager.graph.nodes[item]["action"]
            for item in runner.manager.ancestry_to_node(runner.manager.active_node_id)
        ]
        assert actions.count("raw") == 2
        assert actions.count("scrublet") == 2
        assert actions.count("qc") == 2
        assert actions.count("concat") == 1
        assert "leiden_res0.4" in adata.obs
        assert "X_umap" in adata.obsm
        assert "Integration: none" in summary


def test_legacy_combined_multi_sample_cache_is_readable_but_not_reused(tmp_path):
    paths = [tmp_path / "sample_a.h5ad", tmp_path / "sample_b.h5ad"]
    for path in paths:
        _make_test_adata().write_h5ad(path)
    manager = SCStateManager(storage_dir=str(tmp_path / "dag"), registry=create_default_registry())
    legacy_params = {
        "data_paths": [str(path) for path in paths],
        "sample_ids": ["a", "b"],
        "sample_key": "sample",
        "multi_sample_join": "inner",
    }
    legacy_adata = _make_test_adata()
    legacy_hash = compute_step_hash(manager, "raw", "init", legacy_params)
    legacy_node = manager.register_new_object(
        legacy_adata,
        parent_id=None,
        action="raw",
        params=legacy_params,
        hash_val=legacy_hash,
    )

    with temp_context(str(tmp_path)) as ctx:
        runner = SingleCellPipelineRunner(ctx=ctx, manager=manager, registry=manager.registry)
        adata, _ = runner.execute(target_stage="raw", **legacy_params)

    active_node = manager.active_node_id
    assert adata.n_obs == 60
    assert legacy_node in manager.graph.nodes
    assert legacy_node not in manager.ancestry_to_node(active_node)
    assert len(manager.raw_ancestors(active_node)) == 2


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


def test_legacy_neighbors_without_integration_params_remain_reusable(tmp_path):
    data_path = tmp_path / "source.h5ad"
    data_path.write_text("source fingerprint", encoding="utf-8")
    manager = SCStateManager(storage_dir=str(tmp_path / "dag"), registry=create_default_registry())
    full_params = {
        "data_path": str(data_path),
        "sample_key": "sample",
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
        "batch_correction_method": "none",
        "combat_key": None,
        "max_scale_value": 10,
        "regress_out": True,
        "n_comps": 50,
        "n_neighbors": 10,
        "n_pcs": 40,
        "use_rep": "X_pca",
        "integration_method": "none",
        "integration_batch_key": None,
    }
    actions = ["raw", "scrublet", "qc", "normalize", "hvg", "batch_correct", "scale", "pca", "neighbors"]
    raw_hash = compute_step_hash(manager, "raw", "init", full_params)
    for index, action in enumerate(actions):
        if action == "raw":
            params = {"data_path": str(data_path)}
        else:
            params = {key: full_params.get(key) for key in manager.registry.get(action).param_keys}
            if action == "neighbors":
                params.pop("integration_method")
                params.pop("integration_batch_key")
        manager.graph.add_node(action, action=action, params=params, hash=raw_hash if action == "raw" else None)
        if index:
            manager.graph.add_edge(actions[index - 1], action)

    assert manager.find_node_strict("neighbors", **full_params) == "neighbors"


def test_neighbor_integration_branches_after_shared_pca(tmp_path):
    manager = SCStateManager(storage_dir=str(tmp_path / "dag"), registry=create_default_registry())
    parent = _make_test_adata()
    parent.obs["sample"] = ["a"] * 15 + ["b"] * 15
    parent.obsm["X_pca"] = np.ones((parent.n_obs, 5))
    pca_node = manager.register_new_object(
        parent,
        parent_id=None,
        action="pca",
        params={"n_comps": 5},
        hash_val="pca-hash",
        result_key="X_pca",
    )

    with temp_context(str(tmp_path)) as ctx:
        runner = SingleCellPipelineRunner(ctx=ctx, manager=manager, registry=manager.registry)
        with (
            patch("project.sc_pipeline.sc_rules.sc.pp.neighbors"),
            patch("project.sc_pipeline.sc_rules.sc.external.pp.bbknn"),
        ):
            ordinary_node = runner.run_rule(
                "neighbors",
                pca_node,
                n_neighbors=5,
                n_pcs=5,
                use_rep="X_pca",
                integration_method="none",
                integration_batch_key="sample",
            )
            bbknn_node = runner.run_rule(
                "neighbors",
                pca_node,
                n_neighbors=5,
                n_pcs=5,
                use_rep="X_pca",
                integration_method="bbknn",
                integration_batch_key="sample",
            )
            reused_bbknn = runner.run_rule(
                "neighbors",
                pca_node,
                n_neighbors=5,
                n_pcs=5,
                use_rep="X_pca",
                integration_method="bbknn",
                integration_batch_key="sample",
            )

    assert ordinary_node != bbknn_node
    assert reused_bbknn == bbknn_node
    assert list(manager.graph.predecessors(ordinary_node)) == [pca_node]
    assert list(manager.graph.predecessors(bbknn_node)) == [pca_node]

    none_batch_hash = compute_step_hash(
        manager,
        "batch_correct",
        "hvg-hash",
        {"batch_correction_method": "none", "combat_key": "sample", "integration_method": "none"},
    )
    bbknn_batch_hash = compute_step_hash(
        manager,
        "batch_correct",
        "hvg-hash",
        {"batch_correction_method": "none", "combat_key": "sample", "integration_method": "bbknn"},
    )
    combat_batch_hash = compute_step_hash(
        manager,
        "batch_correct",
        "hvg-hash",
        {"batch_correction_method": "combat", "combat_key": "sample"},
    )
    assert none_batch_hash == bbknn_batch_hash
    assert combat_batch_hash != none_batch_hash


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
    assert all("integration_method" not in item.kwargs for item in runner.execute.call_args_list)


def test_plugin_forwards_only_explicit_parameters():
    ctx = MagicMock()
    ctx.session_id = "test-session"
    plugin = SingleCellPipeline(name="sc_front", ctx=ctx, config={})
    runner = MagicMock()
    runner.execute.return_value = (None, "complete")

    with patch("project.plugins.sc_front.get_runner", return_value=runner):
        plugin(target_stage="annotation")

    runner.execute.assert_called_once_with(target_stage="annotation")


def test_annotation_defaults_to_qwen(tmp_path):
    with temp_context(str(tmp_path)) as ctx:
        runner = get_runner(ctx=ctx, config={"storage_dir": str(tmp_path / "dag")}, force_new=True)
        request = runner.normalize_request(target_stage="annotation", data_path="source.h5ad")

    assert request.params["annotation_model"] == "qwen3.5:122b"
