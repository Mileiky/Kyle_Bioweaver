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

from project.plugins.cell_type_ann import cell_type_ann
from taskweaver.plugin.context import temp_context


def test_cell_type_ann_handles_hvg_subset_with_raw():
    obs = pd.DataFrame({"leiden_res0.7": pd.Categorical(["0", "0", "1", "1"])})
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
    adata = adata[:, ["g1", "g3"]].copy()

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
        plugin = cell_type_ann("run_cell_type_annotation", ctx, {})
        with patch(
            "project.plugins.cell_type_ann.requests.post",
            return_value=FakeResponse(),
        ):
            result = plugin(adata, groupby="leiden_res0.7", n_markers=2)

    assert result.obs["cell_type"].tolist() == ["type_0", "type_0", "type_1", "type_1"]
    assert result.uns["cell_type_annotation"]["annotations"] == {
        "0": "type_0",
        "1": "type_1",
    }
    assert set(result.uns["cell_type_annotation"]["cluster_markers"]) == {"0", "1"}
