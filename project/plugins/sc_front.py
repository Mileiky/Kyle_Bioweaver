import io
import json
import os

import matplotlib.pyplot as plt
import networkx as nx
import scanpy as sc

from taskweaver.plugin import Plugin, register_plugin
from project.utils.sc_dag_v4 import compute_step_hash, ensure, get_manager, register_raw

@register_plugin
class SingleCellPipeline(Plugin):
    def __call__(self):
        # code stuff here yay
        