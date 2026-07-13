"""Compatibility shim for the refactored single-cell pipeline modules."""

from project.sc_pipeline.sc_dag import DEFAULT_STORAGE_DIR, SCStateManager, compute_step_hash, file_fingerprint
from project.sc_pipeline.sc_rules import (
    Rule,
    RuleRegistry,
    annotation_rule,
    batch_correct_rule,
    cluster_rule,
    create_default_registry,
    extract_cluster_markers,
    hvg_rule,
    markers_rule,
    neighbors_rule,
    normalize_rule,
    parse_annotation_response,
    pca_rule,
    qc_filter_rule,
    scale_rule,
    scrublet_rule,
    umap_rule,
)
from project.sc_pipeline.sc_run import get_runner


def get_manager():
    """Return the shared state manager through the refactored runner factory."""
    return get_runner().manager


def run_rule(mgr, rule_name, parent_id, **params):
    """Delegate legacy rule execution calls to the refactored runner."""
    return get_runner().run_rule(rule_name, parent_id, **params)


def ensure(mgr, target, start_state, **params):
    """Delegate legacy ensure calls to the refactored runner."""
    return get_runner().ensure(target=target, start_state=start_state, **params)


def register_raw(mgr, adata, path, params=None):
    """Delegate legacy raw-node registration calls to the refactored runner."""
    return get_runner().register_raw(adata=adata, path=path, params=params)
