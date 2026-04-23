import scanpy as sc
import networkx as nx
import uuid
import hashlib
import json
import inspect
import matplotlib.pyplot as plt

# --- Core DAG Classes ---

class Rule:
    def __init__(self, name, requires, func):
        """
        Defines a transformation step in the pipeline.
        :param name: The identifier for this rule (e.g., 'qc', 'hvg').
        :param requires: List of parent action names this rule depends on (e.g., ['raw']).
        :param func: The function to execute. Signature: func(mgr, parent_id, **params) -> AnnData
        """
        self.name = name
        self.requires = requires
        self.func = func

class RuleRegistry:
    def __init__(self):
        self.rules = {}

    def register(self, rule):
        self.rules[rule.name] = rule

    def get(self, name):
        if name not in self.rules:
            raise ValueError(f"Rule '{name}' is not registered.")
        return self.rules[name]

class SCStateManager:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.states = {}          # Maps state_id -> AnnData object
        self.registry = RuleRegistry()

    def _new_id(self):
        """Generates a short unique ID for a state."""
        return str(uuid.uuid4())[:8]

    def add_state(self, adata, parent, action, params, hash_val=None):
        """
        Registers a new data state in the graph.
        """
        sid = self._new_id()
        self.states[sid] = adata
        
        # Store metadata in the graph node for visualization and debugging
        self.graph.add_node(
            sid,
            action=action,
            params=params,
            hash=hash_val
        )
        
        if parent is not None:
            self.graph.add_edge(parent, sid)
            
        return sid

    def list_states(self):
        """Returns a summary of all states currently in memory."""
        return [
            {
                "state_id": n,
                "action": self.graph.nodes[n]["action"],
                "params": self.graph.nodes[n]["params"],
                "parents": list(self.graph.predecessors(n))
            }
            for n in self.graph.nodes
        ]

# --- Hashing & Execution Logic (Smart Caching) ---

def make_state_hash(parent_id, rule_name, params, mgr):
    """
    Creates a unique fingerprint for a requested operation.
    It considers the parent state, the rule name, and *only* the relevant parameters.
    """
    rule = mgr.registry.get(rule_name)
    
    # Filter params: Only include arguments that are actually accepted by the rule function.
    # This prevents irrelevant args (like passing 'resolution' to a QC step) from breaking the cache.
    sig = inspect.signature(rule.func)
    relevant_params = {k: v for k, v in params.items() if k in sig.parameters}
    
    # Construct the unique signature
    data = {
        "parent_id": parent_id,
        "rule": rule_name,
        "params": relevant_params
    }
    
    # Return MD5 hash of the sorted JSON string
    return hashlib.md5(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()

def run_rule(mgr, rule_name, parent_id, **params):
    """
    Executes a rule. Checks for existing cached states first.
    If a state with the same hash exists, returns that ID instead of re-computing.
    """
    # 1. Compute Hash
    h = make_state_hash(parent_id, rule_name, params, mgr=mgr)

    # 2. Check Cache (Look for existing node with this hash)
    for sid, node in mgr.graph.nodes(data=True):
        if node.get("hash") == h:
            print(f"[SC_DAG] Cache Hit! Reusing state {sid} for rule '{rule_name}'")
            return sid

    # 3. Validation
    rule = mgr.registry.get(rule_name)
    parent_node = mgr.graph.nodes[parent_id]
    parent_action = parent_node["action"]
    
    if parent_action not in rule.requires:
        raise RuntimeError(
            f"Rule '{rule_name}' requires parent to be one of {rule.requires}, "
            f"but parent action is '{parent_action}'."
        )

    # 4. Execution
    print(f"[SC_DAG] Running '{rule_name}' on parent {parent_id}...")
    
    # Filter params again for the actual function call
    sig = inspect.signature(rule.func)
    rule_params = {k: v for k, v in params.items() if k in sig.parameters}

    # Call the actual computation function
    # Note: We pass 'mgr' and 'parent_id' so rules can access the parent data
    new_adata = rule.func(mgr, parent_id, **rule_params)

    # 5. Store Result
    sid = mgr.add_state(new_adata, parent_id, rule_name, params, hash_val=h)
    return sid

def ensure(mgr, target, start_state, **params):
    """
    Recursively ensures dependencies are met to reach the 'target' rule.
    """
    rule = mgr.registry.get(target)
    parent_action = mgr.graph.nodes[start_state]["action"]

    # Case A: We are already at the target
    if parent_action == target:
        return start_state

    # Case B: The current state satisfies the requirement (Direct execution)
    if parent_action in rule.requires:
        return run_rule(mgr, target, start_state, **params)

    # Case C: Dependencies not met. We must build the path recursively.
    # (Simplified: currently assumes linear path based on first requirement)
    for req in rule.requires:
        # Recursively ensure the requirement is met
        pid = ensure(mgr, req, start_state, **params)
        # Then run the target
        return run_rule(mgr, target, pid, **params)
    
    raise ValueError(f"Could not find a path from '{parent_action}' to '{target}'")

# --- Specific Rule Implementations ---

def register_raw(mgr, adata):
    """Helper to inject the initial raw data into the graph."""
    sid = mgr._new_id()
    mgr.states[sid] = adata
    mgr.graph.add_node(sid, action="raw", params=None, hash="init")
    return sid

def qc_filter_rule(mgr, parent_id, qc_min_genes=200, qc_mt_pct=5):
    """
    QC Rule: Filters cells based on gene counts and mitochondrial content.
    """
    # Retrieve parent data (copy it to avoid modifying history!)
    adata = mgr.states[parent_id].copy()
    
    # Calculate metrics
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

    # Filter
    sc.pp.filter_cells(adata, min_genes=qc_min_genes)
    adata = adata[adata.obs['pct_counts_mt'] < qc_mt_pct, :].copy()
    
    return adata

def hvg_rule(mgr, parent_id, n_hvg=2000):
    """
    HVG Rule: Normalizes, Log-transforms, and selects Highly Variable Genes.
    """
    adata = mgr.states[parent_id].copy()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg)
    
    # Subset to HVG
    adata = adata[:, adata.var.highly_variable].copy()
    
    return adata

def cluster_rule(mgr, parent_id, resolution=0.5):
    """
    Cluster Rule: PCA, Neighbors, UMAP, and Leiden Clustering.
    """
    adata = mgr.states[parent_id].copy()
    
    # Standard clustering workflow
    sc.tl.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=resolution, flavor="igraph", key_added='leiden')

    return adata

# --- Global Initialization ---

# Initialize the manager
mgr = SCStateManager()

# Register the standard pipeline rules
mgr.registry.register(Rule(name="qc", requires=["raw"], func=qc_filter_rule))
mgr.registry.register(Rule(name="hvg", requires=["qc"], func=hvg_rule))
mgr.registry.register(Rule(name="cluster", requires=["hvg"], func=cluster_rule))

def get_manager():
    """Accessor for plugins to get the singleton manager."""
    return mgr