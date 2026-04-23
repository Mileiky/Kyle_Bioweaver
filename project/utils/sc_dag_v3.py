import scanpy as sc
import networkx as nx
import uuid
import hashlib
import json
import inspect

# --- Core DAG Classes ---

class Rule:
    def __init__(self, name, requires, func):
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

# --- Centralized Hashing Logic (Single Source of Truth) ---
def compute_step_hash(mgr, rule_name, parent_hash, all_params):
    """
    Calculates a deterministic hash for a pipeline step.
    Used by BOTH 'run_rule' (execution) and 'find_node_strict' (search).
    """
    if rule_name == 'raw':
        return "init"

    # 1. Get Rule & Signature
    rule = mgr.registry.get(rule_name)
    sig = inspect.signature(rule.func)
    
    # 2. Filter Params: Keep ONLY what the function actually uses
    relevant_params = {k: v for k, v in all_params.items() if k in sig.parameters}
    
    # 3. Create Fingerprint
    data = {
        "parent_hash": parent_hash, # Chain of Custody
        "rule": rule_name,
        "params": relevant_params
    }
    
    # 4. Return MD5
    return hashlib.md5(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()


class SCStateManager:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.objects = {}  # THE VAULT: Maps obj_ref -> Physical AnnData Object
        self.registry = RuleRegistry()
        # Fast Lookup: Maps hash -> node_id for O(1) access
        self.hash_index = {} 

    def _new_id(self):
        return str(uuid.uuid4())[:8]

    # --- REGISTRATION ---

    def _register_node(self, node_id, hash_val, **attr):
        """Internal helper to add node and update index."""
        self.graph.add_node(node_id, hash=hash_val, **attr)
        self.hash_index[hash_val] = node_id  # Update O(1) Index

    def register_new_object(self, adata, parent_id, action, params, hash_val=None):
        obj_ref = f"obj_{self._new_id()}"
        self.objects[obj_ref] = adata
        
        node_id = f"node_{self._new_id()}"
        self._register_node(
            node_id, hash_val,
            action=action, params=params, obj_ref=obj_ref, is_virtual=False, shape=adata.shape
        )
        if parent_id: self.graph.add_edge(parent_id, node_id)
        return node_id

    def register_virtual_node(self, adata, parent_id, action, params, result_key, hash_val=None):
        if not parent_id: raise ValueError("Virtual nodes must have a parent.")
        parent_obj_ref = self.graph.nodes[parent_id]['obj_ref']

        node_id = f"node_{self._new_id()}"
        self._register_node(
            node_id, hash_val,
            action=action, params=params, obj_ref=parent_obj_ref, is_virtual=True, result_key=result_key, shape=adata.shape
        )
        self.graph.add_edge(parent_id, node_id)
        return node_id

    # --- STRICT SEARCH IMPLEMENTATION ---
    def find_node_strict(self, target_stage, **full_params):
        """
        Simulates the pipeline hash chain to find the EXACT node matching the full context.
        """
        # 1. Define the Canonical Lineage Order
        # This defines the dependency chain we must verify.
        STANDARD_LINEAGE = ["qc", "hvg", "cluster"]
        
        if target_stage not in STANDARD_LINEAGE:
            return None # Unknown stage

        # 2. Simulate Hash Chain from Root
        current_hash = "init" # Root hash for 'raw'
        
        # Determine steps to simulate (up to and including target)
        steps_to_check = []
        for stage in STANDARD_LINEAGE:
            steps_to_check.append(stage)
            if stage == target_stage:
                break
        
        # 3. Walk the chain
        print(f"[Strict Search] simulating chain for target '{target_stage}'...")
        for stage in steps_to_check:
            # Calculate what the hash SHOULD be for this step
            predicted_hash = compute_step_hash(self, stage, current_hash, full_params)
            current_hash = predicted_hash # Pass this hash as parent to next step
            
        # 4. O(1) Lookup
        found_node = self.hash_index.get(current_hash)
        
        if found_node:
            print(f"[Strict Search] ✅ Match Found! Hash={current_hash[:8]} -> Node={found_node}")
            return found_node
        else:
            print(f"[Strict Search] ❌ No match. Expected Hash={current_hash[:8]}")
            return None

    def find_node(self, action, params_subset):
        """Legacy Loose Search (kept for fallback if needed)."""
        for node_id, attr in self.graph.nodes(data=True):
            if attr.get("action") != action: continue
            node_params = attr.get("params", {})
            match = True
            for k, v in params_subset.items():
                if str(node_params.get(k)) != str(v):
                    match = False
                    break
            if match: return node_id
        return None
    def get_object(self, node_id):
        if node_id not in self.graph.nodes: raise ValueError(f"Node {node_id} not found.")
        obj_ref = self.graph.nodes[node_id].get('obj_ref')
        return self.objects[obj_ref]

# --- Execution Logic ---

def run_rule(mgr, rule_name, parent_id, **params):
    # 1. Get Parent Hash
    parent_hash = "init"
    if parent_id and parent_id in mgr.graph.nodes:
        parent_hash = mgr.graph.nodes[parent_id].get("hash", "init")

    # 2. Compute Expected Hash (Using Centralized Logic)
    h = compute_step_hash(mgr, rule_name, parent_hash, params)

    # 3. Check Cache (O(1) via Index)
    if h in mgr.hash_index:
        existing_node = mgr.hash_index[h]
        print(f"[SC_DAG] Cache Hit! Reusing {existing_node}")
        return existing_node

    # 4. Execute Rule (Cache Miss)
    rule = mgr.registry.get(rule_name)
    sig = inspect.signature(rule.func)
    rule_params = {k: v for k, v in params.items() if k in sig.parameters}

    print(f"[SC_DAG] Running '{rule_name}' on {parent_id}...")
    result = rule.func(mgr, parent_id, **rule_params)
    
    # 5. Register Result
    adata, node_type = result[0], result[1]
    
    if node_type == "new_object":
        return mgr.register_new_object(adata, parent_id, rule_name, params, hash_val=h)
    elif node_type == "virtual":
        return mgr.register_virtual_node(adata, parent_id, rule_name, params, result_key=result[2], hash_val=h)

def ensure(mgr, target, start_state, **params):
    rule = mgr.registry.get(target)
    
    # Base Case: We are at the target action?
    # Note: We check action match, but run_rule handles hash check to ensure correctness
    current_action = mgr.graph.nodes[start_state]["action"] if start_state else "raw"
    if current_action == target:
        # One last check: does the current node actually match the requested params?
        # Re-running run_rule is safe because it checks the hash.
        return run_rule(mgr, target, list(mgr.graph.predecessors(start_state))[0], **params)

    # Dependency Resolution
    if current_action in rule.requires:
        return run_rule(mgr, target, start_state, **params)

    for req in rule.requires:
        pid = ensure(mgr, req, start_state, **params)
        return run_rule(mgr, target, pid, **params)
    
    raise ValueError(f"Cannot path from {current_action} to {target}")

# --- Rule Implementations ---

def register_raw(mgr, adata, path):
    # Register with fixed hash "init"
    # Note: For real-world, hash the file path/content here
    return mgr.register_new_object(adata, None, "raw", {"path": path}, hash_val="init")

def qc_filter_rule(mgr, parent_id, qc_min_genes=200, qc_mt_pct=5):
    old_adata = mgr.get_object(parent_id)
    adata = old_adata.copy()
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True)
    sc.pp.filter_cells(adata, min_genes=qc_min_genes)
    adata = adata[adata.obs['pct_counts_mt'] < qc_mt_pct, :].copy()
    return adata, "new_object"

def hvg_rule(mgr, parent_id, n_hvg=2000):
    adata = mgr.get_object(parent_id)
    if 'log1p' not in adata.uns:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg)
    return adata, "virtual", "var:highly_variable"

def cluster_rule(mgr, parent_id, resolution=0.5):
    adata = mgr.get_object(parent_id)
    key_added = f"leiden_res{resolution}"
    if 'X_pca' not in adata.obsm:
        sc.tl.pca(adata)
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=resolution, flavor="igraph", key_added=key_added)
    return adata, "virtual", key_added

# --- Init ---
mgr = SCStateManager()
mgr.registry.register(Rule("qc", ["raw"], qc_filter_rule))
mgr.registry.register(Rule("hvg", ["qc"], hvg_rule))
mgr.registry.register(Rule("cluster", ["hvg"], cluster_rule))

def get_manager(): return mgr