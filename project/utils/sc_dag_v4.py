import scanpy as sc
import networkx as nx
import uuid
import hashlib
import json
import inspect

# ==============================================================================
# 1. CORE DAG INFRASTRUCTURE (Generalized)
# ==============================================================================

class Rule:
    def __init__(self, name, requires, func):
        self.name = name
        self.requires = requires  # List of parent actions, e.g. ["qc"]
        self.func = func
        
        # AUTOMATIC METADATA EXTRACTION
        # We analyze the function signature once at registration.
        # This tells the system exactly which parameters define this step's "State".
        sig = inspect.signature(func)
        system_params = {'mgr', 'parent_id'}
        self.param_keys = [k for k in sig.parameters if k not in system_params]

class RuleRegistry:
    def __init__(self):
        self.rules = {}
    def register(self, rule):
        self.rules[rule.name] = rule
    def get(self, name):
        if name not in self.rules:
            raise ValueError(f"Rule '{name}' is not registered.")
        return self.rules[name]

def compute_step_hash(mgr, rule_name, parent_hash, all_params):
    """
    Centralized Hashing Logic.
    Used by BOTH execution (run_rule) and search (find_node_strict).
    """
    if rule_name == 'raw': 
        #return "init"
        path = all_params.get('data_path')
        if not path:
            # CRITICAL: Do not allow ambiguous data sources
            raise ValueError("❌ Lineage Error: You must provide 'data_path' when loading raw data to ensure reproducibility.")

        return hashlib.md5(f"raw_{path}".encode()).hexdigest()

    rule = mgr.registry.get(rule_name)
    
    # DYNAMIC FILTERING:
    # Only grab parameters that are explicitly defined in the Rule's function.
    # This prevents 'resolution' from changing the hash of an 'hvg' step.
    relevant_params = {k: v for k, v in all_params.items() if k in rule.param_keys}
    
    data = {
        "parent_hash": parent_hash, # The Chain of Custody
        "rule": rule_name,
        "params": relevant_params
    }
    
    # Deterministic JSON serialization
    return hashlib.md5(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()

# ==============================================================================
# 2. STATE MANAGER (The Graph Brain)
# ==============================================================================

class SCStateManager:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.objects = {}   # The Vault: obj_ref -> AnnData
        self.registry = RuleRegistry()
        self.hash_index = {} # O(1) Lookup: hash -> node_id

    def _new_id(self):
        return str(uuid.uuid4())[:8]

    def _register_node(self, node_id, hash_val, **attr):
        """Internal helper to add node and update the O(1) index."""
        self.graph.add_node(node_id, hash=hash_val, **attr)
        self.hash_index[hash_val] = node_id 

    # --- REGISTRATION METHODS ---

    def register_new_object(self, adata, parent_id, action, params, hash_val=None):
        """For destructive steps (QC) that create new physical objects."""
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
        """For additive steps (Clustering) that share the parent's object."""
        parent_obj_ref = self.graph.nodes[parent_id]['obj_ref']
        
        node_id = f"node_{self._new_id()}"
        self._register_node(
            node_id, hash_val,
            action=action, params=params, obj_ref=parent_obj_ref, is_virtual=True, result_key=result_key, shape=adata.shape
        )
        self.graph.add_edge(parent_id, node_id)
        return node_id

    def get_object(self, node_id):
        if node_id not in self.graph.nodes: raise ValueError(f"Node {node_id} not found.")
        obj_ref = self.graph.nodes[node_id].get('obj_ref')
        return self.objects[obj_ref]

    # --- GENERALIZED SMART SEARCH ---

    def find_node_smart(self, target_stage, **user_params):
        """
        The Master Search Function.
        1. Tries to find an EXACT lineage match (Strict).
        2. Falls back to scanning for ANY match (Fuzzy).
        """
        # Phase 1: Strict (Exact Lineage)
        strict_id = self.find_node_strict(target_stage, **user_params)
        if strict_id:
            return strict_id, "exact_match"

        # Phase 2: Fuzzy (Parameter Scan)
        # We need to know WHICH params define this stage to check them.
        try:
            target_rule = self.registry.get(target_stage)
            target_keys = target_rule.param_keys
        except ValueError:
            return None, "no_match" # Rule doesn't exist

        candidates = []
        
        for node_id, attr in self.graph.nodes(data=True):
            if attr.get("action") != target_stage:
                continue
            
            node_params = attr.get("params", {})
            match = True
            
            # Check ONLY the parameters relevant to this stage
            for key in target_keys:
                user_val = user_params.get(key)
                node_val = node_params.get(key)
                
                # Strict check: If user provided it, it MUST match.
                if user_val is not None:
                    # Convert to string to handle float precision issues roughly
                    if str(user_val) != str(node_val):
                        match = False
                        break
            
            if match:
                candidates.append(node_id)
        
        if not candidates: return None, "no_match"
        if len(candidates) == 1: return candidates[0], "fuzzy_match"
        return candidates, "ambiguous"

    def find_node_strict(self, target_stage, **full_params):
        """
        Simulates the entire hash chain from 'raw' to 'target_stage' to find an exact match.
        Dynamically discovers the lineage using rule.requires.
        """
        # if target_stage == "raw": return None 
        if target_stage == "raw":
            # Direct lookup for raw data
            h = compute_step_hash(self, "raw", "init", full_params)
            return self.hash_index.get(h)

        # 1. Build the Dependency Chain Dynamically
        # e.g., ["qc", "hvg", "cluster"]
        chain = []
        curr = target_stage
        
        # Safety breaker to prevent infinite loops in cyclic graphs
        steps = 0
        while True and steps < 20:
            chain.insert(0, curr)
            if curr == "raw": break # Stop once we added raw
            try:
                rule = self.registry.get(curr)
                if not rule.requires: break # Detached rule
                curr = rule.requires[0]     # Assumption: Linear dependency for strict search
            except ValueError:
                return None # Unknown rule in chain
            steps += 1
            
        # 2. Walk the Chain & Calculate Hashes
        current_hash = "init"
        
        for stage in chain:
            # This uses the centralized hashing logic, ensuring consistency
            predicted_hash = compute_step_hash(self, stage, current_hash, full_params)
            current_hash = predicted_hash
            
        # 3. O(1) Lookup
        return self.hash_index.get(current_hash)

# ==============================================================================
# 3. EXECUTION LOGIC
# ==============================================================================

def run_rule(mgr, rule_name, parent_id, **params):
    """
    Executes a rule. checks cache first.
    """
    # 1. Calculate Expected Hash
    parent_hash = mgr.graph.nodes[parent_id].get("hash", "init") if parent_id else "init"
    h = compute_step_hash(mgr, rule_name, parent_hash, params)
    
    # 2. Cache Hit?
    if h in mgr.hash_index:
        return mgr.hash_index[h]

    # 3. Cache Miss -> Execute
    rule = mgr.registry.get(rule_name)
    
    # Dynamic Param Filter (Safety)
    rule_params = {k: v for k, v in params.items() if k in rule.param_keys}
    
    result = rule.func(mgr, parent_id, **rule_params)
    
    # 4. Register Result
    # Result format: (adata, type_flag, optional_key)
    if result[1] == "new_object":
        return mgr.register_new_object(result[0], parent_id, rule_name, rule_params, hash_val=h)
    elif result[1] == "virtual":
        return mgr.register_virtual_node(result[0], parent_id, rule_name, rule_params, result_key=result[2], hash_val=h)

def ensure(mgr, target, start_state, **params):
    """
    Recursive dependency resolver.
    """
    rule = mgr.registry.get(target)
    
    # Determine current state
    current_action = mgr.graph.nodes[start_state]["action"] if start_state else "raw"
    
    # Base Case: Are we already at the target?
    if current_action == target:
        # We need to verify if the node's parameters match what we want.
        # Simplest way: Run run_rule on the *parent* of start_state.
        preds = list(mgr.graph.predecessors(start_state))
        if preds:
            return run_rule(mgr, target, preds[0], **params)
        return start_state 

    # Case: Is the current state a valid parent?
    if current_action in rule.requires:
        return run_rule(mgr, target, start_state, **params)

    # Case: Need to go deeper (Recursion)
    # Take the first requirement (Primary Parent)
    req = rule.requires[0]
    pid = ensure(mgr, req, start_state, **params)
    return run_rule(mgr, target, pid, **params)

# ==============================================================================
# 4. RULE IMPLEMENTATIONS (The Biology)
# ==============================================================================

# def register_raw(mgr, adata, path):
#     return mgr.register_new_object(adata, None, "raw", {"path": path}, hash_val="init")
def register_raw(mgr, adata, path):
    h = compute_step_hash(mgr, 'raw', None, {"data_path": path})
    
    # Check if exists to avoid duplicates
    if h in mgr.hash_index:
        return mgr.hash_index[h]
        
    return mgr.register_new_object(adata, None, "raw", {"data_path": path}, hash_val=h)

def qc_filter_rule(mgr, parent_id, qc_min_genes=200, qc_mt_pct=5):
    old_adata = mgr.get_object(parent_id)
    adata = old_adata.copy()
    
    # Basic QC Logic
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True)
    sc.pp.filter_cells(adata, min_genes=qc_min_genes)
    adata = adata[adata.obs['pct_counts_mt'] < qc_mt_pct, :].copy()
    
    return adata, "new_object"

def hvg_rule(mgr, parent_id, n_hvg=2000):
    adata = mgr.get_object(parent_id)
    # Additive logic: We modify the object IN PLACE (shared), but safely.
    # Note: Normalization is idempotent-ish in scanpy if run twice on same object.
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

# ==============================================================================
# 5. INITIALIZATION
# ==============================================================================

mgr = SCStateManager()

# Registration configures the Dynamic Search automatically
mgr.registry.register(Rule("qc", ["raw"], qc_filter_rule))
mgr.registry.register(Rule("hvg", ["qc"], hvg_rule))
mgr.registry.register(Rule("cluster", ["hvg"], cluster_rule))

def get_manager(): return mgr