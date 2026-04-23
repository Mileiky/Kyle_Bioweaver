import scanpy as sc
import networkx as nx
import uuid
import hashlib
import json
import inspect

# --- Core DAG Classes ---

class Rule:
    def __init__(self, name, requires, func):
        """
        Defines a transformation step in the pipeline.
        :param func: Must return either:
                     1. (AnnData, "new_object") -> For destructive steps
                     2. (AnnData, "virtual", result_key) -> For additive steps
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
        self.objects = {}  # THE VAULT: Maps obj_ref -> Physical AnnData Object
        self.registry = RuleRegistry()

    def _new_id(self):
        return str(uuid.uuid4())[:8]

    # --- HYBRID REGISTRATION METHODS ---

    def register_new_object(self, adata, parent_id, action, params, hash_val=None):
        """For destructive steps (e.g., QC, Filtering) that create a NEW matrix."""
        # 1. Physical Layer: Save the new object
        obj_ref = f"obj_{self._new_id()}"
        self.objects[obj_ref] = adata

        # 2. Logical Layer: Create the node
        node_id = f"node_{self._new_id()}"
        self.graph.add_node(
            node_id,
            action=action,
            params=params,
            hash=hash_val,
            obj_ref=obj_ref,      # Points to the new object
            is_virtual=False,
            shape=adata.shape
        )
        
        if parent_id:
            self.graph.add_edge(parent_id, node_id)
        return node_id

    def register_virtual_node(self, adata, parent_id, action, params, result_key, hash_val=None):
        """For additive steps (e.g., Clustering) that add a key to an EXISTING object."""
        # 1. Resolve Parent's Object Reference
        # We assume the 'adata' passed in is the SAME physical object as the parent's
        if parent_id:
            parent_obj_ref = self.graph.nodes[parent_id]['obj_ref']
        else:
            raise ValueError("Virtual nodes must have a parent.")

        # 2. Logical Layer: Create the node pointing to the OLD object
        node_id = f"node_{self._new_id()}"
        self.graph.add_node(
            node_id,
            action=action,
            params=params,
            hash=hash_val,
            obj_ref=parent_obj_ref, # Shared Reference!
            is_virtual=True,
            result_key=result_key,   # Where the data lives (e.g., 'leiden_res0.5')
            shape=adata.shape
        )
        
        self.graph.add_edge(parent_id, node_id)
        return node_id

    # --- RETRIEVAL & SEARCH ---

    def get_object(self, node_id):
        """Retrieves the physical object associated with a node."""
        if node_id not in self.graph.nodes:
            raise ValueError(f"Node {node_id} not found.")
        
        obj_ref = self.graph.nodes[node_id].get('obj_ref')
        if obj_ref not in self.objects:
            raise ValueError(f"Physical object {obj_ref} missing from memory!")
            
        return self.objects[obj_ref]

    def find_node(self, action, params_subset):
        """
        Smart Search: Locates a node matching the action and specific parameters.
        Returns the node_id or None.
        """
        for node_id, attr in self.graph.nodes(data=True):
            if attr.get("action") != action:
                continue
            
            # Check if requested params are a subset of node params
            node_params = attr.get("params", {})
            match = True
            for k, v in params_subset.items():
                # We cast to string to handle float precision issues roughly
                if str(node_params.get(k)) != str(v):
                    match = False
                    break
            
            if match:
                return node_id
        return None

    # def find_node(self, action, params_subset):
    #     """
    #     Locates ALL nodes where the action matches and parameters match the subset.
    #     Returns a list of node_ids (e.g., ['node_a1', 'node_b2']).
    #     """
    #     matches = []
        
    #     for node_id, attr in self.graph.nodes(data=True):
    #         if attr.get("action") != action:
    #             continue
            
    #         node_params = attr.get("params", {})
    #         match = True
    #         for k, v in params_subset.items():
    #             # Cast to string for loose matching
    #             if str(node_params.get(k)) != str(v):
    #                 match = False
    #                 break
            
    #         if match:
    #             matches.append(node_id)
                
    #     return matches

    def print_status(self):
        """Prints a tabular summary of the current graph state."""
        print(f"\n{'='*60}")
        print(f"📊 PIPELINE STATE SUMMARY (Active Node: {self.active_node if hasattr(self, 'active_node') else 'None'})")
        print(f"{'='*60}")
        print(f"{'Node ID':<10} | {'Action':<10} | {'Type':<10} | {'Parent':<10} | {'Details'}")
        print(f"{'-'*60}")

        # Iterate via topological sort to show flow (Lineage order)
        try:
            ordered_nodes = list(nx.topological_sort(self.graph))
        except nx.NetworkXUnfeasible:
            ordered_nodes = self.graph.nodes()

        for n in ordered_nodes:
            attr = self.graph.nodes[n]
            
            # Determine attributes
            action = attr.get('action', 'N/A')
            is_virtual = attr.get('is_virtual', False)
            node_type = "View/Key" if is_virtual else "New Obj"
            
            # Get Parent (assuming single parent for now)
            parents = list(self.graph.predecessors(n))
            parent = parents[0][:6] if parents else "ROOT"
            
            # Format Details
            if action == 'cluster':
                details = f"res={attr['params'].get('resolution')} -> {attr.get('result_key')}"
            # elif action == 'qc':
            #     details = f"min={attr['params'].get('qc_min_genes')}"
            elif action == 'raw':
                details = "Initial Data"
            else:
                details = str(attr.get('params', {}))
            
            # Print row
            print(f"{n[:8]:<10} | {action:<10} | {node_type:<10} | {parent:<10} | {details}")
        
        print(f"{'='*60}")
        print(f"🧠 Physical Memory Vault: {len(self.objects)} objects stored.")
        print(f"{'='*60}\n")
# --- Hashing & Execution Logic ---

def make_state_hash(parent_id, rule_name, params, mgr):
    """Creates a fingerprint for the operation."""
    # Get parent's hash to ensure lineage dependency
    parent_hash = "root"
    if parent_id and parent_id in mgr.graph.nodes:
        parent_hash = mgr.graph.nodes[parent_id].get("hash", "root")

    rule = mgr.registry.get(rule_name)
    sig = inspect.signature(rule.func)
    relevant_params = {k: v for k, v in params.items() if k in sig.parameters}
    
    data = {
        "parent_hash": parent_hash, # Chain of custody
        "rule": rule_name,
        "params": relevant_params
    }
    return hashlib.md5(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()

def run_rule(mgr, rule_name, parent_id, **params):
    """Executes a rule using the Hybrid Strategy."""
    
    # 1. Check Cache
    h = make_state_hash(parent_id, rule_name, params, mgr)
    for sid, node in mgr.graph.nodes(data=True):
        if node.get("hash") == h:
            print(f"[SC_DAG] Cache Hit! Reusing {sid}")
            return sid

    # 2. Prepare Execution
    rule = mgr.registry.get(rule_name)
    sig = inspect.signature(rule.func)
    rule_params = {k: v for k, v in params.items() if k in sig.parameters}

    print(f"[SC_DAG] Running '{rule_name}' on {parent_id}...")
    
    # 3. Execute Rule
    # Returns tuple: (adata, type_flag, optional_key)
    result = rule.func(mgr, parent_id, **rule_params)
    
    # 4. Hybrid Registration
    adata = result[0]
    node_type = result[1]
    
    if node_type == "new_object":
        return mgr.register_new_object(adata, parent_id, rule_name, params, hash_val=h)
    
    elif node_type == "virtual":
        result_key = result[2]
        return mgr.register_virtual_node(adata, parent_id, rule_name, params, result_key, hash_val=h)
    
    else:
        raise ValueError(f"Unknown node type returned by rule: {node_type}")

def ensure(mgr, target, start_state, **params):
    """Recursively ensures dependencies."""
    rule = mgr.registry.get(target)
    parent_action = mgr.graph.nodes[start_state]["action"]

    if parent_action == target:
        return start_state

    if parent_action in rule.requires:
        return run_rule(mgr, target, start_state, **params)

    # Simple linear search for dependency
    for req in rule.requires:
        pid = ensure(mgr, req, start_state, **params)
        return run_rule(mgr, target, pid, **params)
    
    raise ValueError(f"Cannot path from {parent_action} to {target}")

# --- Specific Rule Implementations (Refactored) ---

def register_raw(mgr, adata, path):
    """Entry point for raw data."""
    return mgr.register_new_object(adata, None, "raw", {"path": path}, hash_val="init")

def qc_filter_rule(mgr, parent_id, qc_min_genes=200, qc_mt_pct=5):
    """DESTRUCTIVE: Creates new object."""
    old_adata = mgr.get_object(parent_id)
    adata = old_adata.copy() # Deep copy
    
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True)
    sc.pp.filter_cells(adata, min_genes=qc_min_genes)
    adata = adata[adata.obs['pct_counts_mt'] < qc_mt_pct, :].copy()
    
    return adata, "new_object"

# def hvg_rule(mgr, parent_id, n_hvg=2000):
#     """DESTRUCTIVE: Subsetting genes creates new object."""
#     old_adata = mgr.get_object(parent_id)
#     adata = old_adata.copy()
    
#     sc.pp.normalize_total(adata, target_sum=1e4)
#     sc.pp.log1p(adata)
#     sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg)
#     adata = adata[:, adata.var.highly_variable].copy()
    
#     return adata, "new_object"

def hvg_rule(mgr, parent_id, n_hvg=2000):
    """
    ADDITIVE: Calculates stats and flags genes, but DOES NOT remove data.
    """
    # 1. Get shared object (No Copy!)
    adata = mgr.get_object(parent_id)
    
    # 2. Run Normalization & Log1p IN-PLACE
    # (Note: These change X values, so we are modifying the shared QC object.
    #  If you need the raw counts later, you might need a layer, but standard 
    #  scanpy pipelines often overwrite raw X after QC).
    if 'log1p' not in adata.uns:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    # 3. Calculate HVG (Adds 'highly_variable' to adata.var)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg)
    
    # 4. Return as Virtual Node
    # We use a special key to indicate this node's primary output is in .var
    return adata, "virtual", "var:highly_variable"

def cluster_rule(mgr, parent_id, resolution=0.5):
    """ADDITIVE: In-place edit with unique key."""
    # 1. Get shared object (No Copy!)
    adata = mgr.get_object(parent_id)
    
    # 2. Generate Unique Key
    # We use a clean string representation of resolution to avoid '.' issues in some tools if needed
    key_added = f"leiden_res{resolution}"
    
    # 3. Compute (if not already present)
    if 'X_pca' not in adata.obsm:
        sc.tl.pca(adata)
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)
    
    # 4. Cluster into the specific key
    sc.tl.leiden(adata, resolution=resolution, flavor="igraph",key_added=key_added)
    
    # 5. Return with 'virtual' flag
    return adata, "virtual", key_added

# --- Initialization ---
mgr = SCStateManager()
mgr.registry.register(Rule("qc", ["raw"], qc_filter_rule))
mgr.registry.register(Rule("hvg", ["qc"], hvg_rule))
mgr.registry.register(Rule("cluster", ["hvg"], cluster_rule))

def get_manager(): return mgr