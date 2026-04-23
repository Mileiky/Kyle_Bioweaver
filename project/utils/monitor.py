import functools
import scanpy as sc
from .state_graph import graph_manager

def report_changes(func):
    """
    Decorator to monitor Scanpy operations.
    It expects the first argument of the decorated function to be 'self' (Plugin instance),
    and one of the arguments to be an AnnData object.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 1. Extract Plugin Context (self)
        plugin_instance = args[0] # The 'self' of the plugin class
        
        # 2. Find AnnData object in args
        adata = None
        for arg in args:
            if isinstance(arg, sc.AnnData):
                adata = arg
                break
        
        # 3. Snapshot State BEFORE
        if adata:
            shape_before = adata.shape
            obs_before = set(adata.obs.columns)
        
        # 4. EXECUTE THE PLUGIN
        result = func(*args, **kwargs)
        
        # 5. Snapshot State AFTER & Calculate Diffs
        msg = [f"✅ **Execution Success:** `{func.__name__}`"]
        
        if adata:
            shape_after = adata.shape
            obs_after = set(adata.obs.columns)
            new_obs = list(obs_after - obs_before)
            
            # Message Construction
            if shape_before != shape_after:
                msg.append(f"📉 **Filtering:** Cells {shape_before[0]} -> {shape_after[0]}")
            if new_obs:
                msg.append(f"🆕 **New Metadata:** `{new_obs}`")
            
            # 6. Update Graph
            # Create a label for the graph node
            node_label = f"N_cells: {shape_after[0]}"
            if new_obs:
                node_label += f"\nAdded: {new_obs[0]}" # Just show first new col to save space
            
            # Check if user requested branching via parent_node kwarg
            parent = kwargs.get('parent_node', None)
            
            graph_manager.add_step(
                action_name=func.__name__, 
                description=node_label, 
                parent_node=parent
            )
            
            # 7. Generate & Attach Visualization
            img_path = graph_manager.visualize()
            
            # TaskWeaver Specific: Register the image artifact so LLM sees it
            plugin_instance.ctx.add_artifact(
                name="Workflow_State_Graph",
                file_path=img_path,
                type="image"
            )
            
            # Add text summary of graph to response
            msg.append(graph_manager.get_lineage_text())

        return "\n".join(msg)

    return wrapper