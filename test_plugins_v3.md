import sys
import os
import shutil

# 1. SETUP PATHS
current_dir = os.getcwd()
sys.path.append(current_dir)

print(f"📂 Working Directory: {current_dir}")

# 2. MOCK TASKWEAVER OBJECTS
class MockContext:
    """Mimics the TaskWeaver Context object"""
    def log(self, level: str, source: str, message: str):
        # Adjusted signature to match TaskWeaver's self.ctx.log(level, source, message)
        print(f"   [LOG] [{level.upper()}] {message}")

    def add_artifact(self, name: str, file_name: str, type: str, val: str = None, desc: str = ""):
        print(f"   [ARTIFACT] {name} ({type}): {file_name}")

class MockConfig:
    """Mimics the Plugin Config object"""
    def get(self, key, default=None):
        return default
        
try:
    # Adjust this import if your folder structure is slightly different
    from project.plugins.sc_pipeline_v2 import SingleCellPipeline
    # We also import the manager to verify internal state
    from project.utils.sc_dag_v2 import get_manager 
    print("✅ SUCCESS: Imported 'project.plugins.sc_pipeline_v2'")
except ImportError as e:
    print(f"❌ FATAL ERROR: Could not import plugin.\n   Reason: {e}")
    print("\n   TIP: Run this script from the folder containing the 'project' directory.")
    sys.exit(1)

print("\n--- 1. Instantiating Plugin ---")
try:
    # We must create the mock objects first
    mock_ctx = MockContext()
    mock_config = MockConfig()
    
    # We pass them to the plugin constructor exactly as TaskWeaver does
    plugin = SingleCellPipeline(name="sc_pipeline_v2", ctx=mock_ctx, config=mock_config)
    
    print("✅ Plugin instantiated successfully.")
except Exception as e:
    print(f"❌ Failed to instantiate plugin: {e}")
    import traceback
    traceback.print_exc()

```python
test_data_path = os.path.join("/project/lji226_uksr/taskweaver/project/data/processed_scRNA.h5ad")

# --- Test Case 1: Initial Run (Load -> QC) ---
print("\n--- 2. Testing Execution: Load Data + QC ---")
try:
    # Run the plugin
    # Note: Your updated plugin returns (adata, description_string)
    result_tuple = plugin(
        target_stage="qc",
        data_path=test_data_path, 
        min_genes=10 
    )
    
    # Validate Return Structure
    if isinstance(result_tuple, tuple):
        adata, description = result_tuple
        print("\n🔹 PLUGIN OUTPUT (Description):")
        print(description)
        
        if "Pipeline Failed" in description:
            print("\n❌ TEST FAILED: Plugin reported an error.")
        else:
            print(f"\n✅ TEST PASSED: QC ran successfully. Result Shape: {adata.shape}")
    else:
         # Fallback if plugin returned error string directly
         print(f"\n❌ UNEXPECTED OUTPUT: {result_tuple}")

except Exception as e:
    print(f"\n❌ CRASH: {e}")
    import traceback
    traceback.print_exc()
```


```python
mgr = get_manager()
mgr.print_status()
```


```python
# --- Test Case 2: Caching (QC -> HVG) ---
print("\n--- 3. Testing Execution: HVG (Should Reuse QC State) ---")
try:
    # We verify that the previous step actually saved something to the Manager
    mgr = get_manager()
    print(f"   [DEBUG] Current Nodes in Graph: {len(mgr.graph.nodes)}")
    
    # Run HVG
    # No data_path provided; relies on SCStateManager caching/search
    result_tuple_hvg = plugin(
        target_stage="hvg",
        min_genes=10, # Must match previous run to find the parent!
        n_hvg=2000
    )
    
    if isinstance(result_tuple_hvg, tuple):
        adata_hvg, desc_hvg = result_tuple_hvg
        print("\n🔹 PLUGIN OUTPUT (HVG):")
        print(desc_hvg)
        
        if "Stage 'hvg' Complete" in desc_hvg:
             print("\n✅ TEST PASSED: HVG ran successfully using cached parent.")
        else:
             print("\n❌ TEST FAILED: HVG did not complete as expected.")
    else:
         print(f"\n❌ UNEXPECTED OUTPUT: {result_tuple_hvg}")

except Exception as e:
    print(f"\n❌ CRASH: {e}")
    import traceback
    traceback.print_exc()
```


```python
mgr = get_manager()
mgr.print_status()
```


```python
# --- Test Case 3: Test branching on data loss
print("\n--- 3. Testing QC with min_genes=200")
try:
    # Run the plugin
    # Note: Your updated plugin returns (adata, description_string)
    result_tuple = plugin(
        target_stage="qc",
        data_path=test_data_path, 
        min_genes=400 
    )
    
    # Validate Return Structure
    if isinstance(result_tuple, tuple):
        adata, description = result_tuple
        print("\n🔹 PLUGIN OUTPUT (Description):")
        print(description)
        
        if "Pipeline Failed" in description:
            print("\n❌ TEST FAILED: Plugin reported an error.")
        else:
            print(f"\n✅ TEST PASSED: QC ran successfully. Result Shape: {adata.shape}")
    else:
         # Fallback if plugin returned error string directly
         print(f"\n❌ UNEXPECTED OUTPUT: {result_tuple}")

except Exception as e:
    print(f"\n❌ CRASH: {e}")
    import traceback
    traceback.print_exc()
```


```python
mgr = get_manager()
mgr.print_status()
```


```python
# --- Test Case 2: Caching (QC -> HVG) ---
print("\n--- 3. Testing Execution: HVG (Should Reuse QC State) ---")
try:
    # We verify that the previous step actually saved something to the Manager
    mgr = get_manager()
    print(f"   [DEBUG] Current Nodes in Graph: {len(mgr.graph.nodes)}")
    
    # Run HVG
    # No data_path provided; relies on SCStateManager caching/search
    result_tuple_hvg = plugin(
        target_stage="hvg",
        min_genes=400, # Must match previous run to find the parent!
        n_hvg=3000
    )
    
    if isinstance(result_tuple_hvg, tuple):
        adata_hvg, desc_hvg = result_tuple_hvg
        print("\n🔹 PLUGIN OUTPUT (HVG):")
        print(desc_hvg)
        
        if "Stage 'hvg' Complete" in desc_hvg:
             print("\n✅ TEST PASSED: HVG ran successfully using cached parent.")
        else:
             print("\n❌ TEST FAILED: HVG did not complete as expected.")
    else:
         print(f"\n❌ UNEXPECTED OUTPUT: {result_tuple_hvg}")

except Exception as e:
    print(f"\n❌ CRASH: {e}")
    import traceback
    traceback.print_exc()
```


```python
mgr = get_manager()
mgr.print_status()
```


```python
search_criteria = {"n_hvg": 3000}
found_id = mgr.find_node("hvg", search_criteria)

print(f"Found     : {found_id}")
```


```python
search_criteria = {"qc_mt_pct": 5}
found_id = mgr.find_node("qc", search_criteria)
print(len(found_id))
print(f"Found     : {found_id}")
```
