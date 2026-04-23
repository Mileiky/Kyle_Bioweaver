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
    def log(self, message: str):
        print(f"   [LOG] {message}")

    def add_artifact(self, name: str, file_path: str, type: str):
        print(f"   [ARTIFACT] {name} ({type}): {file_path}")

class MockConfig:
    """Mimics the Plugin Config object (can be empty for this test)"""
    def get(self, key, default=None):
        return default

# 3. IMPORT THE PLUGIN
try:
    # Adjust this import if your folder structure is slightly different
    from project.plugins.sc_pipeline import SingleCellPipeline
    print("✅ SUCCESS: Imported 'project.plugins.sc_pipeline'")
except ImportError as e:
    print(f"❌ FATAL ERROR: Could not import plugin.\n   Reason: {e}")
    print("\n   TIP: Run this script from the folder containing the 'project' directory.")
    sys.exit(1)

# 4. TEST RUNNER
def run_tests():
    print("\n--- 1. Instantiating Plugin ---")
    try:
        # --- FIX IS HERE ---
        # We must create the mock objects first
        mock_ctx = MockContext()
        mock_config = MockConfig()
        
        # We pass them to the plugin constructor exactly as TaskWeaver does
        plugin = SingleCellPipeline(name="sc_pipeline", ctx=mock_ctx, config=mock_config)
        
        print("✅ Plugin instantiated successfully.")
    except Exception as e:
        print(f"❌ Failed to instantiate plugin: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- Prepare Dummy Data ---
    test_data_path = os.path.join("/project/lji226_uksr/taskweaver/project/data/processed_scRNA.h5ad")
    if not os.path.exists(test_data_path):
        print(f"\n⚠️ Test data not found at: {test_data_path}")
        print("   Creating synthetic test data now...")
        try:
            import scanpy as sc
            import numpy as np
            import anndata
            
            os.makedirs(os.path.dirname(test_data_path), exist_ok=True)
            
            # Generate tiny dataset (50 cells x 100 genes)
            dummy = anndata.AnnData(np.random.rand(50, 100))
            dummy.obs_names = [f"cell_{i}" for i in range(50)]
            dummy.var_names = [f"gene_{i}" for i in range(100)]
            # Add some dummy mitochondrial genes for QC testing
            dummy.var_names.values[0:5] = [f"MT-{i}" for i in range(5)]
            
            dummy.write(test_data_path)
            print("   ✅ Dummy data created.")
        except Exception as e:
            print(f"   ❌ Failed to create dummy data: {e}")
            return

    # --- Test Case 1: Initial Run (Load -> QC) ---
    print("\n--- 2. Testing Execution: Load Data + QC ---")
    try:
        result = plugin(
            target_stage="qc",
            data_path=test_data_path, 
            min_genes=10 
        )
        print("\n🔹 PLUGIN OUTPUT:")
        print(result)
        
        if "Error" in result or "Failed" in result:
             print("\n❌ TEST FAILED: Plugin reported an error.")
        else:
             print("\n✅ TEST PASSED: QC ran successfully.")

    except Exception as e:
        print(f"\n❌ CRASH: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- Test Case 2: Caching (QC -> HVG) ---
    print("\n--- 3. Testing Execution: HVG (Should Reuse QC State) ---")
    try:
        # No data_path provided; relies on SCStateManager caching
        result = plugin(
            target_stage="hvg",
            min_genes=10,
            n_hvg=2000
        )
        print("\n🔹 PLUGIN OUTPUT:")
        print(result)
        
        if "State ID" in result:
             print("\n✅ TEST PASSED: HVG ran successfully.")
        else:
             print("\n❌ TEST FAILED: Plugin did not return a valid state.")

    except Exception as e:
        print(f"\n❌ CRASH: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    run_tests()