import subprocess
import sys
import os

def run_script(script_name):
    print(f"\n{'='*60}")
    print(f"🚀 STARTING: {script_name}")
    print(f"{'='*60}")
    
    try:
        # sys.executable ensures it uses the same virtual environment/Python version
        # check=True ensures that if a script crashes, the whole pipeline stops
        subprocess.run([sys.executable, script_name], check=True)
        print(f"✅ FINISHED: {script_name} successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR: {script_name} failed with exit code {e.returncode}.")
        sys.exit(1)

def main():
    # Change to the directory containing the scripts
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # List the scripts in the exact order they need to run
    pipeline = [
        "w3_preprocess.py",
        "w5_train_final.py",
        "w5_threshold.py",
        "w6_onnx_export.py"
    ]
    
    for script in pipeline:
        run_script(script)
        
    print("🎉 All pipeline stages completed successfully!")

if __name__ == "__main__":
    main()
