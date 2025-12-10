"""Quick start script for GraphSAGE implementation."""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False


def main():
    """Main quick start function."""
    parser = argparse.ArgumentParser(description="GraphSAGE Quick Start")
    parser.add_argument("--action", choices=["install", "train", "demo", "test", "all"], 
                       default="all", help="Action to perform")
    parser.add_argument("--dataset", default="cora", help="Dataset to use")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    
    args = parser.parse_args()
    
    # Get project root
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    print("🚀 GraphSAGE Implementation Quick Start")
    print(f"Project root: {project_root}")
    
    if args.action in ["install", "all"]:
        print("\n📦 Installing dependencies...")
        if not run_command([sys.executable, "-m", "pip", "install", "-e", ".[dev]"], 
                           "Installing package with dev dependencies"):
            print("❌ Installation failed. Please check your Python environment.")
            return
    
    if args.action in ["test", "all"]:
        print("\n🧪 Running tests...")
        if not run_command([sys.executable, "-m", "pytest", "tests/", "-v"], 
                           "Running unit tests"):
            print("⚠️  Some tests failed, but continuing...")
    
    if args.action in ["train", "all"]:
        print("\n🏋️ Training model...")
        train_cmd = [
            sys.executable, "scripts/train.py",
            "--dataset", args.dataset,
            "--epochs", str(args.epochs)
        ]
        if not run_command(train_cmd, f"Training GraphSAGE on {args.dataset}"):
            print("⚠️  Training failed, but continuing...")
    
    if args.action in ["demo", "all"]:
        print("\n🎨 Starting interactive demo...")
        print("The demo will open in your browser at http://localhost:8501")
        print("Press Ctrl+C to stop the demo")
        
        try:
            demo_cmd = [sys.executable, "-m", "streamlit", "run", "demo/app.py"]
            subprocess.run(demo_cmd, check=True)
        except KeyboardInterrupt:
            print("\n👋 Demo stopped by user")
        except subprocess.CalledProcessError as e:
            print(f"❌ Demo failed: {e}")
    
    print("\n🎉 Quick start completed!")
    print("\nNext steps:")
    print("1. Explore the interactive demo: python scripts/run_demo.py")
    print("2. Train on different datasets: python scripts/train.py --dataset citeseer")
    print("3. Check the README.md for advanced usage")
    print("4. Run tests: pytest tests/ -v")


if __name__ == "__main__":
    main()
