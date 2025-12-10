#!/usr/bin/env python3
"""Script to run the GraphSAGE demo."""

import subprocess
import sys
import os
from pathlib import Path


def main():
    """Run the Streamlit demo."""
    # Get the project root directory
    project_root = Path(__file__).parent
    
    # Change to project root
    os.chdir(project_root)
    
    # Check if demo app exists
    demo_path = project_root / "demo" / "app.py"
    if not demo_path.exists():
        print("Error: Demo app not found at demo/app.py")
        sys.exit(1)
    
    # Run Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(demo_path), "--server.port", "8501"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nDemo stopped by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
