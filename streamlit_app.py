import sys
import os
from pathlib import Path

# Add 'src' to PYTHONPATH
src_path = str(Path(__file__).parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import the main app function
from schrodinger_solver_web.app import main

if __name__ == "__main__":
    main()
