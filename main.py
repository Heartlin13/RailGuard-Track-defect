"""Top-level CLI launcher for RailGuard."""

from pathlib import Path
import subprocess
import sys

project_root = Path(__file__).resolve().parent / "rail-guard"
main_script = project_root / "main.py"

if __name__ == "__main__":
    sys.path.insert(0, str(project_root))
    # Import and run directly
    import runpy
    sys.argv[0] = str(main_script)
    runpy.run_path(str(main_script), run_name="__main__")
