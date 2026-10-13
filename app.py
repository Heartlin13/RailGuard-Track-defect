"""Top-level Streamlit launcher for RailGuard."""

from pathlib import Path
import runpy
import sys

project_root = Path(__file__).resolve().parent / "rail-guard"
app_script = project_root / "app.py"

sys.path.insert(0, str(project_root))
sys.argv[0] = str(app_script)
runpy.run_path(str(app_script), run_name="__main__")
