import os
from pathlib import Path

PLOT_DIR = Path(__file__).parent.parent / "figures"
assert os.path.isdir(PLOT_DIR), f"Could not find {PLOT_DIR}"
