import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # points to backend/
DEBUG_SAVE_ROOT = PROJECT_ROOT / "debug_results"
DEBUG_SAVE_ROOT.mkdir(parents=True, exist_ok=True)

def clean_up_debug():
    """Remove old debug results inside project folder."""
    if DEBUG_SAVE_ROOT.exists():
        shutil.rmtree(DEBUG_SAVE_ROOT, ignore_errors=True)
        DEBUG_SAVE_ROOT.mkdir(parents=True, exist_ok=True)
        print(f"🧹 Cleared old debug_results at {DEBUG_SAVE_ROOT}")