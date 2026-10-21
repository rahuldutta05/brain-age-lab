"""Central configuration and project-relative paths."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CHECKPOINT_DIR = ROOT / "checkpoints"
RAW_PREDICTIONS = DATA_DIR / "raw_m1_predictions.csv"
BAG_PREDICTIONS = DATA_DIR / "bag_m1_predictions.csv"
DEMO_PREDICTIONS = DATA_DIR / "demo_predictions_100.csv"
METRICS_JSON = DATA_DIR / "v3_accuracy_m1.json"
CHECKPOINT_PATH = CHECKPOINT_DIR / "m1_f0_s0.pt"
CHECKPOINT_META_PATH = CHECKPOINT_DIR / "m1_f0_s0.json"
BINS_PATH = CHECKPOINT_DIR / "bins_v3.json"
EXPECTED_SHAPE = (160, 192, 160)
APP_TITLE = "Brain Age Lab"
