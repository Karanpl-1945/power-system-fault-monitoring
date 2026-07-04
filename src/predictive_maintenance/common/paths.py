from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LABELS_PATH = PROJECT_ROOT / "hv_double_line_90kv_labels.csv"
DEFAULT_WAVEFORM_DIR = PROJECT_ROOT / "hv_double_line_90kv_preprocessed_data"

