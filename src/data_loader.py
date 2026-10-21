"""Cached loading and basic validation for packaged evaluation artifacts."""
import json
import pandas as pd
import streamlit as st
from .config import RAW_PREDICTIONS, BAG_PREDICTIONS, DEMO_PREDICTIONS, METRICS_JSON

REQUIRED_PREDICTION_COLUMNS = {"age", "predicted_age"}

@st.cache_data(show_spinner=False)
def load_tables():
    """Load packaged CSVs and metrics once; paths are resolved from this project."""
    raw = pd.read_csv(RAW_PREDICTIONS)
    bag = pd.read_csv(BAG_PREDICTIONS)
    demo = pd.read_csv(DEMO_PREDICTIONS)
    metrics = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
    for label, frame in (("raw predictions", raw), ("BAG predictions", bag), ("demo predictions", demo)):
        missing = REQUIRED_PREDICTION_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"The {label} file is missing required columns: {', '.join(sorted(missing))}")
    return raw, bag, metrics, demo
