"""Checkpoint loading and single-volume inference for the supplied V3 model."""
import json
import numpy as np
import streamlit as st
import torch
from .config import CHECKPOINT_PATH, CHECKPOINT_META_PATH, BINS_PATH, EXPECTED_SHAPE
from .model import SFCN


@st.cache_resource(show_spinner="Loading the V3 model…")
def load_model():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH.name}")
    torch.set_num_threads(min(2, torch.get_num_threads()))
    bins = json.loads(BINS_PATH.read_text(encoding="utf-8"))
    meta = json.loads(CHECKPOINT_META_PATH.read_text(encoding="utf-8"))
    model = SFCN(n_bins=int(bins["n_bins"]))
    state = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    centers = bins["bin_min"] + bins["bin_step"] / 2 + np.arange(bins["n_bins"]) * bins["bin_step"]
    return model, torch.device("cpu"), bins, meta, torch.tensor(centers, dtype=torch.float32)


def validate_volume(array):
    x = np.asarray(array, dtype=np.float32)
    x = np.squeeze(x)
    if x.shape != EXPECTED_SHAPE:
        raise ValueError(f"Expected volume shape {EXPECTED_SHAPE}; received {x.shape}.")
    if not np.isfinite(x).all():
        raise ValueError("Volume contains NaN or infinite values.")
    if not np.any(x):
        raise ValueError("Volume is empty or contains only zeros.")
    return x


def prepare_volume(array, cv):
    x = validate_volume(array)
    tensor = torch.from_numpy(x.copy()).unsqueeze(0).unsqueeze(0)
    mask = tensor != 0
    fraction = mask.float().flatten(1).mean(1).clamp_min(1e-3).view(-1, 1, 1, 1, 1)
    return torch.where(mask, ((1.0 + float(cv) * tensor) / fraction).clamp_min(0), torch.zeros_like(tensor))


def predict_array(array, model_bundle=None):
    model, device, bins, meta, centers = model_bundle or load_model()
    x = prepare_volume(array, float(meta.get("cv", 0.35))).to(device)
    with torch.inference_mode():
        log_probs = model(x)
        probabilities = log_probs.float().exp()
        age = float((probabilities * centers.to(device)).sum(1).item())
    return {"predicted_age": age, "checkpoint": meta.get("tag", "m1_f0_s0"), "input_shape": list(np.squeeze(array).shape), "input_mode": meta.get("input_mode", "meannorm")}
