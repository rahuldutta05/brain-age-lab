"""Inference utilities for the supplied V3 m1_f0_s0 SFCN checkpoint.

Input must be a preprocessed NPZ containing a `data` array with shape
(160, 192, 160), representing the brain-only z-scored volume used by V3.
This module does not implement raw MRI preprocessing.
"""
import json
from pathlib import Path
import numpy as np
import torch

from model import SFCN

ROOT = Path(__file__).resolve().parent
CKPT_PATH = ROOT / "checkpoints" / "m1_f0_s0.pt"
BINS_PATH = ROOT / "checkpoints" / "bins_v3.json"
META_PATH = ROOT / "checkpoints" / "m1_f0_s0.json"
EXPECTED_SHAPE = (160, 192, 160)

def load_model():
    torch.set_num_threads(min(2, torch.get_num_threads()))
    device = torch.device("cpu")
    bins = json.loads(BINS_PATH.read_text())
    meta = json.loads(META_PATH.read_text())
    model = SFCN(n_bins=int(bins["n_bins"]))
    state = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    centers = bins["bin_min"] + bins["bin_step"] / 2 + np.arange(bins["n_bins"]) * bins["bin_step"]
    return model, device, bins, meta, torch.tensor(centers, dtype=torch.float32)

def prepare_volume(array, bins):
    x = np.asarray(array, dtype=np.float32)
    # Accept only a single preprocessed volume; do not silently resize or reorient.
    x = np.squeeze(x)
    if x.shape != EXPECTED_SHAPE:
        raise ValueError(f"Expected preprocessed volume shape {EXPECTED_SHAPE}; received {x.shape}.")
    if not np.isfinite(x).all():
        raise ValueError("The volume contains NaN or infinite values.")
    # V3 m1 input convention: approximate UKB mean-normalized intensity from brain-only z-score.
    z = torch.from_numpy(x.copy()).unsqueeze(0).unsqueeze(0)
    mask = z != 0
    fraction = mask.float().flatten(1).mean(1).clamp_min(1e-3).view(-1, 1, 1, 1, 1)
    cv = float(bins.get("cv", 0.35))
    # V3's cv is stored in checkpoint metadata; caller can override from meta.
    return torch.where(mask, ((1.0 + cv * z) / fraction).clamp_min(0), torch.zeros_like(z))

def predict_array(array, model_bundle=None):
    model, device, bins, meta, centers = model_bundle or load_model()
    # The run metadata stores CV; the default V3 configuration used 0.35.
    x = prepare_volume(array, {"cv": float(meta.get("cv", 0.35))}).to(device)
    with torch.inference_mode():
        log_probs = model(x)
        probabilities = log_probs.float().exp()
        age = float((probabilities * centers.to(device)).sum(1).item())
    return {
        "predicted_age": age,
        "checkpoint": meta.get("tag", "m1_f0_s0"),
        "input_shape": list(np.squeeze(array).shape),
        "input_mode": meta.get("input_mode", "meannorm"),
    }
