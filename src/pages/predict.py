"""Live inference page for compatible preprocessed NPZ volumes."""
import time
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from ..config import EXPECTED_SHAPE
from ..inference import load_model, predict_array, validate_volume


def render():
    st.subheader("Run inference on a preprocessed MRI")
    st.write("Upload one `.npz` file containing a `data` array. The volume must already match the V3 preprocessing pipeline and intensity convention.")
    st.code(f"Expected shape: {EXPECTED_SHAPE}\nNPZ key: data", language="text")
    st.warning("Raw `.nii` and `.nii.gz` files are not supported. This app does not perform skull stripping, registration, or raw-MRI preprocessing.")
    uploaded = st.file_uploader("Preprocessed volume (.npz)", type=["npz"], accept_multiple_files=False)
    if uploaded is None:
        return
    try:
        # Limit uploaded bytes before loading the compressed archive.
        if uploaded.size > 150 * 1024 * 1024:
            raise ValueError("The upload exceeds the 150 MB limit.")
        with np.load(uploaded, allow_pickle=False) as archive:
            if "data" not in archive.files:
                raise ValueError("The NPZ file must contain an array named 'data'.")
            volume = validate_volume(archive["data"])
        st.markdown("#### Volume preview")
        a, b, c = st.columns(3)
        with a:
            ix = st.slider("Axial slice", 0, volume.shape[2] - 1, volume.shape[2] // 2)
        with b:
            iy = st.slider("Coronal slice", 0, volume.shape[1] - 1, volume.shape[1] // 2)
        with c:
            iz = st.slider("Sagittal slice", 0, volume.shape[0] - 1, volume.shape[0] // 2)
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), dpi=120)
        axes[0].imshow(np.rot90(volume[:, :, ix]), cmap="gray"); axes[0].set_title("Axial")
        axes[1].imshow(np.rot90(volume[:, iy, :]), cmap="gray"); axes[1].set_title("Coronal")
        axes[2].imshow(np.rot90(volume[iz, :, :]), cmap="gray"); axes[2].set_title("Sagittal")
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        if st.button("Predict brain age", type="primary", use_container_width=True):
            try:
                start = time.perf_counter()
                result = predict_array(volume, model_bundle=load_model())
                elapsed = time.perf_counter() - start
                st.success("Inference completed.")
                st.metric("Estimated brain age", f"{result['predicted_age']:.1f} years")
                st.caption(f"Checkpoint: {result['checkpoint']} · prediction time: {elapsed:.2f} seconds (model cached after first load)")
                st.caption("Single-checkpoint research demonstration; not a clinical diagnosis and not the same as the cross-validation dashboard results.")
            except Exception as exc:
                st.error(f"Inference failed: {exc}")
    except Exception as exc:
        st.error(f"The uploaded file could not be validated: {exc}")
