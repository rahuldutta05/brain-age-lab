"""Methodology, provenance, and limitations page."""
import streamlit as st


def render():
    st.subheader("Methodology and provenance")
    st.markdown("""
**Model architecture:** SFCN, a 3D convolutional neural network with a 64-bin age-distribution output.  
**Included checkpoint:** `m1_f0_s0.pt`, a single V3 fold/seed checkpoint.  
**Input dimensions:** `160 × 192 × 160`.  
**Input convention:** preprocessed brain-only volume using the V3 orientation, crop, registration, and intensity convention.  
**Age estimate:** expected value over the predicted age-bin probabilities.

The overview charts use the packaged, precomputed V3 cross-validation predictions. The upload tab runs the included checkpoint directly. These are distinct evaluation modes and must not be conflated.
""")
    with st.expander("Metric definitions"):
        st.markdown("""
- **MAE:** mean absolute difference between estimated and chronological age.
- **RMSE:** square root of mean squared prediction error.
- **Pearson correlation:** strength of the linear association between estimated and chronological age.
- **R²:** coefficient of determination under the supplied evaluation calculation.
- **Three-class age-group accuracy:** classification accuracy after assigning ages to the experiment's three categories; it is not ordinary regression accuracy.
""")
    with st.expander("Deployment checklist"):
        st.markdown("""
- Verify the displayed metrics correspond to the exact experiment being presented.
- Test the application from a fresh environment before the panel presentation.
- Confirm that live inference works with a known compatible preprocessed volume.
- Keep a local copy ready if cloud CPU or memory limits cause problems.
- Use de-identified scans only when permitted. Do not use these estimates for diagnosis or treatment decisions.
""")
    st.caption("Research demonstration only. Brain-age estimates are not clinical diagnoses.")
