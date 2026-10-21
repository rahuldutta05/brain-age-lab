"""Recorded prediction exploration page."""
import pandas as pd
import streamlit as st


def render(demo):
    st.subheader("Explore recorded V3 predictions")
    st.caption("These records were computed previously during evaluation. Selecting a row does not run the neural network.")
    valid_ages = demo.age.dropna()
    age_min = int(valid_ages.min()) if not valid_ages.empty else 40
    age_max = int(valid_ages.max()) if not valid_ages.empty else 100
    age_range = st.slider("Chronological age range", min_value=age_min, max_value=age_max, value=(age_min, age_max))
    filtered = demo[demo.age.between(age_range[0], age_range[1])].copy()
    filtered["absolute_error"] = (filtered.predicted_age - filtered.age).abs()
    filtered["brain_age_gap_raw"] = filtered.predicted_age - filtered.age
    cols = [c for c in ["subject_id", "mri_label", "age", "predicted_age", "brain_age", "BAG", "absolute_error", "fold", "n_models"] if c in filtered.columns]
    st.dataframe(filtered[cols], use_container_width=True, hide_index=True)
    if filtered.empty:
        st.warning("No records fall within this age range.")
        return
    id_col = "mri_label" if "mri_label" in filtered.columns else filtered.columns[0]
    options = filtered.index.tolist()
    chosen = st.selectbox("Inspect a recorded scan", options, format_func=lambda idx: f"{filtered.loc[idx, id_col]} · age {filtered.loc[idx, 'age']:.1f} years")
    row = filtered.loc[chosen]
    a, b, c = st.columns(3)
    a.metric("Chronological age", f"{row.age:.1f} years")
    b.metric("Recorded prediction", f"{row.predicted_age:.1f} years")
    c.metric("Absolute error", f"{abs(row.predicted_age-row.age):.1f} years")
    details = []
    for field in ("subject_id", "fold", "n_models"):
        if field in row.index:
            details.append(f"{field.replace('_', ' ').title()}: {row[field]}")
    if details:
        st.caption(" · ".join(details))
