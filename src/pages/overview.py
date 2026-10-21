"""Overview page renderer."""
import pandas as pd
import streamlit as st
from ..charts import predicted_vs_actual, error_distribution, age_group_mae, residual_vs_age


def render(raw, metrics):
    st.subheader("V3 evaluation summary")
    st.caption("Reported metrics are loaded from the supplied V3 evaluation artifact. They are not recomputed from the 100-record demo subset.")
    columns = st.columns(4)
    values = [
        ("Bias-corrected MAE", f"{metrics['MAE']:.2f} years"),
        ("Raw MAE", f"{metrics['raw_MAE_uncorrected']:.2f} years"),
        ("Age-group accuracy", f"{100 * metrics['age_group_acc_3class']:.1f}%"),
        ("Pearson correlation", f"{metrics['pearson_r']:.3f}"),
    ]
    for col, (label, value) in zip(columns, values):
        col.metric(label, value)
    extra = st.columns(3)
    for col, (label, value) in zip(extra, [("RMSE", f"{metrics['RMSE']:.2f} years"), ("R²", f"{metrics['R2']:.3f}"), ("Evaluation scans", f"{int(metrics['n_CN_scans']):,}")]):
        col.metric(label, value)
    st.info("These are supplied V3 results: approximately 78.16% three-class age-group accuracy and 3.76-year bias-corrected MAE. Do not present them as V4 results unless the deployed checkpoint and evaluation artifacts are verified to be from V4.")
    left, right = st.columns(2)
    with left:
        st.pyplot(predicted_vs_actual(raw), use_container_width=True)
    with right:
        st.pyplot(error_distribution(raw), use_container_width=True)
    st.pyplot(age_group_mae(raw), use_container_width=True)
    with st.expander("Additional diagnostic: residuals across age"):
        st.pyplot(residual_vs_age(raw), use_container_width=True)
    bands = [(0, 60, "Under 60"), (60, 70, "60–69"), (70, 80, "70–79"), (80, 130, "80+")]
    rows = []
    for low, high, label in bands:
        part = raw[(raw.age >= low) & (raw.age < high)].dropna(subset=["age", "predicted_age"])
        if len(part):
            rows.append({"Age group": label, "Scans": len(part), "MAE (years)": float((part.predicted_age - part.age).abs().mean())})
    if rows:
        st.markdown("#### Age-group summary")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
