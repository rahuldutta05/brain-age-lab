"""Brain Age Lab — four-layer Streamlit demo.

Layer 1 — MRI viewer and scan selection (OASIS preprocessed NPZ volumes)
Layer 2 — Real-time brain-age prediction with latency reporting
Layer 3 — Brain Age Gap (BAG) and per-scan error analysis
Layer 4 — Research performance explorer (test-set charts and metrics)
"""

from pathlib import Path
import json
import time
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import streamlit as st

from inference import load_model, predict_array, EXPECTED_SHAPE

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
PREP = ROOT.parent / "brain age prep and meth" / "T1 preprocessed v2"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Brain Age Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: radial-gradient(ellipse at 20% 10%, #0d1e3a 0%, #080f1c 60%, #050a14 100%);
    color: #e2e8f0;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background: #07101f; }

/* Hero banner */
.hero {
    padding: 2rem 2.4rem;
    border: 1px solid rgba(125,211,252,0.18);
    border-radius: 20px;
    background: linear-gradient(130deg, rgba(15,30,60,0.95) 0%, rgba(10,22,45,0.95) 100%);
    backdrop-filter: blur(12px);
    margin-bottom: 1.6rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -80px;
    width: 260px; height: 260px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(99,179,237,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.eyebrow {
    font-size: .72rem;
    letter-spacing: .18em;
    color: #7dd3fc;
    font-weight: 700;
    text-transform: uppercase;
    margin-bottom: .4rem;
}
.hero h1 {
    margin: 0 0 .5rem 0;
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #f0f9ff 30%, #7dd3fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero p { color: #94a3b8; margin-bottom: 0; font-size: .97rem; }

/* Metric cards */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #0f1e38 0%, #0a1628 100%);
    padding: 1.1rem 1.3rem;
    border: 1px solid rgba(125,211,252,0.15);
    border-radius: 16px;
    transition: border-color .2s, transform .15s;
}
div[data-testid="stMetric"]:hover {
    border-color: rgba(125,211,252,0.4);
    transform: translateY(-2px);
}
div[data-testid="stMetricValue"] {
    font-size: 1.7rem !important;
    font-weight: 700 !important;
}

/* Section headers */
.section-header {
    font-size: 1.05rem;
    font-weight: 600;
    color: #bfdbfe;
    letter-spacing: .04em;
    margin-bottom: .6rem;
    display: flex;
    align-items: center;
    gap: .5rem;
}
.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(125,211,252,0.25) 0%, transparent 100%);
}

/* Layer badges */
.layer-badge {
    display: inline-block;
    padding: .25rem .7rem;
    border-radius: 999px;
    font-size: .7rem;
    font-weight: 700;
    letter-spacing: .1em;
    text-transform: uppercase;
    margin-bottom: .8rem;
}
.l1 { background: rgba(52,211,153,0.15); color: #34d399; border: 1px solid rgba(52,211,153,0.3); }
.l2 { background: rgba(99,179,237,0.15); color: #63b3ed; border: 1px solid rgba(99,179,237,0.3); }
.l3 { background: rgba(251,191,36,0.15); color: #fbbf24; border: 1px solid rgba(251,191,36,0.3); }
.l4 { background: rgba(167,139,250,0.15); color: #a78bfa; border: 1px solid rgba(167,139,250,0.3); }

/* Info box */
.info-box {
    background: rgba(15,30,60,0.6);
    border: 1px solid rgba(125,211,252,0.12);
    border-radius: 12px;
    padding: .8rem 1rem;
    font-size: .84rem;
    color: #94a3b8;
    margin-top: .6rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Matplotlib defaults
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor":  "#0a1628",
    "axes.facecolor":    "#0a1628",
    "axes.edgecolor":    "#1e3a5f",
    "axes.labelcolor":   "#94a3b8",
    "xtick.color":       "#64748b",
    "ytick.color":       "#64748b",
    "text.color":        "#e2e8f0",
    "grid.color":        "#1e3a5f",
    "grid.linewidth":    0.6,
    "font.family":       "sans-serif",
    "font.size":         9.5,
})

ACCENT  = "#7dd3fc"
ACCENT2 = "#a78bfa"
POS_CLR = "#f87171"
NEG_CLR = "#34d399"


def styled_fig(w=7, h=4.5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("#0a1628")
    ax.set_facecolor("#0a1628")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#1e3a5f")
    return fig, ax


def fmt(v, d=2):
    try:
        return f"{float(v):.{d}f}"
    except Exception:
        return "—"


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero">
  <div class="eyebrow">Deep Learning · Structural MRI · OASIS-3</div>
  <h1>🧠 Brain Age Lab</h1>
  <p>
    An interactive research demo of the V3 SFCN brain-age model trained on OASIS-3.
    Navigate the four layers: <b>MRI Viewer</b> → <b>Prediction</b> →
    <b>Brain Age Gap</b> → <b>Performance Explorer</b>.
  </p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Load cached tables
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_tables():
    raw     = pd.read_csv(DATA / "raw_m1_predictions.csv")
    bag     = pd.read_csv(DATA / "bag_m1_predictions.csv")
    metrics = json.loads((DATA / "v3_accuracy_m1.json").read_text())
    demo    = pd.read_csv(DATA / "demo_predictions_100.csv")
    return raw, bag, metrics, demo


with st.spinner("Loading data tables…"):
    raw, bag, metrics, demo = load_tables()


# ---------------------------------------------------------------------------
# Scan catalogue (demo subset that has .npz files on disk)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def build_scan_catalogue():
    records = []
    for _, row in demo.iterrows():
        fp = str(row["file_path"]).replace("/mnt/d/", "D:/").replace("/", "\\")
        p  = Path(fp)
        if p.exists():
            records.append({
                "label":         row["mri_label"],
                "subject_id":    row["subject_id"],
                "age":           row.get("age",          float("nan")),
                "sex":           row.get("sex",          "?"),
                "fold":          row.get("fold",         "?"),
                "predicted_age": row.get("predicted_age", float("nan")),
                "brain_age":     row.get("brain_age",    float("nan")),
                "BAG":           row.get("BAG",          float("nan")),
                "MMSE":          row.get("MMSE",         float("nan")),
                "CDRTOT":        row.get("CDRTOT",       float("nan")),
                "npz_path":      str(p),
            })
    return records


catalogue = build_scan_catalogue()


@st.cache_data(show_spinner=False)
def load_volume(npz_path: str) -> np.ndarray:
    with np.load(npz_path, allow_pickle=False) as f:
        return np.asarray(f["data"], dtype=np.float32)


# ---------------------------------------------------------------------------
# Model (load once)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_model():
    return load_model()


# ===========================================================================
# TABS
# ===========================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🔬 Layer 1 · MRI Viewer",
    "⚡ Layer 2 · Prediction",
    "📊 Layer 3 · Brain Age Gap",
    "🔭 Layer 4 · Performance",
])


# ===========================================================================
# LAYER 1 — MRI Viewer & Scan Selection
# ===========================================================================
with tab1:
    st.markdown(
        '<div class="layer-badge l1">Layer 1 — MRI Viewer &amp; Scan Selection</div>',
        unsafe_allow_html=True,
    )

    if not catalogue:
        st.error(
            "No preprocessed NPZ files were found on disk. "
            f"Expected path: {PREP}"
        )
    else:
        col_sel, col_viewer = st.columns([1, 2.5], gap="large")

        with col_sel:
            st.markdown(
                '<div class="section-header">📂 Select Scan</div>',
                unsafe_allow_html=True,
            )

            age_vals    = [r["age"] for r in catalogue if not np.isnan(r["age"])]
            age_min_val = int(np.floor(min(age_vals))) if age_vals else 40
            age_max_val = int(np.ceil(max(age_vals)))  if age_vals else 100

            age_filt = st.slider(
                "Filter by age",
                min_value=age_min_val, max_value=age_max_val,
                value=(age_min_val, age_max_val),
                key="l1_age_filt",
            )
            sex_filt = st.multiselect(
                "Sex", ["M", "F"], default=["M", "F"], key="l1_sex",
            )

            filtered_cat = [
                r for r in catalogue
                if (np.isnan(r["age"]) or age_filt[0] <= r["age"] <= age_filt[1])
                and r["sex"] in sex_filt
            ]

            if not filtered_cat:
                st.warning("No scans match the filters.")
                st.stop()

            scan_labels = [
                (
                    f"{r['label']}  ·  {r['age']:.1f}y  ·  {r['sex']}"
                    if not np.isnan(r["age"])
                    else f"{r['label']}  ·  {r['sex']}"
                )
                for r in filtered_cat
            ]
            chosen_idx = st.selectbox(
                f"Available scans ({len(filtered_cat)} shown)",
                range(len(filtered_cat)),
                format_func=lambda i: scan_labels[i],
                key="l1_scan",
            )
            scan = filtered_cat[chosen_idx]

            # Metadata card
            st.markdown("---")
            st.markdown(
                '<div class="section-header">🪪 Scan Metadata</div>',
                unsafe_allow_html=True,
            )
            meta_items = [
                ("Subject ID",  scan["subject_id"]),
                ("MRI Label",   scan["label"]),
                ("Age at scan", f"{scan['age']:.2f} y" if not np.isnan(scan["age"]) else "N/A"),
                ("Sex",         scan["sex"]),
                ("CV Fold",     str(scan["fold"])),
                ("MMSE",        fmt(scan["MMSE"], 0) if not np.isnan(scan["MMSE"]) else "N/A"),
                ("CDR Total",   fmt(scan["CDRTOT"], 1) if not np.isnan(scan["CDRTOT"]) else "N/A"),
            ]
            for k, v in meta_items:
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'padding:.22rem 0;border-bottom:1px solid #1e3a5f;">'
                    f'<span style="color:#64748b;font-size:.83rem;">{k}</span>'
                    f'<span style="font-weight:600;font-size:.85rem;color:#e2e8f0;">{v}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # Upload override
            st.markdown("---")
            st.markdown(
                '<div class="section-header">📤 Upload Your NPZ</div>',
                unsafe_allow_html=True,
            )
            uploaded = st.file_uploader(
                "Upload preprocessed NPZ (overrides catalogue scan)",
                type=["npz"], accept_multiple_files=False, key="l1_upload",
            )
            if uploaded:
                st.code(
                    f"Required shape: {EXPECTED_SHAPE}\nRequired key: data",
                    language="text",
                )

        # Viewer column
        with col_viewer:
            volume = None
            volume_source = "catalogue"

            if uploaded is not None:
                try:
                    raw_bytes = uploaded.read()
                    with np.load(io.BytesIO(raw_bytes), allow_pickle=False) as f:
                        if "data" not in f.files:
                            st.error("NPZ must contain an array named 'data'.")
                        else:
                            vol_tmp = np.squeeze(
                                np.asarray(f["data"], dtype=np.float32)
                            )
                            if vol_tmp.shape != EXPECTED_SHAPE:
                                st.error(
                                    f"Expected shape {EXPECTED_SHAPE}, "
                                    f"got {vol_tmp.shape}."
                                )
                            elif not np.isfinite(vol_tmp).all():
                                st.error("Volume contains NaN/Inf values.")
                            else:
                                volume = vol_tmp
                                volume_source = "upload"
                except Exception as exc:
                    st.error(f"Could not read NPZ: {exc}")

            if volume is None:
                try:
                    vol_tmp = load_volume(scan["npz_path"])
                    volume  = np.squeeze(vol_tmp)
                    if volume.shape != EXPECTED_SHAPE:
                        st.error(f"Unexpected shape {volume.shape}.")
                        volume = None
                except Exception as exc:
                    st.error(f"Could not load scan: {exc}")

            if volume is not None:
                st.markdown(
                    '<div class="section-header">🖼 Orthogonal Slice Viewer</div>',
                    unsafe_allow_html=True,
                )

                sx, sy, sz = volume.shape

                # Sliders in display order: Axial (Z) → Coronal (Y) → Sagittal (X)
                c1, c2, c3 = st.columns(3)
                axi_idx = c1.slider("Axial (Z)",    0, sz - 1, sz // 2, key="axi")
                cor_idx = c2.slider("Coronal (Y)",  0, sy - 1, sy // 2, key="cor")
                sag_idx = c3.slider("Sagittal (X)", 0, sx - 1, sx // 2, key="sag")

                # constrained_layout avoids tight_layout + GridSpec warnings
                fig = plt.figure(figsize=(13, 4.4), constrained_layout=True)
                fig.patch.set_facecolor("#0a1628")
                gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.06)

                view_data = [
                    ("Axial",    np.rot90(volume[:, :, axi_idx]), f"z={axi_idx}"),
                    ("Coronal",  np.rot90(volume[:, cor_idx, :]), f"y={cor_idx}"),
                    ("Sagittal", np.rot90(volume[sag_idx, :, :]), f"x={sag_idx}"),
                ]
                for i, (title, slc, coord) in enumerate(view_data):
                    ax_v = fig.add_subplot(gs[i])
                    ax_v.imshow(slc, cmap="gray", interpolation="bilinear", aspect="auto")
                    ax_v.set_title(
                        f"{title}  ({coord})",
                        color="#bfdbfe", fontsize=10, fontweight="bold", pad=6,
                    )
                    ax_v.axis("off")
                    for spine in ax_v.spines.values():
                        spine.set_visible(True)
                        spine.set_edgecolor(ACCENT)
                        spine.set_linewidth(1.2)

                st.pyplot(fig, clear_figure=True)

                # Intensity profile
                with st.expander(
                    "📈 Intensity profile (centre row, axial slice)", expanded=False
                ):
                    mid_row = volume[:, sy // 2, axi_idx]
                    fig2, ax2 = styled_fig(11, 2.4)
                    ax2.plot(mid_row, color=ACCENT, linewidth=1.2, alpha=0.85)
                    ax2.fill_between(
                        range(len(mid_row)), mid_row, alpha=0.18, color=ACCENT
                    )
                    ax2.set_xlabel("Voxel index (X)")
                    ax2.set_ylabel("Intensity (z-score)")
                    ax2.grid(True, alpha=0.3)
                    fig2.tight_layout()
                    st.pyplot(fig2, clear_figure=True)

                # Store for Layer 2
                st.session_state["_volume"] = volume
                st.session_state["_scan"]   = scan
                st.session_state["_vsrc"]   = volume_source

                if volume_source == "upload":
                    st.info(
                        "📤 Showing **uploaded** volume. "
                        "Go to **Layer 2** to run inference."
                    )
                else:
                    st.info(
                        f"Showing **{scan['label']}** from the OASIS-3 demo collection. "
                        "Go to **Layer 2** to run inference."
                    )


# ===========================================================================
# LAYER 2 — Real-time Brain Age Prediction
# ===========================================================================
with tab2:
    st.markdown(
        '<div class="layer-badge l2">Layer 2 — Real-time Brain Age Prediction</div>',
        unsafe_allow_html=True,
    )

    col_info2, col_pred2 = st.columns([1.3, 2], gap="large")

    with col_info2:
        st.markdown(
            '<div class="section-header">🤖 Model Info</div>',
            unsafe_allow_html=True,
        )
        st.markdown("""
<div class="info-box">
<b>Architecture:</b> SFCN (Simplified Fully Convolutional Network)<br>
<b>Checkpoint:</b> <code>m1_f0_s0.pt</code> — V3 experiment, fold 0, seed 0<br>
<b>Input:</b> 160 × 192 × 160 voxel brain-only z-scored NPZ<br>
<b>Output:</b> 64-bin age probability distribution<br>
<b>Prediction:</b> Expected value of the bin distribution<br>
<b>Device:</b> CPU (research demo)
</div>
""", unsafe_allow_html=True)

        st.markdown(
            '<div class="section-header" style="margin-top:1.2rem;">📌 Active Scan</div>',
            unsafe_allow_html=True,
        )
        if "_scan" in st.session_state:
            s = st.session_state["_scan"]
            st.markdown(
                f'<div class="info-box">'
                f'<b>Label:</b> {s["label"]}<br>'
                f'<b>Subject:</b> {s["subject_id"]}<br>'
                f'<b>Age:</b> {fmt(s["age"])} y<br>'
                f'<b>Sex:</b> {s["sex"]}'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("Go to **Layer 1** and select a scan first.")

    with col_pred2:
        st.markdown(
            '<div class="section-header">⚡ Run Inference</div>',
            unsafe_allow_html=True,
        )

        volume_ready = (
            "_volume" in st.session_state
            and st.session_state["_volume"] is not None
        )

        if not volume_ready:
            st.warning("No volume loaded — select a scan in **Layer 1** first.")
        else:
            st.markdown(
                f'<p style="color:#64748b;font-size:.88rem;">Volume shape: '
                f'<code>{st.session_state["_volume"].shape}</code></p>',
                unsafe_allow_html=True,
            )

        predict_btn = st.button(
            "🧠  Predict Brain Age",
            type="primary",
            use_container_width=True,
            disabled=not volume_ready,
            key="predict_btn",
        )

        if predict_btn and volume_ready:
            vol = st.session_state["_volume"]
            with st.spinner("Loading checkpoint and running CPU inference…"):
                bundle  = get_model()
                t0      = time.perf_counter()
                result  = predict_array(vol, model_bundle=bundle)
                elapsed = time.perf_counter() - t0

            st.session_state["_result"]  = result
            st.session_state["_elapsed"] = elapsed

        if "_result" in st.session_state:
            result   = st.session_state["_result"]
            elapsed  = st.session_state["_elapsed"]
            pred_age = result["predicted_age"]

            st.success("✅ Inference complete")

            r1, r2, r3 = st.columns(3)
            r1.metric("🧠 Predicted Brain Age", f"{pred_age:.1f} y")
            r2.metric("⏱ Inference Latency",   f"{elapsed:.2f} s")
            r3.metric("📌 Checkpoint",          result.get("checkpoint", "m1_f0_s0"))

            # Illustrative probability distribution
            st.markdown(
                '<div class="section-header" style="margin-top:1rem;">'
                '📊 Model Confidence Visualization</div>',
                unsafe_allow_html=True,
            )
            bins_data = json.loads(
                (ROOT / "checkpoints" / "bins_v3.json").read_text()
            )
            bin_min  = bins_data["bin_min"]
            bin_step = bins_data["bin_step"]
            n_bins   = bins_data["n_bins"]
            centers  = (
                bin_min + bin_step / 2 + np.arange(n_bins) * bin_step
            )
            sigma        = 5.5
            pseudo_probs = np.exp(-0.5 * ((centers - pred_age) / sigma) ** 2)
            pseudo_probs /= pseudo_probs.sum()

            fig3, ax3 = styled_fig(10, 3.2)
            ax3.bar(
                centers, pseudo_probs,
                width=bin_step * 0.85, color=ACCENT, alpha=0.65,
                label="Probability mass",
            )
            ax3.axvline(
                pred_age, color=POS_CLR, linewidth=2, linestyle="--",
                label=f"Predicted: {pred_age:.1f} y",
            )
            if "_scan" in st.session_state:
                chron = st.session_state["_scan"]["age"]
                if not np.isnan(chron):
                    ax3.axvline(
                        chron, color=NEG_CLR, linewidth=2, linestyle=":",
                        label=f"Chronological: {chron:.1f} y",
                    )
            ax3.set_xlabel("Age (years)")
            ax3.set_ylabel("Probability mass")
            ax3.legend(framealpha=0, labelcolor="#e2e8f0", fontsize=9)
            ax3.grid(True, alpha=0.25)
            fig3.tight_layout()
            st.pyplot(fig3, clear_figure=True)

            st.caption(
                "ℹ️ The distribution above is an illustrative Gaussian centred on the model's "
                "predicted expected age. The actual SFCN head outputs 64 discrete softmax bins."
            )

            st.session_state["_pred_age"] = pred_age


# ===========================================================================
# LAYER 3 — Brain Age Gap & Error Analysis
# ===========================================================================
with tab3:
    st.markdown(
        '<div class="layer-badge l3">Layer 3 — Brain Age Gap &amp; Error Analysis</div>',
        unsafe_allow_html=True,
    )

    left3, right3 = st.columns([1.3, 2], gap="large")

    with left3:
        st.markdown(
            '<div class="section-header">🎯 BAG for Selected Scan</div>',
            unsafe_allow_html=True,
        )

        has_pred = "_pred_age" in st.session_state
        has_scan = "_scan"     in st.session_state

        if has_pred and has_scan:
            pred_age = st.session_state["_pred_age"]
            scan_s   = st.session_state["_scan"]
            chron    = scan_s["age"]

            if not np.isnan(chron):
                bag_val  = pred_age - chron
                abs_err  = abs(bag_val)
                bag_sign = "+" if bag_val >= 0 else ""

                if bag_val > 0.5:
                    bag_cls, interp_col = "color:#f87171", "#f87171"
                    interp_text = (
                        "⬆️ Predicted older than chronological age — brain may appear "
                        "more aged than peers of the same age."
                    )
                elif bag_val < -0.5:
                    bag_cls, interp_col = "color:#34d399", "#34d399"
                    interp_text = (
                        "⬇️ Predicted younger than chronological age — brain may appear "
                        "less aged than peers of the same age."
                    )
                else:
                    bag_cls, interp_col = "color:#fbbf24", "#fbbf24"
                    interp_text = (
                        "↔️ Predicted age is close to chronological age — within "
                        "expected model variation."
                    )

                st.markdown(
                    f'<div style="text-align:center;padding:1.2rem;'
                    f'background:rgba(15,30,60,0.6);'
                    f'border:1px solid rgba(125,211,252,0.15);border-radius:16px;">'
                    f'<div style="font-size:.75rem;letter-spacing:.1em;color:#64748b;'
                    f'text-transform:uppercase;margin-bottom:.4rem;">Brain Age Gap (BAG)</div>'
                    f'<div style="font-size:2.2rem;font-weight:700;{bag_cls}">'
                    f'{bag_sign}{bag_val:.2f} years</div>'
                    f'<div style="font-size:.8rem;color:#64748b;margin-top:.4rem;">'
                    f'Absolute error: '
                    f'<b style="color:#e2e8f0;">{abs_err:.2f} y</b></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                c1, c2 = st.columns(2)
                c1.metric("Chronological Age", f"{chron:.1f} y")
                c2.metric("Predicted Age",     f"{pred_age:.1f} y")

                st.markdown(
                    f'<div class="info-box" '
                    f'style="border-left:3px solid {interp_col};margin-top:.8rem;">'
                    f'{interp_text}</div>',
                    unsafe_allow_html=True,
                )

                st.warning(
                    "⚠️ **Disclaimer**: BAG is a model-derived research estimate, "
                    "not a clinical diagnosis. Positive or negative gaps reflect model "
                    "uncertainty and training distribution effects — not brain pathology."
                )
            else:
                st.info("Chronological age not available for this scan.")
                st.metric("Predicted Age", f"{pred_age:.1f} y")
        else:
            st.info(
                "Complete **Layer 2** (run a prediction) to see the BAG "
                "for the selected scan."
            )

    with right3:
        st.markdown(
            '<div class="section-header">🔄 Compare Multiple Scans</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Uses precomputed V3 cross-validation predictions — "
            "no re-inference required."
        )

        demo_with_bag = demo.dropna(subset=["age", "predicted_age"]).copy()
        demo_with_bag["BAG_computed"] = (
            demo_with_bag["predicted_age"] - demo_with_bag["age"]
        )
        demo_with_bag["abs_error"] = demo_with_bag["BAG_computed"].abs()

        subj_options = demo_with_bag["mri_label"].tolist()
        default_sel  = subj_options[: min(6, len(subj_options))]
        selected_scans = st.multiselect(
            "Select demo scans to compare",
            options=subj_options,
            default=default_sel,
            key="l3_multisel",
        )

        if selected_scans:
            compare_df = demo_with_bag[
                demo_with_bag["mri_label"].isin(selected_scans)
            ].copy()

            fig4, ax4 = styled_fig(10, 4.2)
            xs      = np.arange(len(compare_df))
            labels4 = compare_df["mri_label"].tolist()
            bag_arr = compare_df["BAG_computed"].values
            colors4 = [POS_CLR if b > 0 else NEG_CLR for b in bag_arr]

            ax4.bar(xs, bag_arr, color=colors4, alpha=0.8, width=0.6)
            ax4.axhline(0, color=ACCENT, linewidth=1, linestyle="--")
            ax4.set_xticks(xs)
            ax4.set_xticklabels(
                labels4, rotation=40, ha="right", fontsize=7.5
            )
            ax4.set_ylabel("Brain Age Gap (years)")
            ax4.set_title(
                "BAG per Scan  (red = older-looking, green = younger-looking)",
                fontsize=10,
            )
            ax4.grid(axis="y", alpha=0.3)
            fig4.tight_layout()
            st.pyplot(fig4, clear_figure=True)

            tbl = compare_df[
                ["mri_label", "subject_id", "age",
                 "predicted_age", "BAG_computed", "abs_error"]
            ].copy()
            tbl.columns = [
                "MRI Label", "Subject", "Chron. Age",
                "Pred. Age", "BAG", "Abs. Error",
            ]
            st.dataframe(tbl.round(2), use_container_width=True, hide_index=True)
        else:
            st.info("Select at least one scan above.")


# ===========================================================================
# LAYER 4 — Research Performance Explorer
# ===========================================================================
with tab4:
    st.markdown(
        '<div class="layer-badge l4">Layer 4 — Research Performance Explorer</div>',
        unsafe_allow_html=True,
    )

    # Key metrics banner
    st.markdown(
        '<div class="section-header">'
        '📐 V3 Evaluation Metrics (held-out cross-validation)</div>',
        unsafe_allow_html=True,
    )

    m1c, m2c, m3c, m4c, m5c, m6c, m7c = st.columns(7)
    m1c.metric(
        "Bias-corr. MAE",
        f"{metrics['MAE']:.2f} y",
        delta=f"raw {metrics['raw_MAE_uncorrected']:.2f} y",
        delta_color="inverse",
    )
    m2c.metric("RMSE",             f"{metrics['RMSE']:.2f} y")
    m3c.metric("Pearson r",        f"{metrics['pearson_r']:.3f}")
    m4c.metric("R²",               f"{metrics['R2']:.3f}")
    m5c.metric(
        "3-class accuracy",
        f"{100 * metrics['age_group_acc_3class']:.1f}%",
        delta=f"baseline {100 * metrics['age_group_majority_baseline']:.1f}%",
    )
    m6c.metric("Median AE",  f"{metrics['median_abs_error']:.2f} y")
    m7c.metric("CN scans",   f"{int(metrics['n_CN_scans']):,}")

    st.markdown("---")

    plot_df  = raw.dropna(subset=["age", "predicted_age"]).copy()
    if len(plot_df) > 3000:
        plot_df = plot_df.sample(3000, random_state=42)
    err_vals = plot_df["predicted_age"] - plot_df["age"]

    lcol4, rcol4 = st.columns(2, gap="medium")

    # Predicted vs actual
    with lcol4:
        st.markdown(
            '<div class="section-header">🎯 Predicted vs Chronological Age</div>',
            unsafe_allow_html=True,
        )
        fig5, ax5 = styled_fig(6.4, 4.6)
        sc = ax5.scatter(
            plot_df["age"], plot_df["predicted_age"],
            s=8, alpha=0.45,
            c=err_vals.abs(), cmap="plasma", vmin=0, vmax=12,
        )
        lo4 = float(min(plot_df["age"].min(), plot_df["predicted_age"].min()))
        hi4 = float(max(plot_df["age"].max(), plot_df["predicted_age"].max()))
        ax5.plot(
            [lo4, hi4], [lo4, hi4], "--",
            color=ACCENT, linewidth=1.5, label="Perfect prediction",
        )
        cb = fig5.colorbar(sc, ax=ax5, shrink=0.8, pad=0.02)
        cb.set_label("|Error| (years)", color="#94a3b8", fontsize=8)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="#94a3b8", fontsize=7.5)

        if "_pred_age" in st.session_state and "_scan" in st.session_state:
            s4  = st.session_state["_scan"]
            pa4 = st.session_state["_pred_age"]
            if not np.isnan(s4["age"]):
                ax5.scatter(
                    [s4["age"]], [pa4], s=140, color=POS_CLR,
                    zorder=5, marker="*", label="Selected scan",
                )

        ax5.legend(framealpha=0, labelcolor="#e2e8f0", fontsize=8.5)
        ax5.set_xlabel("Chronological Age (years)")
        ax5.set_ylabel("Predicted Age (years)")
        ax5.grid(True, alpha=0.25)
        fig5.tight_layout()
        st.pyplot(fig5, clear_figure=True)

    # Error distribution
    with rcol4:
        st.markdown(
            '<div class="section-header">📉 Prediction Error Distribution</div>',
            unsafe_allow_html=True,
        )
        fig6, ax6 = styled_fig(6.4, 4.6)
        n6, bins6, patches6 = ax6.hist(err_vals, bins=40, edgecolor="#0a1628", linewidth=0.3)
        norm6 = matplotlib.colors.Normalize(vmin=bins6[0], vmax=bins6[-1])
        cmap6 = matplotlib.colormaps["coolwarm"]
        for patch6, left6 in zip(patches6, bins6[:-1]):
            patch6.set_facecolor(cmap6(norm6(left6)))
            patch6.set_alpha(0.8)
        ax6.axvline(0, color=ACCENT, linewidth=2, linestyle="--", label="Zero error")
        ax6.axvline(
            err_vals.mean(), color="#fbbf24", linewidth=1.5, linestyle=":",
            label=f"Mean: {err_vals.mean():.2f} y",
        )
        ax6.set_xlabel("Predicted − Chronological Age (years)")
        ax6.set_ylabel("Scan count")
        ax6.legend(framealpha=0, labelcolor="#e2e8f0", fontsize=8.5)
        ax6.grid(axis="y", alpha=0.25)
        fig6.tight_layout()
        st.pyplot(fig6, clear_figure=True)

    st.markdown("---")
    b1col, b2col = st.columns(2, gap="medium")

    # MAE by age group
    with b1col:
        st.markdown(
            '<div class="section-header">📊 MAE by Age Group</div>',
            unsafe_allow_html=True,
        )
        bands4 = [
            (0,  60, "<60"),
            (60, 65, "60–64"),
            (65, 70, "65–69"),
            (70, 75, "70–74"),
            (75, 80, "75–79"),
            (80, 90, "80–89"),
            (90, 130, "90+"),
        ]
        grp_rows4 = []
        for low4, high4, lbl4 in bands4:
            part4 = raw[
                (raw["age"] >= low4) & (raw["age"] < high4)
            ].dropna(subset=["age", "predicted_age"])
            if len(part4):
                mae_g4 = float(
                    np.mean(np.abs(part4["predicted_age"] - part4["age"]))
                )
                grp_rows4.append(
                    {"Age Group": lbl4, "Scans": len(part4), "MAE (years)": mae_g4}
                )

        if grp_rows4:
            grp_df4   = pd.DataFrame(grp_rows4)
            bar_clrs4 = matplotlib.colormaps["viridis"](
                np.linspace(0.3, 0.9, len(grp_df4))
            )
            fig7, ax7 = styled_fig(6.4, 4.2)
            bars7 = ax7.bar(
                grp_df4["Age Group"], grp_df4["MAE (years)"],
                color=bar_clrs4, alpha=0.85, width=0.6,
            )
            for bar7, val7 in zip(bars7, grp_df4["MAE (years)"]):
                ax7.text(
                    bar7.get_x() + bar7.get_width() / 2,
                    bar7.get_height() + 0.05,
                    f"{val7:.2f}", ha="center", va="bottom",
                    fontsize=8, color="#e2e8f0",
                )
            ax7.axhline(
                metrics["MAE"], color=ACCENT, linewidth=1.5, linestyle="--",
                label=f"Overall MAE: {metrics['MAE']:.2f} y",
            )
            ax7.legend(framealpha=0, labelcolor="#e2e8f0", fontsize=8.5)
            ax7.set_xlabel("Age Group")
            ax7.set_ylabel("MAE (years)")
            ax7.grid(axis="y", alpha=0.25)
            fig7.tight_layout()
            st.pyplot(fig7, clear_figure=True)
            st.dataframe(grp_df4.round(3), use_container_width=True, hide_index=True)

    # Training curve + BAG vs age
    with b2col:
        st.markdown(
            '<div class="section-header">📈 Training Loss Curve</div>',
            unsafe_allow_html=True,
        )
        meta_ckpt = json.loads(
            (ROOT / "checkpoints" / "m1_f0_s0.json").read_text()
        )
        history8  = meta_ckpt.get("history", [])
        if history8:
            epochs8     = [h["epoch"]      for h in history8]
            train_loss8 = [h["train_loss"] for h in history8]
            val_mae8    = [h["val_mae"]    for h in history8]
            best_val8   = meta_ckpt.get("best_val_mae", None)

            fig8, ax8 = styled_fig(6.4, 4.2)
            ax8r = ax8.twinx()
            l1h, = ax8.plot(
                epochs8, train_loss8, color=ACCENT, linewidth=2, label="Train loss"
            )
            l2h, = ax8r.plot(
                epochs8, val_mae8, color=ACCENT2,
                linewidth=2, linestyle="--", label="Val MAE",
            )
            if best_val8 is not None:
                ax8r.axhline(
                    best_val8, color=POS_CLR, linewidth=1.2, linestyle=":",
                    label=f"Best val MAE: {best_val8:.2f} y",
                )
            ax8.set_xlabel("Epoch")
            ax8.set_ylabel("Train Loss (KL-div)", color=ACCENT)
            ax8r.set_ylabel("Val MAE (years)", color=ACCENT2)
            ax8r.spines["right"].set_color("#1e3a5f")
            ax8r.tick_params(colors="#64748b")
            ax8r.yaxis.label.set_color(ACCENT2)
            ax8.yaxis.label.set_color(ACCENT)
            leg_lines  = [l1h, l2h]
            leg_labels = ["Train loss", "Val MAE"]
            if best_val8 is not None:
                leg_labels.append(f"Best: {best_val8:.2f} y")
            ax8.legend(
                leg_lines, leg_labels,
                framealpha=0, labelcolor="#e2e8f0",
                fontsize=8.5, loc="upper right",
            )
            ax8.grid(True, alpha=0.2)
            fig8.tight_layout()
            st.pyplot(fig8, clear_figure=True)

        st.markdown(
            '<div class="section-header" style="margin-top:1rem;">'
            '🔗 BAG vs Chronological Age</div>',
            unsafe_allow_html=True,
        )
        bag_plot9 = raw.dropna(subset=["age", "predicted_age"]).copy()
        bag_plot9["bag_v"] = bag_plot9["predicted_age"] - bag_plot9["age"]
        if len(bag_plot9) > 2000:
            bag_plot9 = bag_plot9.sample(2000, random_state=0)

        fig9, ax9 = styled_fig(6.4, 3.6)
        ax9.scatter(
            bag_plot9["age"], bag_plot9["bag_v"],
            s=7, alpha=0.38,
            c=bag_plot9["bag_v"].abs(), cmap="RdYlGn_r", vmin=0, vmax=10,
        )
        ax9.axhline(0, color=ACCENT, linewidth=1.5, linestyle="--")
        ax9.set_xlabel("Chronological Age (years)")
        ax9.set_ylabel("BAG (Predicted − Actual)")
        ax9.grid(True, alpha=0.22)
        st.caption(
            f"Slope = {metrics['BAG_vs_age_slope']:.4f}  |  "
            f"p = {metrics['BAG_vs_age_p']:.3f}  |  "
            f"Mean BAG (CN) = {metrics['mean_BAG_CN']:.3f} y"
        )
        fig9.tight_layout()
        st.pyplot(fig9, clear_figure=True)

    st.markdown("---")

    with st.expander("📋 Full metrics JSON", expanded=False):
        st.json(metrics)

    with st.expander(
        "🔍 Explore full prediction table (V3 cross-val)", expanded=False
    ):
        age_range_full = st.slider(
            "Filter chronological age",
            int(np.floor(raw["age"].min())),
            int(np.ceil(raw["age"].max())),
            (int(np.floor(raw["age"].min())), int(np.ceil(raw["age"].max()))),
            key="l4_age_filter",
        )
        disp_df = raw[
            raw["age"].between(age_range_full[0], age_range_full[1])
        ].copy()
        disp_df["abs_error"] = (
            disp_df["predicted_age"] - disp_df["age"]
        ).abs()
        show_cols = [
            c for c in
            ["subject_id", "mri_label", "age", "predicted_age",
             "abs_error", "sex", "fold", "n_models"]
            if c in disp_df.columns
        ]
        st.dataframe(disp_df[show_cols].round(3), use_container_width=True, hide_index=True)
        st.caption(f"{len(disp_df):,} rows shown.")

    with st.expander("📖 Methodology & Provenance", expanded=False):
        st.markdown("""
**Dataset:** OASIS-3 longitudinal study, cognitively normal (CN) subset.  
**Preprocessing (V3):** skull-strip → affine registration to MNI152 1 mm → brain-only z-score →
160×192×160 crop/resample → NPZ export.  
**Model:** SFCN — 6-layer 3D CNN feature extractor, 64-bin softmax head, expected-age
prediction via probability-weighted sum of bin centres.  
**Training:** 5-fold cross-validation; checkpoint `m1_f0_s0` is fold 0, seed 0
(best_val_mae = 2.80 y at epoch 5).  
**Input mode (m1):** approximate UK Biobank mean-normalisation applied to the brain-only
z-score (cv = 0.35).  
**Bias correction:** linear regression of raw predicted age on chronological age per fold
(invert method).  
**Evaluation set:** 1,896 CN scans from 767 subjects (held-out folds).

> *Research use only. Brain-age estimates are not a clinical diagnosis.*
        """)
