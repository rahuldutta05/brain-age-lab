"""Brain Age Lab — five-layer Streamlit demo.

Layer 0 — Raw MRI viewer & scan selection
Layer 1 — Preprocessed MRI viewer with metadata (auto-fetched from preprocessed mri/)
Layer 2 — Real-time brain-age prediction with latency reporting
Layer 3 — Brain Age Gap (BAG) and per-scan error analysis
Layer 4 — Research performance explorer (test-set charts and metrics)
"""

from pathlib import Path
import json
import time
import io
import re
import gzip

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import streamlit as st
try:
    import nibabel as nib
    _NIBABEL_OK = True
except ImportError:
    _NIBABEL_OK = False

from inference import load_model, predict_array, EXPECTED_SHAPE

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT      = Path(__file__).resolve().parent
DATA      = ROOT / "data"
RAW_MRI   = ROOT / "raw mri"
PREPROC   = ROOT / "preprocessed mri"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Brain Age Lab",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Theme state & styling tokens
# ---------------------------------------------------------------------------
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = True

is_dark = st.session_state["dark_mode"]

if is_dark:
    PAGE_BG        = "#0B1020"
    SURFACE_BG     = "#131F30"
    SURFACE_BORDER = "#1E293B"
    INK            = "#F1F5F9"
    MUTED          = "#94A3B8"
    RULE           = "#334155"
    ACCENT         = "#38BDF8"   # luminous sky cyan
    ACCENT2        = "#94A3B8"
    POS_CLR        = "#F87171"   # vibrant coral red (older-looking / positive gap)
    NEG_CLR        = "#34D399"   # emerald green (younger-looking / negative gap)
    AMBER          = "#FBBF24"
    GRID_CLR       = "#1E293B"
    CARD_BG        = "#131F30"
    INFO_BG        = "rgba(56, 189, 248, 0.08)"
    INFO_BORDER    = "#38BDF8"
    EMPTY_BG       = "rgba(19, 31, 48, 0.65)"
    EMPTY_BORDER   = "#334155"
    SPEC_BG        = "#131F30"
    SPEC_BORDER    = "#1E293B"
    CODE_BG        = "#1E293B"
    CODE_CLR       = "#38BDF8"
    CODE_BORDER    = "#334155"
    SEQ_CMAP       = matplotlib.colors.LinearSegmentedColormap.from_list(
        "dark_error", ["#38BDF8", "#FBBF24", "#F87171"], N=256
    )
else:
    PAGE_BG        = "#FFFFFF"
    SURFACE_BG     = "#F5F5F3"
    SURFACE_BORDER = "#D1D5DB"
    INK            = "#111827"
    MUTED          = "#4B5563"
    RULE           = "#9CA3AF"
    ACCENT         = "#1D4ED8"   # rich blue
    ACCENT2        = "#6B7280"
    POS_CLR        = "#B91C1C"   # strong red
    NEG_CLR        = "#15803D"   # strong green
    AMBER          = "#92400E"
    GRID_CLR       = "#E5E7EB"
    CARD_BG        = "#FFFFFF"
    INFO_BG        = "rgba(29, 78, 216, 0.06)"
    INFO_BORDER    = "#3B82F6"
    EMPTY_BG       = "#F9FAFB"
    EMPTY_BORDER   = "#D1D5DB"
    SPEC_BG        = "#F9FAFB"
    SPEC_BORDER    = "#D1D5DB"
    CODE_BG        = "#EFF6FF"
    CODE_CLR       = "#1D4ED8"
    CODE_BORDER    = "#BFDBFE"
    SEQ_CMAP       = matplotlib.colors.ListedColormap(
        matplotlib.colormaps["Blues"](np.linspace(0.3, 1.0, 256))
    )

st.markdown(f"""
<style>
.block-container {{ padding-top: 2.2rem; max-width: 1320px; }}
[data-testid="stHeader"] {{ background: transparent; }}
hr {{ margin: 1.4rem 0; border-color: {SURFACE_BORDER}; }}

/* Base container */
.stApp {{
    background-color: {PAGE_BG};
    color: {INK};
}}

/* Force all Streamlit text to theme ink color */
.stApp p, .stApp label, .stApp span, .stApp div {{
    color: {INK};
}}

/* Page header */
.hero {{ padding-bottom: 0.8rem; margin-bottom: 1.2rem; border-bottom: 1px solid {SURFACE_BORDER}; }}
.eyebrow {{ font-size: .75rem; letter-spacing: .09em; text-transform: uppercase; color: {ACCENT}; font-weight: 600; }}
.hero h1 {{ margin: .2rem 0 .4rem 0; padding: 0; font-size: 1.85rem; font-weight: 700; color: {INK}; letter-spacing: -0.02em; }}
.hero p {{ margin: 0; color: {MUTED}; font-size: .95rem; max-width: 62rem; line-height: 1.5; }}

/* Layer label and section headings */
.layer-badge {{
    font-size: .75rem;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: {ACCENT};
    background: {INFO_BG};
    border: 1px solid {SURFACE_BORDER};
    padding: 0.25rem 0.7rem;
    border-radius: 9999px;
    display: inline-block;
    margin: .4rem 0 1rem 0;
    font-weight: 600;
}}
.section-header {{
    font-size: 1.02rem;
    font-weight: 600;
    color: {INK};
    padding-bottom: .4rem;
    margin-bottom: .85rem;
    border-bottom: 1px solid {SURFACE_BORDER};
}}

/* Metrics */
div[data-testid="stMetric"] {{
    padding: .5rem .8rem;
    background: {CARD_BG};
    border: 1px solid {SURFACE_BORDER};
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}}
div[data-testid="stMetricLabel"] p {{ font-size: .8rem; color: {MUTED}; font-weight: 500; }}
div[data-testid="stMetricValue"] {{ font-size: 1.55rem !important; font-weight: 700 !important; color: {INK} !important; }}
div[data-testid="stMetricDelta"] {{ color: {ACCENT} !important; font-size: .8rem; }}

.stMarkdown code {{
    color: {CODE_CLR};
    background: {CODE_BG};
    padding: 0.15rem 0.4rem;
    border-radius: 4px;
    border: 1px solid {CODE_BORDER};
}}

/* Notes, key-value rows */
.info-box {{
    border-left: 3px solid {INFO_BORDER};
    background: {INFO_BG};
    padding: .55rem .85rem;
    border-radius: 0 6px 6px 0;
    margin-top: .6rem;
    font-size: .88rem;
    line-height: 1.6;
    color: {INK};
}}
.info-box b {{ color: {INK}; }}
.kv {{
    display: flex;
    justify-content: space-between;
    padding: .35rem 0;
    border-bottom: 1px solid {SURFACE_BORDER};
}}
.kv span:first-child {{ color: {MUTED}; font-size: .84rem; }}
.kv span:last-child {{ font-weight: 600; font-size: .86rem; color: {INK}; }}

.bag-box {{
    text-align: center;
    padding: 1.3rem;
    background: {CARD_BG};
    border: 1px solid {SURFACE_BORDER};
    border-radius: 8px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
}}
.bag-box .lbl {{
    font-size: .75rem;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: {MUTED};
    margin-bottom: .35rem;
    font-weight: 600;
}}
.bag-box .val {{ font-size: 2.15rem; font-weight: 700; }}
.bag-box .sub {{ font-size: .84rem; color: {MUTED}; margin-top: .35rem; }}
.bag-box .sub b {{ color: {INK} !important; }}

/* Empty state and spec cards */
.empty-state {{
    border: 1px dashed {EMPTY_BORDER};
    border-radius: 8px;
    padding: 2.4rem 1.6rem;
    text-align: center;
    background: {EMPTY_BG};
    margin: 0.5rem 0 1.2rem 0;
}}
.empty-state h3 {{
    margin: 0 0 0.5rem 0;
    font-size: 1.15rem;
    color: {INK};
    font-weight: 600;
}}
.empty-state p {{
    color: {MUTED};
    font-size: 0.9rem;
    margin: 0 auto;
    max-width: 34rem;
    line-height: 1.5;
}}
.spec-box {{
    background: {SPEC_BG};
    border: 1px solid {SPEC_BORDER};
    border-radius: 8px;
    padding: 0.95rem 1.15rem;
    font-size: 0.85rem;
    color: {INK};
    margin-top: 0.6rem;
}}
.spec-box b {{ color: {INK}; }}
.spec-box li {{ color: {INK}; }}

/* Streamlit Tabs styling */
.stTabs [data-baseweb="tab-list"] {{
    gap: 8px;
    border-bottom: 1px solid {SURFACE_BORDER};
}}
.stTabs [data-baseweb="tab"] {{
    padding: 8px 16px;
    color: {MUTED};
    border-radius: 6px 6px 0 0;
    font-weight: 500;
}}
.stTabs [aria-selected="true"] {{
    color: {ACCENT} !important;
    border-bottom-color: {ACCENT} !important;
    font-weight: 600 !important;
}}

/* Expanders */
div[data-testid="stExpander"] {{
    background: {CARD_BG};
    border: 1px solid {SURFACE_BORDER};
    border-radius: 8px;
}}
div[data-testid="stExpander"] summary {{
    color: {INK};
}}
div[data-testid="stExpander"] p,
div[data-testid="stExpander"] span,
div[data-testid="stExpander"] label {{
    color: {INK};
}}

/* File uploader */
[data-testid="stFileUploaderDropzone"] {{
    background: {CARD_BG};
    border: 1px dashed {SURFACE_BORDER};
    border-radius: 8px;
}}
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] p {{
    color: {MUTED};
}}

/* Inputs */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {{
    background: {CARD_BG};
    color: {INK};
    border-color: {SURFACE_BORDER};
}}
[data-testid="stSelectbox"] > div > div {{
    background: {CARD_BG};
    color: {INK};
    border-color: {SURFACE_BORDER};
}}

/* Toggle label */
[data-testid="stToggle"] span {{
    color: {INK} !important;
}}

/* Sidebar */
[data-testid="stSidebar"] {{
    background: {SURFACE_BG};
    border-right: 1px solid {SURFACE_BORDER};
}}
[data-testid="stSidebar"] * {{ color: {INK}; }}

/* Dataframe */
[data-testid="stDataFrame"] {{
    border: 1px solid {SURFACE_BORDER};
    border-radius: 8px;
}}

/* Success/info/warning boxes */
[data-testid="stAlert"] {{
    border-radius: 8px;
}}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Matplotlib defaults
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor":  CARD_BG,
    "axes.facecolor":    CARD_BG,
    "axes.edgecolor":    RULE,
    "axes.labelcolor":   INK,
    "axes.titlecolor":   INK,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "text.color":        INK,
    "grid.color":        GRID_CLR,
    "grid.linewidth":    0.6,
    "font.family":       "sans-serif",
    "font.size":         9.5,
})


def styled_fig(w=7, h=4.5):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=CARD_BG)
    ax.set_facecolor(CARD_BG)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(RULE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(INK)
    ax.grid(True, color=GRID_CLR, linewidth=0.6, alpha=0.7)
    return fig, ax


def fmt(v, d=2):
    try:
        return f"{float(v):.{d}f}"
    except Exception:
        return "—"


# ---------------------------------------------------------------------------
# Hero & Theme Control
# ---------------------------------------------------------------------------
col_hero, col_theme = st.columns([4.4, 1.2])

with col_hero:
    st.markdown("""
<div class="hero">
  <div class="eyebrow">Deep Learning · Structural MRI · OASIS-3</div>
  <h1>Brain Age Lab</h1>
  <p>
    An interactive research demo of the V3 SFCN brain-age model trained on OASIS-3.
    Navigate the four layers: <b>MRI Viewer</b> → <b>Prediction</b> →
    <b>Brain Age Gap</b> → <b>Performance Explorer</b>.
  </p>
</div>
""", unsafe_allow_html=True)

with col_theme:
    st.markdown('<div style="display: flex; justify-content: flex-end; align-items: center; padding-top: 1.2rem;">', unsafe_allow_html=True)
    cur_mode = st.session_state.get("dark_mode", True)
    new_mode = st.toggle(
        "Dark Mode" if cur_mode else "Light Mode",
        value=cur_mode,
        key="main_theme_toggle",
    )
    if new_mode != cur_mode:
        st.session_state["dark_mode"] = new_mode
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Display Preferences")
    st.caption("Active Theme: **" + ("Dark Mode" if is_dark else "Light Mode") + "**")
    st.markdown("---")


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
# CSV metadata lookup — match uploaded NPZ filename to demo/raw CSV rows
# ---------------------------------------------------------------------------
def lookup_metadata_from_filename(stem: str) -> dict | None:
    """Try to find metadata (age, sex) in CSVs by matching the NPZ filename stem.

    The stem is something like 'sub-OAS30001_ses-d0129_run-02_T1w_preproc'.
    We try to extract subject_id (e.g. OAS30001) and the session day (e.g. d0129 → 129)
    and look up matching rows in the demo CSV, then the raw CSV as fallback.
    """
    import re

    # Extract OASIS subject ID  e.g. OAS30001
    subj_match = re.search(r"(OAS3\d+)", stem, re.IGNORECASE)
    if not subj_match:
        return None
    subject_id = subj_match.group(1).upper()

    # Extract session day integer e.g. d0129 → 129
    day_match = re.search(r"[_-]d(\d+)", stem, re.IGNORECASE)
    day_val = int(day_match.group(1)) if day_match else None

    # Extract run number e.g. run-02 → 2
    run_match = re.search(r"run[_-](\d+)", stem, re.IGNORECASE)
    run_val = int(run_match.group(1)) if run_match else None

    # Search demo CSV first, then raw CSV
    for df in (demo, raw):
        if "subject_id" not in df.columns:
            continue
        df_sub = df[df["subject_id"].astype(str).str.upper() == subject_id]
        if df_sub.empty:
            continue
        # Narrow by day if available
        if day_val is not None and "day" in df_sub.columns:
            day_rows = df_sub[df_sub["day"].astype(float, errors="ignore") == day_val]
            if not day_rows.empty:
                df_sub = day_rows
        # Narrow by run if available
        if run_val is not None and "run" in df_sub.columns:
            run_rows = df_sub[df_sub["run"].astype(float, errors="ignore") == run_val]
            if not run_rows.empty:
                df_sub = run_rows
        row = df_sub.iloc[0]
        age_v = float(row["age"]) if "age" in df_sub.columns and pd.notna(row.get("age")) else float("nan")
        sex_v = str(row["sex"]).strip() if "sex" in df_sub.columns and pd.notna(row.get("sex")) else "?"
        return {
            "subject_id": subject_id,
            "age":        age_v,
            "sex":        sex_v,
            "MMSE":       float(row["MMSE"])  if "MMSE"   in df_sub.columns and pd.notna(row.get("MMSE"))   else float("nan"),
            "CDRTOT":     float(row["CDRTOT"]) if "CDRTOT" in df_sub.columns and pd.notna(row.get("CDRTOT")) else float("nan"),
        }
    return None


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


def parse_npz_bytes(raw_bytes: bytes) -> np.ndarray:
    """Extract and validate 3D float32 volume from uploaded NPZ bytes."""
    with np.load(io.BytesIO(raw_bytes), allow_pickle=False) as f:
        if "data" in f.files:
            arr = f["data"]
        elif len(f.files) == 1:
            arr = f[f.files[0]]
        else:
            raise KeyError(
                f"NPZ archive must contain an array named 'data'. Available keys: {list(f.files)}"
            )
        vol = np.squeeze(np.asarray(arr, dtype=np.float32))
        if vol.shape != EXPECTED_SHAPE:
            raise ValueError(
                f"Expected volume shape {EXPECTED_SHAPE} (160 × 192 × 160); received {vol.shape}."
            )
        if not np.isfinite(vol).all():
            raise ValueError("The volume contains NaN or infinite values.")
        return vol


def generate_synthetic_volume() -> np.ndarray:
    """Generate a synthetic 3D brain phantom volume of shape (160, 192, 160) for testing."""
    x = np.linspace(-1, 1, 160, dtype=np.float32)[:, None, None]
    y = np.linspace(-1, 1, 192, dtype=np.float32)[None, :, None]
    z = np.linspace(-1, 1, 160, dtype=np.float32)[None, None, :]
    r = (x / 0.7) ** 2 + (y / 0.85) ** 2 + (z / 0.7) ** 2
    brain_mask = r < 1.0
    ventricle_r = (x / 0.18) ** 2 + (y / 0.35) ** 2 + (z / 0.18) ** 2
    ventricle_mask = ventricle_r < 1.0

    vol = np.zeros((160, 192, 160), dtype=np.float32)
    vol[brain_mask] = 1.0 - 0.25 * r[brain_mask]
    vol[ventricle_mask] = 0.1
    rng = np.random.RandomState(42)
    noise = rng.normal(0, 0.05, (160, 192, 160)).astype(np.float32)
    vol[brain_mask] += noise[brain_mask]
    brain_vals = vol[brain_mask]
    vol[brain_mask] = (brain_vals - brain_vals.mean()) / (brain_vals.std() + 1e-6)
    return vol


@st.cache_data(show_spinner=False)
def get_synthetic_npz_bytes() -> bytes:
    """Return compressed bytes of a synthetic demo brain NPZ file."""
    vol = generate_synthetic_volume()
    buf = io.BytesIO()
    np.savez_compressed(buf, data=vol)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Model (load once)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_model():
    return load_model()


# ===========================================================================
# PREPROCESSED SCAN MATCHER & SESSION CATALOGUE
# ===========================================================================
def find_matching_preprocessed_npz(name_or_stem: str) -> Path | None:
    """Find the matching .npz file in PREPROC based on filename or stem."""
    if not PREPROC.exists():
        return None
    stem = Path(name_or_stem).stem.replace(".nii", "").replace("_preproc", "")
    exact = list(PREPROC.rglob(f"{stem}_preproc.npz"))
    if exact:
        return exact[0]
    m_sub = re.search(r"(OAS3\d+)", stem, re.IGNORECASE)
    m_day = re.search(r"d(\d+)", stem, re.IGNORECASE)
    m_run = re.search(r"run[_-](\d+)", stem, re.IGNORECASE)
    if m_sub and m_day:
        sub = m_sub.group(1).upper()
        day = m_day.group(1)
        cands = list(PREPROC.rglob(f"*{sub}*d{day}*.npz"))
        if m_run and cands:
            r = int(m_run.group(1))
            run_cands = [
                c for c in cands
                if f"run-{r:02d}" in c.name or f"run_{r:02d}" in c.name or f"run-{r}" in c.name
            ]
            if run_cands:
                return run_cands[0]
        if cands:
            return cands[0]
    return None


def build_session_catalogue():
    """Return list of dicts describing each session that has BOTH raw and
    preprocessed MRI on disk."""
    sessions = []
    if not RAW_MRI.exists() or not PREPROC.exists():
        return sessions
    npz_map = {p.stem.replace("_preproc", ""): p for p in PREPROC.rglob("*.npz")}
    for raw_nii in sorted(RAW_MRI.rglob("*.nii.gz")):
        stem = raw_nii.stem.replace(".nii", "")
        prep_npz = npz_map.get(stem)
        anat_name = raw_nii.parent.name
        sess_name = raw_nii.parent.parent.name
        jsons = list(raw_nii.parent.glob("*.json"))
        raw_json = jsons[0] if jsons else None

        m = re.match(r"(OAS3\d+)_MR_d(\d+)", sess_name)
        subject_id = m.group(1) if m else sess_name
        day_val    = int(m.group(2)) if m else 0

        sessions.append({
            "session":    sess_name,
            "anat":       anat_name,
            "label":      f"{sess_name} / {anat_name}",
            "subject_id": subject_id,
            "day":        day_val,
            "raw_nii":    str(raw_nii),
            "raw_json":   str(raw_json) if raw_json else None,
            "prep_npz":   str(prep_npz) if prep_npz else None,
            "prep_nii":   None,
        })
    return sessions


@st.cache_data(show_spinner=False)
def load_nifti_volume(nii_path: str) -> np.ndarray:
    """Load a .nii.gz file and return its 3-D float32 data array."""
    img  = nib.load(nii_path)
    data = np.asarray(img.dataobj, dtype=np.float32)
    # Squeeze any extra dims
    while data.ndim > 3:
        data = data[..., 0]
    return data


session_catalogue = build_session_catalogue()

# ===========================================================================
# TABS
# ===========================================================================
tab0, tab1, tab2, tab3, tab4 = st.tabs([
    "Layer 0 · Raw MRI",
    "Layer 1 · Preprocessed",
    "Layer 2 · Prediction",
    "Layer 3 · Brain Age Gap",
    "Layer 4 · Performance",
])


# ===========================================================================
# LAYER 0 — Raw MRI Viewer & Scan Selection
# ===========================================================================
with tab0:
    st.markdown(
        '<div class="layer-badge">Layer 0 — Raw MRI Viewer &amp; Scan Selection</div>',
        unsafe_allow_html=True,
    )

    col_sel0, col_view0 = st.columns([1.15, 2.35], gap="large")

    with col_sel0:
        st.markdown('<div class="section-header">Select Scan</div>', unsafe_allow_html=True)

        if not _NIBABEL_OK:
            st.error("nibabel is not installed. Run: `pip install nibabel`")
        
        chosen_sess = None
        if session_catalogue:
            sess_labels = [s["label"] for s in session_catalogue]
            chosen_idx  = st.selectbox(
                "Session / Run",
                range(len(sess_labels)),
                format_func=lambda i: sess_labels[i],
                key="l0_sess_sel",
            )
            chosen_sess = session_catalogue[chosen_idx]
        else:
            st.info(
                "No preloaded sessions found on this server. "
                "Upload a raw .nii.gz below to view orthogonal slices."
            )

        st.markdown("---")
        st.markdown('<div class="section-header">Or Upload Raw MRI</div>', unsafe_allow_html=True)
        raw_upload = st.file_uploader(
            "Upload raw NIfTI (.nii.gz)",
            type=["gz"],
            help="Upload a raw T1-weighted .nii.gz file.",
            key="l0_raw_upload",
        )

        st.markdown("---")
        load_raw_btn = st.button(
            "Load Selected Scan →",
            type="primary",
            use_container_width=True,
            key="btn_load_raw",
        )

        if load_raw_btn:
            if raw_upload is not None:
                # User-uploaded NIfTI
                try:
                    raw_upload.seek(0)
                    raw_bytes = raw_upload.read()
                    if raw_upload.name.endswith(".gz") or (len(raw_bytes) >= 2 and raw_bytes[:2] == b"\x1f\x8b"):
                        raw_bytes = gzip.decompress(raw_bytes)
                    with io.BytesIO(raw_bytes) as buf:
                        fh   = nib.FileHolder(fileobj=buf)
                        img  = nib.Nifti1Image.from_file_map({"header": fh, "image": fh})
                        vol0 = np.asarray(img.dataobj, dtype=np.float32)
                    while vol0.ndim > 3:
                        vol0 = vol0[..., 0]
                    st.session_state["_raw_vol"]   = vol0
                    raw_stem = Path(raw_upload.name).stem.replace(".nii", "")
                    st.session_state["_raw_label"] = raw_stem
                    st.session_state["_raw_sess"]  = None

                    # Auto-link matching preprocessed NPZ
                    matched_npz = find_matching_preprocessed_npz(raw_upload.name)
                    if matched_npz:
                        st.session_state["_raw_prep_npz"] = str(matched_npz)
                        try:
                            _prep_vol = load_volume(str(matched_npz))
                            st.session_state["_volume"]         = np.squeeze(_prep_vol)
                            st.session_state["_loaded_file_id"] = f"raw_sess_{matched_npz}"
                            st.session_state["_vsrc"]           = "catalogue"
                        except Exception:
                            pass
                    else:
                        st.session_state["_raw_prep_npz"] = None
                        st.session_state.pop("_volume", None)
                        st.session_state.pop("_loaded_file_id", None)

                    # Auto-link matching BIDS JSON
                    if RAW_MRI.exists():
                        matched_jsons = list(RAW_MRI.rglob(f"{raw_stem}.json"))
                        st.session_state["_raw_json"] = str(matched_jsons[0]) if matched_jsons else None
                    else:
                        st.session_state["_raw_json"] = None

                    # Auto-lookup metadata for subsequent layers
                    meta0 = lookup_metadata_from_filename(raw_stem)
                    age0  = meta0["age"] if meta0 else float("nan")
                    sex0  = meta0["sex"] if meta0 else "?"
                    st.session_state["_scan_label"] = raw_stem
                    st.session_state["_scan_age"]   = age0
                    st.session_state["_has_age"]    = not np.isnan(age0)
                    st.session_state["_scan_sex"]   = (
                        "Female" if sex0 == "F" else ("Male" if sex0 == "M" else "Unknown")
                    )

                    # Reset prediction state to run fresh on new scan
                    st.session_state.pop("_scan", None)
                    st.session_state.pop("_pred_age", None)
                    st.session_state.pop("_result", None)
                    st.session_state.pop("_elapsed", None)

                    if matched_npz:
                        st.success("Raw MRI loaded! Matching preprocessed volume found and linked.")
                    else:
                        st.success("Raw MRI loaded from upload.")
                except Exception as exc:
                    st.error(f"Could not load NIfTI: {exc}")
            elif chosen_sess is not None:
                # Load from catalogue
                    try:
                        vol0 = load_nifti_volume(chosen_sess["raw_nii"])
                        st.session_state["_raw_vol"]      = vol0
                        st.session_state["_raw_label"]    = chosen_sess["label"]
                        st.session_state["_raw_json"]     = chosen_sess["raw_json"]
                        st.session_state["_raw_sess"]     = chosen_sess
                        st.session_state["_raw_prep_npz"] = chosen_sess["prep_npz"]
                        st.session_state["_raw_prep_nii"] = chosen_sess["prep_nii"]
                        # Pull metadata for subsequent layers
                        meta0 = lookup_metadata_from_filename(
                            Path(chosen_sess["raw_nii"]).stem.replace(".nii", "")
                        )
                        age0  = meta0["age"] if meta0 else float("nan")
                        sex0  = meta0["sex"] if meta0 else "?"
                        st.session_state["_scan_label"] = chosen_sess["label"]
                        st.session_state["_scan_age"]   = age0
                        st.session_state["_has_age"]    = not np.isnan(age0)
                        st.session_state["_scan_sex"]   = (
                            "Female" if sex0 == "F" else ("Male" if sex0 == "M" else "Unknown")
                        )
                        # Clear prior prediction state
                        st.session_state.pop("_volume", None)
                        st.session_state.pop("_loaded_file_id", None)
                        st.session_state.pop("_scan", None)
                        st.session_state.pop("_pred_age", None)
                        st.session_state.pop("_result", None)
                        st.session_state.pop("_elapsed", None)
                        st.success(f"Loaded **{chosen_sess['label']}**. Navigate to Layer 1 to see the preprocessed volume.")
                    except Exception as exc:
                        st.error(f"Failed to load scan: {exc}")

        # Acquisition metadata panel
        if st.session_state.get("_raw_json"):
            st.markdown("---")
            st.markdown('<div class="section-header">Acquisition Metadata</div>', unsafe_allow_html=True)
            try:
                acq = json.loads(Path(st.session_state["_raw_json"]).read_text())
                acq_items = [
                    ("Manufacturer",       acq.get("Manufacturer", "—")),
                    ("Model",              acq.get("ManufacturersModelName", "—")),
                    ("Field Strength (T)", acq.get("MagneticFieldStrength", "—")),
                    ("Acquisition Type",   acq.get("MRAcquisitionType", "—")),
                    ("Sequence",           acq.get("SeriesDescription", "—")),
                    ("TR (s)",             acq.get("RepetitionTime", "—")),
                    ("TE (s)",             acq.get("EchoTime", "—")),
                    ("TI (s)",             acq.get("InversionTime", "—")),
                    ("Flip Angle (°)",     acq.get("FlipAngle", "—")),
                    ("Slice Thickness",    acq.get("SliceThickness", "—")),
                    ("Base Resolution",    acq.get("BaseResolution", "—")),
                ]
                for k, v in acq_items:
                    st.markdown(
                        f'<div class="kv"><span>{k}</span><span>{v}</span></div>',
                        unsafe_allow_html=True,
                    )
            except Exception:
                pass

    with col_view0:
        raw_vol = st.session_state.get("_raw_vol", None)

        # Self-healing: if raw scan is active but prep_npz wasn't linked yet, auto-match now
        if raw_vol is not None and not st.session_state.get("_raw_prep_npz"):
            curr_stem = st.session_state.get("_raw_label", "")
            matched_fallback = find_matching_preprocessed_npz(curr_stem)
            if matched_fallback:
                st.session_state["_raw_prep_npz"] = str(matched_fallback)
                if st.session_state.get("_volume") is None:
                    try:
                        _pvol = load_volume(str(matched_fallback))
                        st.session_state["_volume"]         = np.squeeze(_pvol)
                        st.session_state["_loaded_file_id"] = f"raw_sess_{matched_fallback}"
                        st.session_state["_vsrc"]           = "catalogue"
                    except Exception:
                        pass
                if not st.session_state.get("_has_age", False):
                    meta_fb = lookup_metadata_from_filename(curr_stem)
                    age_fb  = meta_fb["age"] if meta_fb else float("nan")
                    sex_fb  = meta_fb["sex"] if meta_fb else "?"
                    st.session_state["_scan_age"] = age_fb
                    st.session_state["_has_age"]  = not np.isnan(age_fb)
                    st.session_state["_scan_sex"] = (
                        "Female" if sex_fb == "F" else ("Male" if sex_fb == "M" else "Unknown")
                    )

        if raw_vol is not None:
            st.markdown('<div class="section-header">Orthogonal Slice Viewer — Raw T1w</div>', unsafe_allow_html=True)

            # Normalise for display
            vmin0 = float(np.percentile(raw_vol, 1))
            vmax0 = float(np.percentile(raw_vol, 99))
            disp0 = np.clip(raw_vol, vmin0, vmax0)

            sx0, sy0, sz0 = disp0.shape
            c1r, c2r, c3r = st.columns(3)
            axi0 = c1r.slider("Axial (Z)",    0, sz0-1, sz0//2, key="l0_axi")
            cor0 = c2r.slider("Coronal (Y)",  0, sy0-1, sy0//2, key="l0_cor")
            sag0 = c3r.slider("Sagittal (X)", 0, sx0-1, sx0//2, key="l0_sag")

            fig0 = plt.figure(figsize=(13, 4.4), constrained_layout=True, facecolor=CARD_BG)
            gs0  = gridspec.GridSpec(1, 3, figure=fig0, wspace=0.06)
            view0_data = [
                ("Axial",    np.rot90(disp0[:, :, axi0]), f"z={axi0}"),
                ("Coronal",  np.rot90(disp0[:, cor0, :]), f"y={cor0}"),
                ("Sagittal", np.rot90(disp0[sag0, :, :]), f"x={sag0}"),
            ]
            for i, (title0, slc0, coord0) in enumerate(view0_data):
                ax0v = fig0.add_subplot(gs0[i])
                ax0v.set_facecolor("black")
                ax0v.imshow(slc0, cmap="gray", interpolation="bilinear",
                            aspect="auto", vmin=vmin0, vmax=vmax0)
                ax0v.set_title(f"{title0}  ({coord0})", color=INK,
                               fontsize=10, fontweight="medium", pad=6)
                ax0v.axis("off")
            st.pyplot(fig0, clear_figure=True)

            lbl0 = st.session_state.get("_raw_label", "Raw MRI")
            if st.session_state.get("_raw_prep_npz"):
                st.info(
                    f"Raw scan **{lbl0}** loaded. "
                    "Navigate to **Layer 1 · Preprocessed** to inspect the skull-stripped, "
                    "registered, z-scored volume and run brain-age inference."
                )
            else:
                st.info(f"Raw scan **{lbl0}** loaded (no linked preprocessed file found).")

            # Volume stats
            with st.expander("Raw volume statistics", expanded=False):
                svol = raw_vol[raw_vol > 0]
                st.markdown(
                    f'<div class="kv"><span>Shape</span><span>{raw_vol.shape[0]} × {raw_vol.shape[1]} × {raw_vol.shape[2]}</span></div>'
                    f'<div class="kv"><span>Min / Max</span><span>{raw_vol.min():.1f} / {raw_vol.max():.1f}</span></div>'
                    f'<div class="kv"><span>Mean (non-zero)</span><span>{svol.mean():.1f}</span></div>'
                    f'<div class="kv"><span>Std (non-zero)</span><span>{svol.std():.1f}</span></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown('<div class="section-header">Orthogonal Slice Viewer — Raw T1w</div>', unsafe_allow_html=True)
            st.markdown("""
<div class="empty-state">
  <h3>Select a Raw MRI Scan</h3>
  <p>Choose a session from the dropdown on the left (or upload a <code>.nii.gz</code> file),
  then click <b>Load Selected Scan →</b> to view orthogonal slices of the raw T1-weighted volume.</p>
</div>

<div class="spec-box">
  <b>Available data:</b>
  <ul style="margin: 0.4rem 0 0.2rem 1.2rem; padding: 0;">
    <li><b>50 sessions</b> from OASIS-3 subjects OAS30001–OAS30015</li>
    <li>Raw T1w NIfTI (<code>.nii.gz</code>) + BIDS JSON sidecar</li>
    <li>Paired skull-stripped preprocessed NPZ ready in Layer 1</li>
  </ul>
</div>
""", unsafe_allow_html=True)


# ===========================================================================
# LAYER 1 — Preprocessed MRI Viewer & Scan Input
# ===========================================================================
with tab1:
    st.markdown(
        '<div class="layer-badge">Layer 1 — Preprocessed MRI &amp; Scan Metadata</div>',
        unsafe_allow_html=True,
    )

    # ---- Auto-fetch preprocessed NPZ from Layer 0 session selection --------
    _prep_npz_path = st.session_state.get("_raw_prep_npz", None)
    if _prep_npz_path and st.session_state.get("_volume") is None:
        try:
            _prep_vol = load_volume(_prep_npz_path)
            st.session_state["_volume"]         = np.squeeze(_prep_vol)
            st.session_state["_loaded_file_id"] = f"raw_sess_{_prep_npz_path}"
            st.session_state["_vsrc"]           = "catalogue"
            st.session_state.setdefault("_scan_label", Path(_prep_npz_path).stem)
        except Exception as _exc:
            st.warning(f"Could not auto-load preprocessed NPZ: {_exc}")

    col_input, col_viewer = st.columns([1.15, 2.35], gap="large")

    with col_input:
        st.markdown(
            '<div class="section-header">Scan Input</div>',
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Upload preprocessed MRI (.npz)",
            type=["npz"],
            help="Select a 3D preprocessed brain volume in .npz format (160×192×160 voxels, 'data' array).",
            key="l1_user_upload",
        )

        # Quick demo & sample test helpers
        with st.expander("Need a sample scan to test?", expanded=False):
            st.caption(
                "Generate a synthetic 3D brain phantom to explore the viewer, "
                "run prediction, and test the 4-layer pipeline without an external MRI file."
            )
            c_btn1, c_btn2 = st.columns(2)
            with c_btn1:
                if st.button("Load Demo Scan", use_container_width=True, key="btn_load_synth"):
                    st.session_state["_volume"] = generate_synthetic_volume()
                    st.session_state["_loaded_file_id"] = "synthetic_demo"
                    st.session_state["_vsrc"] = "synthetic"
                    st.session_state["_scan_label"] = "Synthetic_Brain_Demo"
                    st.session_state["_scan_age"] = 65.0
                    st.session_state["_has_age"] = True
                    st.session_state["_scan_sex"] = "Unknown"
                    # Reset prediction state
                    st.session_state.pop("_pred_age", None)
                    st.session_state.pop("_result", None)
                    st.session_state.pop("_elapsed", None)
                    st.rerun()
            with c_btn2:
                st.download_button(
                    label="Download .npz",
                    data=get_synthetic_npz_bytes(),
                    file_name="sample_brain_t1w_preproc.npz",
                    mime="application/octet-stream",
                    use_container_width=True,
                    key="btn_dl_synth",
                )

        # Optional: if catalogue scans happen to exist on disk, allow selection
        if catalogue:
            with st.expander(f"Select from disk catalogue ({len(catalogue)} scans)", expanded=False):
                cat_labels = [
                    f"{r['label']} · {r['age']:.1f}y · {r['sex']}" if not np.isnan(r["age"]) else f"{r['label']} · {r['sex']}"
                    for r in catalogue
                ]
                chosen_cat_idx = st.selectbox(
                    "Catalogue scan",
                    range(len(catalogue)),
                    format_func=lambda i: cat_labels[i],
                    key="sel_cat_scan",
                )
                if st.button("Load catalogue scan", key="btn_load_cat"):
                    chosen_r = catalogue[chosen_cat_idx]
                    try:
                        vol_cat = load_volume(chosen_r["npz_path"])
                        st.session_state["_volume"] = np.squeeze(vol_cat)
                        st.session_state["_loaded_file_id"] = f"cat_{chosen_r['label']}"
                        st.session_state["_vsrc"] = "catalogue"
                        st.session_state["_scan_label"] = chosen_r["label"]
                        st.session_state["_scan_age"] = chosen_r["age"]
                        st.session_state["_has_age"] = not np.isnan(chosen_r["age"])
                        st.session_state["_scan_sex"] = "Female" if chosen_r["sex"] == "F" else ("Male" if chosen_r["sex"] == "M" else "Unknown")
                        st.session_state.pop("_pred_age", None)
                        st.session_state.pop("_result", None)
                        st.session_state.pop("_elapsed", None)
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Could not load scan: {exc}")

        # Handle file upload event
        if uploaded_file is not None:
            file_id = f"{uploaded_file.name}_{uploaded_file.size}"
            if st.session_state.get("_loaded_file_id") != file_id:
                try:
                    uploaded_file.seek(0)
                    vol_loaded = parse_npz_bytes(uploaded_file.read())
                    stem = Path(uploaded_file.name).stem
                    # --- Auto-lookup metadata from CSV ---
                    meta = lookup_metadata_from_filename(stem)
                    st.session_state["_volume"] = vol_loaded
                    st.session_state["_loaded_file_id"] = file_id
                    st.session_state["_vsrc"] = "upload"
                    st.session_state["_scan_label"] = stem
                    if meta:
                        st.session_state["_scan_age"]  = meta["age"]
                        st.session_state["_has_age"]   = not np.isnan(meta["age"])
                        sex_raw = meta["sex"]
                        st.session_state["_scan_sex"] = (
                            "Female" if sex_raw == "F"
                            else ("Male" if sex_raw == "M" else "Unknown")
                        )
                    else:
                        # No match — leave defaults; user can fill manually
                        st.session_state.setdefault("_scan_age", float("nan"))
                        st.session_state.setdefault("_has_age", False)
                        st.session_state.setdefault("_scan_sex", "Unknown")
                    # Reset prediction state
                    st.session_state.pop("_pred_age", None)
                    st.session_state.pop("_result", None)
                    st.session_state.pop("_elapsed", None)
                except Exception as exc:
                    st.error(f"Error parsing uploaded NPZ: {exc}")
        elif st.session_state.get("_vsrc") == "upload" and uploaded_file is None:
            # User cleared uploader
            st.session_state.pop("_volume", None)
            st.session_state.pop("_loaded_file_id", None)
            st.session_state.pop("_vsrc", None)
            st.session_state.pop("_scan", None)
            st.session_state.pop("_scan_age", None)
            st.session_state.pop("_has_age", None)
            st.session_state.pop("_scan_sex", None)
            st.session_state.pop("_pred_age", None)
            st.session_state.pop("_result", None)
            st.session_state.pop("_elapsed", None)

        # Metadata inputs when volume is loaded
        if st.session_state.get("_volume") is not None:
            st.markdown("---")
            st.markdown(
                '<div class="section-header">Scan Metadata (Optional)</div>',
                unsafe_allow_html=True,
            )

            current_lbl = st.session_state.get("_scan_label", "Scan_01")
            scan_label = st.text_input(
                "Scan / Subject ID",
                value=current_lbl,
                key="input_scan_label",
            )
            st.session_state["_scan_label"] = scan_label

            c_age1, c_age2 = st.columns([1.1, 1.4])
            with c_age1:
                has_chron = st.checkbox(
                    "Known age",
                    value=st.session_state.get("_has_age", False),
                    help="Check to enter chronological age for Brain Age Gap (BAG) computation.",
                    key="cb_known_age",
                )
                st.session_state["_has_age"] = has_chron
            with c_age2:
                if has_chron:
                    default_age = float(st.session_state.get("_scan_age", 65.0))
                    if np.isnan(default_age):
                        default_age = 65.0
                    chron_val = st.number_input(
                        "Age (years)",
                        min_value=1.0,
                        max_value=120.0,
                        value=default_age,
                        step=0.5,
                        key="num_chron_age",
                    )
                    st.session_state["_scan_age"] = chron_val
                else:
                    chron_val = float("nan")

            sex_opts = ["Unknown", "Female", "Male"]
            prev_sex = st.session_state.get("_scan_sex", "Unknown")
            sex_idx = sex_opts.index(prev_sex) if prev_sex in sex_opts else 0
            sex_opt = st.selectbox(
                "Sex",
                sex_opts,
                index=sex_idx,
                key="sel_scan_sex",
            )
            st.session_state["_scan_sex"] = sex_opt
            sex_code = "F" if sex_opt == "Female" else ("M" if sex_opt == "Male" else "?")

            vsrc_label = {
                "upload": "User Upload",
                "synthetic": "Synthetic Demo",
                "catalogue": "Catalogue Disk Scan",
            }.get(st.session_state.get("_vsrc", ""), "User Input")

            st.session_state["_scan"] = {
                "label": scan_label,
                "subject_id": scan_label,
                "age": chron_val,
                "sex": sex_code,
                "fold": vsrc_label,
                "MMSE": float("nan"),
                "CDRTOT": float("nan"),
            }

            vol = st.session_state["_volume"]
            st.markdown("---")
            st.markdown(
                '<div class="section-header">Volume Summary</div>',
                unsafe_allow_html=True,
            )
            summary_items = [
                ("Subject / Label", scan_label),
                ("Chronological Age", f"{chron_val:.1f} y" if not np.isnan(chron_val) else "Not specified"),
                ("Sex", sex_opt),
                ("Dimensions", f"{vol.shape[0]} × {vol.shape[1]} × {vol.shape[2]}"),
                ("Intensity (z-score)", f"[{vol.min():.2f}, {vol.max():.2f}]"),
                ("Source", vsrc_label),
            ]
            for k, v in summary_items:
                st.markdown(
                    f'<div class="kv"><span>{k}</span><span>{v}</span></div>',
                    unsafe_allow_html=True,
                )

            if st.button("Unload / Clear Scan", key="btn_clear_scan"):
                st.session_state.pop("_volume", None)
                st.session_state.pop("_loaded_file_id", None)
                st.session_state.pop("_vsrc", None)
                st.session_state.pop("_scan", None)
                st.session_state.pop("_pred_age", None)
                st.session_state.pop("_result", None)
                st.session_state.pop("_elapsed", None)
                st.rerun()

    with col_viewer:
        volume = st.session_state.get("_volume", None)

        if volume is not None:
            st.markdown(
                '<div class="section-header">Orthogonal Slice Viewer</div>',
                unsafe_allow_html=True,
            )

            sx, sy, sz = volume.shape

            # Sliders in display order: Axial (Z) → Coronal (Y) → Sagittal (X)
            c1, c2, c3 = st.columns(3)
            axi_idx = c1.slider("Axial (Z)",    0, sz - 1, sz // 2, key="axi")
            cor_idx = c2.slider("Coronal (Y)",  0, sy - 1, sy // 2, key="cor")
            sag_idx = c3.slider("Sagittal (X)", 0, sx - 1, sx // 2, key="sag")

            fig = plt.figure(figsize=(13, 4.4), constrained_layout=True, facecolor=CARD_BG)
            gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.06)

            view_data = [
                ("Axial",    np.rot90(volume[:, :, axi_idx]), f"z={axi_idx}"),
                ("Coronal",  np.rot90(volume[:, cor_idx, :]), f"y={cor_idx}"),
                ("Sagittal", np.rot90(volume[sag_idx, :, :]), f"x={sag_idx}"),
            ]
            for i, (title, slc, coord) in enumerate(view_data):
                ax_v = fig.add_subplot(gs[i])
                ax_v.set_facecolor(CARD_BG)
                ax_v.imshow(slc, cmap="gray", interpolation="bilinear", aspect="auto")
                ax_v.set_title(
                    f"{title}  ({coord})",
                    color=INK, fontsize=10, fontweight="medium", pad=6,
                )
                ax_v.axis("off")

            st.pyplot(fig, clear_figure=True)

            # Intensity profile
            with st.expander(
                "Intensity profile (centre row, axial slice)", expanded=False
            ):
                mid_row = volume[:, sy // 2, axi_idx]
                fig2, ax2 = styled_fig(11, 2.4)
                ax2.plot(mid_row, color=ACCENT, linewidth=1.2)
                ax2.fill_between(
                    range(len(mid_row)), mid_row, alpha=0.15, color=ACCENT
                )
                ax2.set_xlabel("Voxel index (X)")
                ax2.set_ylabel("Intensity (z-score)")
                ax2.grid(True, color=GRID_CLR, alpha=0.6)
                fig2.tight_layout()
                st.pyplot(fig2, clear_figure=True)

            active_lbl = st.session_state["_scan"]["label"]
            st.info(
                f"Scan **{active_lbl}** is loaded and ready. "
                "Navigate to **Layer 2 · Prediction** to run brain-age inference."
            )
        else:
            st.markdown(
                '<div class="section-header">Orthogonal Slice Viewer</div>',
                unsafe_allow_html=True,
            )
            st.markdown("""
<div class="empty-state">
  <h3>Awaiting Preprocessed MRI Scan</h3>
  <p>Upload a preprocessed 3D T1-weighted volume in <code>.npz</code> format using the panel on the left to inspect orthogonal slices and estimate brain age.</p>
</div>

<div class="spec-box">
  <b>Preprocessed Input Requirements:</b>
  <ul style="margin: 0.4rem 0 0.2rem 1.2rem; padding: 0;">
    <li><b>File Format:</b> NumPy <code>.npz</code> archive</li>
    <li><b>Array Key:</b> <code>data</code> (or single root array)</li>
    <li><b>Shape:</b> Exactly <code>(160, 192, 160)</code> voxels</li>
    <li><b>Registration:</b> Affine registered to MNI152 (1 mm)</li>
    <li><b>Normalization:</b> Skull-stripped, brain-only intensity z-scored</li>
  </ul>
</div>
""", unsafe_allow_html=True)



# ===========================================================================
# LAYER 2 — Real-time Brain Age Prediction
# ===========================================================================
with tab2:
    st.markdown(
        '<div class="layer-badge">Layer 2 — Real-time Brain Age Prediction</div>',
        unsafe_allow_html=True,
    )

    col_info2, col_pred2 = st.columns([1.3, 2], gap="large")

    with col_info2:
        st.markdown(
            '<div class="section-header">Model Info</div>',
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
            '<div class="section-header" style="margin-top:1.4rem;">Active Scan</div>',
            unsafe_allow_html=True,
        )
        if "_scan" in st.session_state and "_volume" in st.session_state and st.session_state["_volume"] is not None:
            s = st.session_state["_scan"]
            age_disp = f"{fmt(s['age'])} y" if not np.isnan(s.get("age", float("nan"))) else "Not specified"
            st.markdown(
                f'<div class="info-box">'
                f'<b>Label:</b> {s["label"]}<br>'
                f'<b>Subject:</b> {s["subject_id"]}<br>'
                f'<b>Age:</b> {age_disp}<br>'
                f'<b>Sex:</b> {s["sex"]}<br>'
                f'<b>Source:</b> {s.get("fold", "User Upload")}'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("Go to **Layer 1** and upload a scan first.")

    with col_pred2:
        st.markdown(
            '<div class="section-header">Run Inference</div>',
            unsafe_allow_html=True,
        )

        volume_ready = (
            "_volume" in st.session_state
            and st.session_state["_volume"] is not None
        )

        if not volume_ready:
            st.warning("No volume loaded — please upload a scan in **Layer 1** first.")
        else:
            st.markdown(
                f'<p style="color:{MUTED};font-size:.88rem;">Volume shape: '
                f'<code>{st.session_state["_volume"].shape}</code></p>',
                unsafe_allow_html=True,
            )

        predict_btn = st.button(
            "Predict Brain Age",
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

            st.success("Inference complete")

            # Determine how many metric columns to show
            chron_for_l2 = float("nan")
            if "_scan" in st.session_state:
                chron_for_l2 = st.session_state["_scan"].get("age", float("nan"))

            if not np.isnan(chron_for_l2):
                bag_for_l2 = pred_age - chron_for_l2
                bag_sign   = "+" if bag_for_l2 >= 0 else ""
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("Predicted Brain Age",  f"{pred_age:.1f} y")
                r2.metric("Chronological Age",    f"{chron_for_l2:.1f} y")
                r3.metric("Inference Latency",    f"{elapsed:.2f} s")
                r4.metric("Checkpoint",           result.get("checkpoint", "m1_f0_s0"))
                bag_col_l2 = POS_CLR if bag_for_l2 > 0.5 else (NEG_CLR if bag_for_l2 < -0.5 else AMBER)
                st.markdown(
                    f'<div style="margin-top:.5rem;padding:.5rem .9rem;background:{INFO_BG};'
                    f'border-left:3px solid {bag_col_l2};border-radius:0 6px 6px 0;'
                    f'font-size:.88rem;color:{INK};">'
                    f'<b>Brain Age Gap (BAG):</b> {bag_sign}{bag_for_l2:.2f} y &nbsp;·&nbsp; '
                    f'Abs. error: <b>{abs(bag_for_l2):.2f} y</b></div>',
                    unsafe_allow_html=True,
                )
            else:
                r1, r2, r3 = st.columns(3)
                r1.metric("Predicted Brain Age", f"{pred_age:.1f} y")
                r2.metric("Inference Latency",   f"{elapsed:.2f} s")
                r3.metric("Checkpoint",          result.get("checkpoint", "m1_f0_s0"))

            # Illustrative probability distribution
            st.markdown(
                '<div class="section-header" style="margin-top:1.2rem;">'
                'Model Confidence Visualization</div>',
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
                width=bin_step * 0.85, color=ACCENT, alpha=0.5,
                label="Probability mass",
            )
            ax3.axvline(
                pred_age, color=POS_CLR, linewidth=1.8, linestyle="--",
                label=f"Predicted: {pred_age:.1f} y",
            )
            if "_scan" in st.session_state:
                chron = st.session_state["_scan"].get("age", float("nan"))
                if not np.isnan(chron):
                    ax3.axvline(
                        chron, color=NEG_CLR, linewidth=1.8, linestyle=":",
                        label=f"Chronological: {chron:.1f} y",
                    )
            ax3.set_xlabel("Age (years)")
            ax3.set_ylabel("Probability mass")
            ax3.legend(frameon=False, fontsize=9, labelcolor=INK)
            ax3.grid(True, color=GRID_CLR, alpha=0.6)
            fig3.tight_layout()
            st.pyplot(fig3, clear_figure=True)

            st.caption(
                "The distribution above is an illustrative Gaussian centred on the model's "
                "predicted expected age. The actual SFCN head outputs 64 discrete softmax bins."
            )

            st.session_state["_pred_age"] = pred_age


# ===========================================================================
# LAYER 3 — Brain Age Gap & Error Analysis
# ===========================================================================
with tab3:
    st.markdown(
        '<div class="layer-badge">Layer 3 — Brain Age Gap &amp; Error Analysis</div>',
        unsafe_allow_html=True,
    )

    left3, right3 = st.columns([1.3, 2], gap="large")

    with left3:
        st.markdown(
            '<div class="section-header">BAG for Active Scan</div>',
            unsafe_allow_html=True,
        )

        has_pred = "_pred_age" in st.session_state
        has_scan = (
            "_scan" in st.session_state
            and "_volume" in st.session_state
            and st.session_state["_volume"] is not None
        )

        if has_pred and has_scan:
            pred_age = st.session_state["_pred_age"]
            scan_s   = st.session_state["_scan"]
            chron    = scan_s.get("age", float("nan"))

            if not np.isnan(chron):
                bag_val  = pred_age - chron
                abs_err  = abs(bag_val)
                bag_sign = "+" if bag_val >= 0 else ""

                if bag_val > 0.5:
                    bag_col = interp_col = POS_CLR
                    interp_text = (
                        "Predicted older than chronological age — brain may appear "
                        "more aged than peers of the same age."
                    )
                elif bag_val < -0.5:
                    bag_col = interp_col = NEG_CLR
                    interp_text = (
                        "Predicted younger than chronological age — brain may appear "
                        "less aged than peers of the same age."
                    )
                else:
                    bag_col = interp_col = AMBER
                    interp_text = (
                        "Predicted age is close to chronological age — within "
                        "expected model variation."
                    )

                st.markdown(
                    f'<div class="bag-box">'
                    f'<div class="lbl">Brain Age Gap (BAG)</div>'
                    f'<div class="val" style="color:{bag_col};">'
                    f'{bag_sign}{bag_val:.2f} years</div>'
                    f'<div class="sub">Absolute error: '
                    f'<b style="color:{INK};">{abs_err:.2f} y</b></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                c1, c2 = st.columns(2)
                c1.metric("Chronological Age", f"{chron:.1f} y")
                c2.metric("Predicted Age",     f"{pred_age:.1f} y")

                st.markdown(
                    f'<div class="info-box" '
                    f'style="border-left-color:{interp_col};margin-top:.8rem;">'
                    f'{interp_text}</div>',
                    unsafe_allow_html=True,
                )

                st.warning(
                    "**Disclaimer**: BAG is a model-derived research estimate, "
                    "not a clinical diagnosis. Positive or negative gaps reflect model "
                    "uncertainty and training distribution effects — not brain pathology."
                )
            else:
                st.metric("Predicted Brain Age", f"{pred_age:.1f} y")
                st.info(
                    "Chronological age was not provided for this scan. "
                    "Brain Age Gap (BAG = Predicted − Chronological) cannot be calculated. "
                    "To view BAG analysis, check 'Known age' and specify chronological age in **Layer 1**."
                )
        elif has_scan:
            st.info(
                "Scan is loaded. Complete **Layer 2** (run prediction) to see the Brain Age Gap "
                "for the active scan."
            )
        else:
            st.info(
                "Upload a scan in **Layer 1** and run prediction in **Layer 2** to see the BAG "
                "for the active scan."
            )

    with right3:
        st.markdown(
            '<div class="section-header">Compare Multiple Scans</div>',
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

        # If user has an active scan with prediction and chronological age, include it
        has_active_bag = (
            "_pred_age" in st.session_state
            and "_scan" in st.session_state
            and not np.isnan(st.session_state["_scan"].get("age", float("nan")))
        )
        if has_active_bag:
            s_act = st.session_state["_scan"]
            p_act = st.session_state["_pred_age"]
            bag_act = p_act - s_act["age"]
            active_row = pd.DataFrame([{
                "mri_label": f"{s_act['label']} (Active)",
                "subject_id": s_act["subject_id"],
                "age": s_act["age"],
                "predicted_age": p_act,
                "BAG_computed": bag_act,
                "abs_error": abs(bag_act),
            }])
            demo_with_bag = pd.concat([active_row, demo_with_bag], ignore_index=True)

        subj_options = demo_with_bag["mri_label"].tolist()
        if has_active_bag:
            default_sel = [subj_options[0]] + subj_options[1: min(6, len(subj_options))]
        else:
            default_sel = subj_options[: min(6, len(subj_options))]

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

            ax4.bar(xs, bag_arr, color=colors4, alpha=0.9, width=0.6)
            ax4.axhline(0, color=INK, linewidth=0.8)
            ax4.set_xticks(xs)
            ax4.set_xticklabels(
                labels4, rotation=40, ha="right", fontsize=7.5
            )
            ax4.set_ylabel("Brain Age Gap (years)")
            ax4.set_title(
                "BAG per Scan  (red = older-looking, green = younger-looking)",
                fontsize=10, loc="left", color=INK,
            )
            ax4.grid(axis="y", color=GRID_CLR)
            ax4.set_axisbelow(True)
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
        '<div class="layer-badge">Layer 4 — Research Performance Explorer</div>',
        unsafe_allow_html=True,
    )

    # Key metrics
    st.markdown(
        '<div class="section-header">'
        'V3 Evaluation Metrics (held-out cross-validation)</div>',
        unsafe_allow_html=True,
    )

    # Two rows (4 + 3) so labels stay legible on laptop-width screens.
    mrow1 = st.columns(4)
    mrow2 = st.columns(4)
    m1c, m2c, m3c, m4c = mrow1
    m5c, m6c, m7c      = mrow2[:3]
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
            '<div class="section-header">Predicted vs Chronological Age</div>',
            unsafe_allow_html=True,
        )
        fig5, ax5 = styled_fig(6.4, 4.6)
        sc = ax5.scatter(
            plot_df["age"], plot_df["predicted_age"],
            s=8, alpha=0.5,
            c=err_vals.abs(), cmap=SEQ_CMAP, vmin=0, vmax=12,
        )
        lo4 = float(min(plot_df["age"].min(), plot_df["predicted_age"].min()))
        hi4 = float(max(plot_df["age"].max(), plot_df["predicted_age"].max()))
        ax5.plot(
            [lo4, hi4], [lo4, hi4], "--",
            color=MUTED, linewidth=1.1, label="Perfect prediction",
        )
        cb = fig5.colorbar(sc, ax=ax5, shrink=0.8, pad=0.02)
        cb.set_label("|Error| (years)", color=MUTED, fontsize=8)
        cb.outline.set_visible(False)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=MUTED, fontsize=7.5)

        if "_pred_age" in st.session_state and "_scan" in st.session_state:
            s4  = st.session_state["_scan"]
            pa4 = st.session_state["_pred_age"]
            s4_age = s4.get("age", float("nan"))
            if s4_age is not None and not np.isnan(s4_age):
                ax5.scatter(
                    [s4_age], [pa4], s=120, color=POS_CLR,
                    zorder=5, marker="*", label=f"Active: {s4['label']}",
                )

        ax5.legend(frameon=False, fontsize=8.5, labelcolor=INK)
        ax5.set_xlabel("Chronological Age (years)")
        ax5.set_ylabel("Predicted Age (years)")
        ax5.grid(True, color=GRID_CLR)
        ax5.set_axisbelow(True)
        fig5.tight_layout()
        st.pyplot(fig5, clear_figure=True)

    # Error distribution
    with rcol4:
        st.markdown(
            '<div class="section-header">Prediction Error Distribution</div>',
            unsafe_allow_html=True,
        )
        fig6, ax6 = styled_fig(6.4, 4.6)
        ax6.hist(err_vals, bins=40, color=ACCENT, alpha=0.75,
                 edgecolor=CARD_BG, linewidth=0.4)
        ax6.axvline(0, color=INK, linewidth=1.2, linestyle="--", label="Zero error")
        ax6.axvline(
            err_vals.mean(), color=POS_CLR, linewidth=1.4, linestyle=":",
            label=f"Mean: {err_vals.mean():.2f} y",
        )
        ax6.set_xlabel("Predicted − Chronological Age (years)")
        ax6.set_ylabel("Scan count")
        ax6.legend(frameon=False, fontsize=8.5, labelcolor=INK)
        ax6.grid(axis="y", color=GRID_CLR)
        ax6.set_axisbelow(True)
        fig6.tight_layout()
        st.pyplot(fig6, clear_figure=True)

    st.markdown("---")
    b1col, b2col = st.columns(2, gap="medium")

    # MAE by age group
    with b1col:
        st.markdown(
            '<div class="section-header">MAE by Age Group</div>',
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
            fig7, ax7 = styled_fig(6.4, 4.2)
            bars7 = ax7.bar(
                grp_df4["Age Group"], grp_df4["MAE (years)"],
                color=ACCENT, alpha=0.8, width=0.6,
            )
            for bar7, val7 in zip(bars7, grp_df4["MAE (years)"]):
                ax7.text(
                    bar7.get_x() + bar7.get_width() / 2,
                    bar7.get_height() + 0.05,
                    f"{val7:.2f}", ha="center", va="bottom",
                    fontsize=8, color=INK,
                )
            ax7.axhline(
                metrics["MAE"], color=POS_CLR, linewidth=1.2, linestyle="--",
                label=f"Overall MAE: {metrics['MAE']:.2f} y",
            )
            ax7.legend(frameon=True, facecolor=CARD_BG, edgecolor=RULE,
                       framealpha=0.9, fontsize=8.5, loc="upper right", labelcolor=INK)
            ax7.set_xlabel("Age Group")
            ax7.set_ylabel("MAE (years)")
            ax7.grid(axis="y", color=GRID_CLR)
            ax7.set_axisbelow(True)
            fig7.tight_layout()
            st.pyplot(fig7, clear_figure=True)
            st.dataframe(grp_df4.round(3), use_container_width=True, hide_index=True)

    # Training curve + BAG vs age
    with b2col:
        st.markdown(
            '<div class="section-header">Training Loss Curve</div>',
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
            ax8r.spines[["top"]].set_visible(False)
            l1h, = ax8.plot(
                epochs8, train_loss8, color=ACCENT, linewidth=1.8, label="Train loss"
            )
            l2h, = ax8r.plot(
                epochs8, val_mae8, color=ACCENT2,
                linewidth=1.8, linestyle="--", label="Val MAE",
            )
            if best_val8 is not None:
                ax8r.axhline(
                    best_val8, color=POS_CLR, linewidth=1.1, linestyle=":",
                    label=f"Best val MAE: {best_val8:.2f} y",
                )
            ax8.set_xlabel("Epoch")
            ax8.set_ylabel("Train Loss (KL-div)", color=ACCENT)
            ax8r.set_ylabel("Val MAE (years)", color=ACCENT2)
            ax8r.spines["right"].set_color(RULE)
            ax8r.tick_params(colors=MUTED)
            ax8r.yaxis.label.set_color(ACCENT2)
            ax8.yaxis.label.set_color(ACCENT)
            leg_lines  = [l1h, l2h]
            leg_labels = ["Train loss", "Val MAE"]
            if best_val8 is not None:
                leg_labels.append(f"Best: {best_val8:.2f} y")
            ax8.legend(
                leg_lines, leg_labels,
                frameon=False, fontsize=8.5, loc="upper right",
                labelcolor=INK,
            )
            ax8.grid(True, color=GRID_CLR, alpha=0.6)
            fig8.tight_layout()
            st.pyplot(fig8, clear_figure=True)

        st.markdown(
            '<div class="section-header" style="margin-top:1.2rem;">'
            'BAG vs Chronological Age</div>',
            unsafe_allow_html=True,
        )
        bag_plot9 = raw.dropna(subset=["age", "predicted_age"]).copy()
        bag_plot9["bag_v"] = bag_plot9["predicted_age"] - bag_plot9["age"]
        if len(bag_plot9) > 2000:
            bag_plot9 = bag_plot9.sample(2000, random_state=0)

        fig9, ax9 = styled_fig(6.4, 3.6)
        ax9.scatter(
            bag_plot9["age"], bag_plot9["bag_v"],
            s=7, alpha=0.5,
            c=bag_plot9["bag_v"].abs(), cmap=SEQ_CMAP, vmin=0, vmax=10,
        )
        ax9.axhline(0, color=INK, linewidth=1.0, linestyle="--")
        ax9.set_xlabel("Chronological Age (years)")
        ax9.set_ylabel("BAG (Predicted − Actual)")
        ax9.grid(True, color=GRID_CLR)
        ax9.set_axisbelow(True)
        st.caption(
            f"Slope = {metrics['BAG_vs_age_slope']:.4f}  |  "
            f"p = {metrics['BAG_vs_age_p']:.3f}  |  "
            f"Mean BAG (CN) = {metrics['mean_BAG_CN']:.3f} y"
        )
        fig9.tight_layout()
        st.pyplot(fig9, clear_figure=True)

    st.markdown("---")

    with st.expander("Full metrics JSON", expanded=False):
        st.json(metrics)

    with st.expander(
        "Explore full prediction table (V3 cross-val)", expanded=False
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

    with st.expander("Methodology & Provenance", expanded=False):
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
