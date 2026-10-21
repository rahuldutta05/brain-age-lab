"""Consistent, restrained charts for research reporting with Dark Mode support."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


def get_theme_colors():
    """Retrieve theme-aware colors based on active session state."""
    is_dark = True
    try:
        is_dark = st.session_state.get("dark_mode", True)
    except Exception:
        is_dark = True

    if is_dark:
        return {
            "bg": "#131F30",
            "accent": "#38BDF8",
            "grey": "#94A3B8",
            "grid": "#1E293B",
            "spine": "#334155",
            "title": "#F1F5F9",
            "line": "#94A3B8",
            "hist_edge": "#131F30",
        }
    else:
        return {
            "bg": "white",
            "accent": "#315A7D",
            "grey": "#667085",
            "grid": "#E4E7EC",
            "spine": "#D0D5DD",
            "title": "#344054",
            "line": "#98A2B3",
            "hist_edge": "white",
        }


def style_axes(ax, theme=None):
    if theme is None:
        theme = get_theme_colors()
    ax.set_facecolor(theme["bg"])
    ax.grid(True, color=theme["grid"], linewidth=0.7, alpha=0.85)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(theme["spine"])
    ax.spines["bottom"].set_color(theme["spine"])
    ax.tick_params(colors=theme["grey"], labelsize=9)
    ax.xaxis.label.set_color(theme["grey"])
    ax.yaxis.label.set_color(theme["grey"])
    ax.title.set_color(theme["title"])
    ax.title.set_fontsize(11)


def predicted_vs_actual(frame):
    theme = get_theme_colors()
    df = frame.dropna(subset=["age", "predicted_age"]).copy()
    if len(df) > 3500:
        df = df.sample(3500, random_state=42)
    fig, ax = plt.subplots(figsize=(6.2, 4.3), dpi=120, facecolor=theme["bg"])
    ax.scatter(df["age"], df["predicted_age"], s=13, alpha=0.5, color=theme["accent"], edgecolors="none", rasterized=True)
    if not df.empty:
        lo = float(min(df["age"].min(), df["predicted_age"].min()))
        hi = float(max(df["age"].max(), df["predicted_age"].max()))
        ax.plot([lo, hi], [lo, hi], color=theme["line"], linestyle="--", linewidth=1.3, label="Ideal estimate")
        ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=theme["title"])
    ax.set(title="Predicted age against chronological age", xlabel="Chronological age (years)", ylabel="Predicted age (years)")
    style_axes(ax, theme); fig.tight_layout()
    return fig


def error_distribution(frame):
    theme = get_theme_colors()
    df = frame.dropna(subset=["age", "predicted_age"])
    error = df["predicted_age"] - df["age"]
    fig, ax = plt.subplots(figsize=(6.2, 4.3), dpi=120, facecolor=theme["bg"])
    ax.hist(error, bins=32, color=theme["accent"], alpha=0.85, edgecolor=theme["hist_edge"], linewidth=0.45)
    ax.axvline(0, color=theme["line"], linestyle="--", linewidth=1.2)
    ax.set(title="Distribution of prediction errors", xlabel="Predicted age − chronological age (years)", ylabel="Number of scans")
    style_axes(ax, theme); fig.tight_layout()
    return fig


def age_group_mae(frame):
    theme = get_theme_colors()
    bands = [(0, 60, "Under 60"), (60, 70, "60–69"), (70, 80, "70–79"), (80, 130, "80+")]
    rows = []
    for low, high, label in bands:
        part = frame[(frame["age"] >= low) & (frame["age"] < high)].dropna(subset=["age", "predicted_age"])
        if len(part):
            rows.append((label, float(np.abs(part["predicted_age"] - part["age"]).mean()), len(part)))
    fig, ax = plt.subplots(figsize=(8.0, 3.6), dpi=120, facecolor=theme["bg"])
    if rows:
        labels, values, counts = zip(*rows)
        bars = ax.bar(labels, values, color=theme["accent"], width=0.62)
        for bar, count in zip(bars, counts):
            ax.annotate(f"n={count:,}", (bar.get_x() + bar.get_width()/2, bar.get_height()), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8, color=theme["grey"])
    ax.set(title="Mean absolute error by chronological-age group", xlabel="Age group", ylabel="MAE (years)")
    style_axes(ax, theme); fig.tight_layout()
    return fig


def residual_vs_age(frame):
    theme = get_theme_colors()
    df = frame.dropna(subset=["age", "predicted_age"])
    if len(df) > 3500:
        df = df.sample(3500, random_state=42)
    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=120, facecolor=theme["bg"])
    ax.scatter(df["age"], df["predicted_age"] - df["age"], s=12, alpha=0.45, color=theme["accent"], edgecolors="none", rasterized=True)
    ax.axhline(0, color=theme["line"], linestyle="--", linewidth=1.2)
    ax.set(title="Prediction residuals across age", xlabel="Chronological age (years)", ylabel="Prediction error (years)")
    style_axes(ax, theme); fig.tight_layout()
    return fig
