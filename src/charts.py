"""Consistent, restrained charts for research reporting."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

NAVY = "#315A7D"
GREY = "#667085"
GRID = "#E4E7EC"


def style_axes(ax):
    ax.set_facecolor("white")
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D0D5DD")
    ax.spines["bottom"].set_color("#D0D5DD")
    ax.tick_params(colors=GREY, labelsize=9)
    ax.xaxis.label.set_color(GREY)
    ax.yaxis.label.set_color(GREY)
    ax.title.set_color("#344054")
    ax.title.set_fontsize(11)


def predicted_vs_actual(frame):
    df = frame.dropna(subset=["age", "predicted_age"]).copy()
    if len(df) > 3500:
        df = df.sample(3500, random_state=42)
    fig, ax = plt.subplots(figsize=(6.2, 4.3), dpi=120)
    ax.scatter(df["age"], df["predicted_age"], s=13, alpha=0.42, color=NAVY, edgecolors="none", rasterized=True)
    if not df.empty:
        lo = float(min(df["age"].min(), df["predicted_age"].min()))
        hi = float(max(df["age"].max(), df["predicted_age"].max()))
        ax.plot([lo, hi], [lo, hi], color="#98A2B3", linestyle="--", linewidth=1.3, label="Ideal estimate")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set(title="Predicted age against chronological age", xlabel="Chronological age (years)", ylabel="Predicted age (years)")
    style_axes(ax); fig.tight_layout()
    return fig


def error_distribution(frame):
    df = frame.dropna(subset=["age", "predicted_age"])
    error = df["predicted_age"] - df["age"]
    fig, ax = plt.subplots(figsize=(6.2, 4.3), dpi=120)
    ax.hist(error, bins=32, color=NAVY, alpha=0.86, edgecolor="white", linewidth=0.45)
    ax.axvline(0, color="#667085", linestyle="--", linewidth=1.2)
    ax.set(title="Distribution of prediction errors", xlabel="Predicted age − chronological age (years)", ylabel="Number of scans")
    style_axes(ax); fig.tight_layout()
    return fig


def age_group_mae(frame):
    bands = [(0, 60, "Under 60"), (60, 70, "60–69"), (70, 80, "70–79"), (80, 130, "80+")]
    rows = []
    for low, high, label in bands:
        part = frame[(frame["age"] >= low) & (frame["age"] < high)].dropna(subset=["age", "predicted_age"])
        if len(part):
            rows.append((label, float(np.abs(part["predicted_age"] - part["age"]).mean()), len(part)))
    fig, ax = plt.subplots(figsize=(8.0, 3.6), dpi=120)
    if rows:
        labels, values, counts = zip(*rows)
        bars = ax.bar(labels, values, color=NAVY, width=0.62)
        for bar, count in zip(bars, counts):
            ax.annotate(f"n={count:,}", (bar.get_x() + bar.get_width()/2, bar.get_height()), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8, color=GREY)
    ax.set(title="Mean absolute error by chronological-age group", xlabel="Age group", ylabel="MAE (years)")
    style_axes(ax); fig.tight_layout()
    return fig


def residual_vs_age(frame):
    df = frame.dropna(subset=["age", "predicted_age"])
    if len(df) > 3500:
        df = df.sample(3500, random_state=42)
    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=120)
    ax.scatter(df["age"], df["predicted_age"] - df["age"], s=12, alpha=0.4, color=NAVY, edgecolors="none", rasterized=True)
    ax.axhline(0, color="#667085", linestyle="--", linewidth=1.2)
    ax.set(title="Prediction residuals across age", xlabel="Chronological age (years)", ylabel="Prediction error (years)")
    style_axes(ax); fig.tight_layout()
    return fig
