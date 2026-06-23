import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.signal import find_peaks, savgol_filter

st.set_page_config(page_title="Raman Compound Detector", layout="wide")

MAX_SHIFT_DEFAULT = 550.0
TARGET_POINTS = 1024


def read_spectrum(uploaded_file):
    text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
    rows = []
    for line in text.splitlines():
        parts = line.replace(",", " ").split()
        if len(parts) >= 2:
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    if not rows:
        raise ValueError("No numeric two-column data found in file.")
    df = pd.DataFrame(rows, columns=["raman_shift", "intensity"])
    df = df.dropna().drop_duplicates().sort_values("raman_shift").reset_index(drop=True)
    return df


def preprocess(df, max_shift=MAX_SHIFT_DEFAULT, target_points=TARGET_POINTS):
    roi = df[(df["raman_shift"] >= 0) & (df["raman_shift"] < max_shift)].copy()
    if len(roi) < 20:
        raise ValueError("Not enough points in Raman shift < 550 region.")
    x = roi["raman_shift"].to_numpy()
    y = roi["intensity"].to_numpy()

    x_new = np.linspace(x.min(), x.max(), target_points)
    y_new = np.interp(x_new, x, y)

    window = 15
    if window >= len(y_new):
        window = len(y_new) - 1 if len(y_new) % 2 == 0 else len(y_new)
    if window < 5:
        window = 5
    if window % 2 == 0:
        window += 1

    y_smooth = savgol_filter(y_new, window_length=window, polyorder=3)
    baseline = pd.Series(y_smooth).rolling(101, center=True, min_periods=1).quantile(0.1).to_numpy()
    y_corr = y_smooth - baseline
    y_norm = (y_corr - y_corr.mean()) / (y_corr.std() + 1e-8)

    return pd.DataFrame({
        "raman_shift": x_new,
        "intensity_interp": y_new,
        "smooth": y_smooth,
        "baseline": baseline,
        "corrected": y_corr,
        "normalized": y_norm
    })


def detect_peaks(proc_df, prominence=0.7, distance=8):
    x = proc_df["raman_shift"].to_numpy()
    y = proc_df["normalized"].to_numpy()
    idx, props = find_peaks(y, prominence=prominence, distance=distance)

    peaks = pd.DataFrame({
        "peak_shift": x[idx],
        "peak_height_norm": y[idx],
        "prominence": props.get("prominences", np.zeros(len(idx)))
    }).sort_values("prominence", ascending=False).reset_index(drop=True)
    return peaks


def simple_compound_match(peaks_df):
    peak_positions = peaks_df["peak_shift"].round(1).tolist()

    reference_db = {
        "Sulfur-like": [82, 150, 220, 440, 473],
        "Calcite-like": [108, 156, 281],
        "Gypsum-like": [100, 113, 414, 492],
        "Quartz-like": [128, 206, 265, 355, 465],
        "Graphite-like": [135]
    }

    tolerance = 12
    scores = {}

    for compound, refs in reference_db.items():
        score = 0
        for r in refs:
            for p in peak_positions:
                if abs(p - r) <= tolerance:
                    score += 1
                    break
        scores[compound] = score / max(len(refs), 1)

    best_compound = max(scores, key=scores.get)
    confidence = scores[best_compound]

    return {
        "predicted_compound": best_compound,
        "confidence": round(float(confidence), 3),
        "all_scores": scores
    }


def plot_spectrum(proc_df, peaks_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=proc_df["raman_shift"],
        y=proc_df["intensity_interp"],
        mode="lines",
        name="Interpolated raw"
    ))
    fig.add_trace(go.Scatter(
        x=proc_df["raman_shift"],
        y=proc_df["smooth"],
        mode="lines",
        name="Smoothed"
    ))
    fig.add_trace(go.Scatter(
        x=proc_df["raman_shift"],
        y=proc_df["baseline"],
        mode="lines",
        name="Baseline"
    ))
    fig.add_trace(go.Scatter(
        x=proc_df["raman_shift"],
        y=proc_df["normalized"],
        mode="lines",
        name="Normalized corrected"
    ))

    if not peaks_df.empty:
        fig.add_trace(go.Scatter(
            x=peaks_df["peak_shift"],
            y=peaks_df["peak_height_norm"],
            mode="markers",
            name="Detected peaks",
            marker=dict(size=9, color="red")
        ))

    fig.update_layout(
        height=520,
        xaxis_title="Raman shift (cm^-1)",
        yaxis_title="Intensity / normalized intensity",
        title="Raman Spectrum Analysis"
    )
    return fig


st.title("Raman Compound Detector")
st.write("Upload a Raman TXT or CSV file. The app automatically preprocesses the data, detects peaks below 550 cm^-1, and predicts the most likely compound pattern.")

uploaded = st.file_uploader("Upload Raman TXT/CSV file", type=["txt", "csv"])

if uploaded is not None:
    try:
        raw_df = read_spectrum(uploaded)
        proc_df = preprocess(raw_df, max_shift=550.0)
        peaks_df = detect_peaks(proc_df, prominence=0.7, distance=8)
        result = simple_compound_match(peaks_df)

        c1, c2, c3 = st.columns(3)
        c1.metric("Raw points", len(raw_df))
        c2.metric("ROI points (<550)", len(proc_df))
        c3.metric("Detected peaks", len(peaks_df))

        st.plotly_chart(plot_spectrum(proc_df, peaks_df), use_container_width=True)

        left, right = st.columns([1.3, 1])

        with left:
            st.subheader("Detected peaks")
            st.dataframe(peaks_df, use_container_width=True)

        with right:
            st.subheader("Predicted compound")
            st.success(f"Most likely: {result['predicted_compound']}")
            st.write(f"Confidence score: {result['confidence']}")
            st.json(result["all_scores"])

        export = {
            "predicted_compound": result["predicted_compound"],
            "confidence": result["confidence"],
            "detected_peak_count": int(len(peaks_df)),
            "peak_positions_cm-1": peaks_df["peak_shift"].round(3).tolist()
        }

        st.download_button(
            "Download results JSON",
            data=json.dumps(export, indent=2),
            file_name="raman_result.json",
            mime="application/json"
        )

        st.download_button(
            "Download peaks CSV",
            data=peaks_df.to_csv(index=False),
            file_name="raman_peaks.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    st.markdown("""
### Expected file format
Two numeric columns:

- Column 1: Raman shift
- Column 2: Intensity

Example:
```text
0.72 481.9
2.07 479.9
3.42 479.8
```
""")
