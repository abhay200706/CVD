import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter

st.set_page_config(page_title="Raman Peak Detector", layout="centered")

ROI_MIN = 50.0
ROI_MAX = 490.0
TARGET_POINTS = 1024


# ---------------------------------------------------------
# STEP 1: Read raw spectrum file
# ---------------------------------------------------------
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
    df = pd.DataFrame(rows, columns=["raman_shift", "intensity"])
    df = df.dropna().drop_duplicates().sort_values("raman_shift").reset_index(drop=True)
    return df


# ---------------------------------------------------------
# STEP 2: Rayleigh correction (remove strong low-shift tail)
# ---------------------------------------------------------
def rayleigh_correction(df):
    x = df["raman_shift"].to_numpy()
    y = df["intensity"].to_numpy()

    # estimate the Rayleigh tail using a rolling minimum at the low end
    tail_mask = x < ROI_MIN
    if tail_mask.sum() > 5:
        tail_level = np.percentile(y[tail_mask], 10)
    else:
        tail_level = np.percentile(y, 1)

    y_corrected = y - tail_level
    y_corrected[y_corrected < 0] = 0
    return pd.DataFrame({"raman_shift": x, "intensity": y_corrected})


# ---------------------------------------------------------
# STEP 3: Restrict to region of interest (50-490 cm-1)
# ---------------------------------------------------------
def select_roi(df, roi_min=ROI_MIN, roi_max=ROI_MAX):
    roi = df[(df["raman_shift"] >= roi_min) & (df["raman_shift"] <= roi_max)].copy()
    return roi.reset_index(drop=True)


# ---------------------------------------------------------
# STEP 4: Interpolate to a uniform grid + smooth + baseline
# ---------------------------------------------------------
def preprocess(roi_df, target_points=TARGET_POINTS):
    x = roi_df["raman_shift"].to_numpy()
    y = roi_df["intensity"].to_numpy()

    x_new = np.linspace(x.min(), x.max(), target_points)
    y_new = np.interp(x_new, x, y)

    window = 15 if target_points > 15 else 5
    if window % 2 == 0:
        window += 1
    y_smooth = savgol_filter(y_new, window_length=window, polyorder=3)

    baseline = pd.Series(y_smooth).rolling(101, center=True, min_periods=1).quantile(0.1).to_numpy()
    y_corr = y_smooth - baseline

    return pd.DataFrame({"raman_shift": x_new, "intensity": y_corr})


# ---------------------------------------------------------
# STEP 5: Detect candidate peaks
# ---------------------------------------------------------
def detect_candidate_peaks(proc_df, prominence=0.05, distance=8):
    x = proc_df["raman_shift"].to_numpy()
    y = proc_df["intensity"].to_numpy()
    y_norm = y / (y.max() + 1e-8)

    idx, props = find_peaks(y_norm, prominence=prominence, distance=distance)
    candidates = pd.DataFrame({
        "peak_shift": x[idx],
        "intensity": y[idx],
        "prominence": props.get("prominences", np.zeros(len(idx)))
    }).sort_values("prominence", ascending=False).reset_index(drop=True)
    return candidates


# ---------------------------------------------------------
# STEP 6: Filter candidates down to final peaks
# ---------------------------------------------------------
def select_final_peaks(candidates, top_n=10, min_prominence=0.08):
    final = candidates[candidates["prominence"] >= min_prominence]
    final = final.sort_values("peak_shift").reset_index(drop=True)
    return final.head(top_n)


# ---------------------------------------------------------
# STEP 7: Plot region of interest
# ---------------------------------------------------------
def plot_roi(proc_df, final_peaks):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(proc_df["raman_shift"], proc_df["intensity"], color="black", linewidth=1)
    ax.scatter(final_peaks["peak_shift"], final_peaks["intensity"], color="red", zorder=5)
    ax.set_xlim(ROI_MIN, ROI_MAX)
    ax.set_ylim(100, 1200)
    ax.set_xlabel("Raman shift (cm$^{-1}$)")
    ax.set_ylabel("Intensity")
    ax.set_title("Region of Interest: 50-490 cm$^{-1}$")
    return fig


# ===========================================================
# MAIN APP
# ===========================================================
st.title("Raman Peak Detector")

uploaded = st.file_uploader("Upload Raman TXT/CSV file", type=["txt", "csv"])

if uploaded is not None:
    raw_df = read_spectrum(uploaded)
    corrected_df = rayleigh_correction(raw_df)
    roi_df = select_roi(corrected_df)
    proc_df = preprocess(roi_df)

    candidates = detect_candidate_peaks(proc_df)
    final_peaks = select_final_peaks(candidates)

    st.subheader("Candidate peaks")
    st.dataframe(candidates[["peak_shift", "prominence"]], use_container_width=True)

    st.subheader("Final peaks")
    st.dataframe(final_peaks[["peak_shift", "intensity"]], use_container_width=True)

    st.pyplot(plot_roi(proc_df, final_peaks))
