import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import pywt
from scipy.signal import find_peaks
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.optimize import curve_fit

st.set_page_config(page_title="Raman Peak Detector", layout="centered")

ROI_MIN = 50.0
ROI_MAX = 490.0


# ---------------------------------------------------------
# STEP 0: Read raw spectrum file
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
# STEP 1: Rayleigh alignment (shift Rayleigh peak to 0 cm-1)
# ---------------------------------------------------------
def rayleigh_alignment(df):
    x = df["raman_shift"].to_numpy()
    y = df["intensity"].to_numpy()

    rayleigh_idx = np.argmax(y)
    shift = x[rayleigh_idx]
    x_aligned = x - shift
    return pd.DataFrame({"raman_shift": x_aligned, "intensity": y})


# ---------------------------------------------------------
# STEP 2: Wavelet denoising
# ---------------------------------------------------------
def wavelet_denoise(df, wavelet="db4", level=4):
    y = df["intensity"].to_numpy()

    coeffs = pywt.wavedec(y, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(y)))

    coeffs[1:] = [pywt.threshold(c, threshold, mode="soft") for c in coeffs[1:]]
    y_denoised = pywt.waverec(coeffs, wavelet)[: len(y)]

    out = df.copy()
    out["intensity"] = y_denoised
    return out


# ---------------------------------------------------------
# STEP 3: ALS baseline removal
# ---------------------------------------------------------
def als_baseline(y, lam=1e5, p=0.01, n_iter=10):
    L = len(y)
    D = diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    D = lam * D.dot(D.transpose())
    w = np.ones(L)
    W = diags(w, 0, shape=(L, L))

    for _ in range(n_iter):
        W.setdiag(w)
        Z = W + D
        baseline = spsolve(Z, w * y)
        w = p * (y > baseline) + (1 - p) * (y < baseline)
    return baseline


def remove_baseline(df):
    y = df["intensity"].to_numpy()
    baseline = als_baseline(y)
    out = df.copy()
    out["intensity"] = y - baseline
    return out


# ---------------------------------------------------------
# STEP 4: ROI selection (50-490 cm-1)
# ---------------------------------------------------------
def select_roi(df, roi_min=ROI_MIN, roi_max=ROI_MAX):
    roi = df[(df["raman_shift"] >= roi_min) & (df["raman_shift"] <= roi_max)].copy()
    return roi.reset_index(drop=True)


# ---------------------------------------------------------
# STEP 5: Noise estimation (MAD estimator)
# ---------------------------------------------------------
def estimate_noise_mad(y):
    median = np.median(y)
    mad = np.median(np.abs(y - median))
    return mad / 0.6745


# ---------------------------------------------------------
# STEP 6: Peak enhancement (Lorentzian matched filter)
# ---------------------------------------------------------
def lorentzian_kernel(width, gamma):
    half = width // 2
    t = np.arange(-half, half + 1)
    kernel = 1.0 / (1.0 + (t / gamma) ** 2)
    return kernel / kernel.sum()


def matched_filter(y, gamma=3.0, width=21):
    kernel = lorentzian_kernel(width, gamma)
    return np.convolve(y, kernel, mode="same")


# ---------------------------------------------------------
# STEP 7: Candidate peak detection (high recall)
# ---------------------------------------------------------
def detect_candidate_peaks(x, y_filtered, noise_level, prominence_mult=2.0, min_distance=4, min_width=2):
    prominence = prominence_mult * noise_level
    idx, props = find_peaks(y_filtered, prominence=prominence, distance=min_distance, width=min_width)

    candidates = pd.DataFrame({
        "peak_shift": x[idx],
        "intensity": y_filtered[idx],
        "prominence": props.get("prominences", np.zeros(len(idx))),
        "width": props.get("widths", np.zeros(len(idx)))
    }).sort_values("peak_shift").reset_index(drop=True)
    return candidates


# ---------------------------------------------------------
# STEP 8: Lorentzian peak refinement (fit each candidate)
# ---------------------------------------------------------
def lorentzian(x, x0, A, gamma):
    return A / (1.0 + ((x - x0) / gamma) ** 2)


def refine_peaks(x, y, candidates, fit_window=15):
    refined = []
    for _, row in candidates.iterrows():
        center_idx = np.argmin(np.abs(x - row["peak_shift"]))
        lo = max(0, center_idx - fit_window)
        hi = min(len(x), center_idx + fit_window)
        x_win = x[lo:hi]
        y_win = y[lo:hi]

        try:
            popt, _ = curve_fit(
                lorentzian, x_win, y_win,
                p0=[row["peak_shift"], row["intensity"], 3.0],
                maxfev=2000
            )
            x0, A, gamma = popt
        except RuntimeError:
            x0, A, gamma = row["peak_shift"], row["intensity"], row["width"]

        refined.append({"peak_shift": x0, "intensity": A, "width": gamma})

    return pd.DataFrame(refined).sort_values("peak_shift").reset_index(drop=True)


# ---------------------------------------------------------
# STEP 9: Plot region of interest
# ---------------------------------------------------------
def plot_roi(x, y, final_peaks):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, y, color="black", linewidth=1)
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
    aligned_df = rayleigh_alignment(raw_df)
    denoised_df = wavelet_denoise(aligned_df)
    debaselined_df = remove_baseline(denoised_df)
    roi_df = select_roi(debaselined_df)

    x = roi_df["raman_shift"].to_numpy()
    y = roi_df["intensity"].to_numpy()

    noise_level = estimate_noise_mad(y)
    y_filtered = matched_filter(y)

    candidates = detect_candidate_peaks(x, y_filtered, noise_level)
    final_peaks = refine_peaks(x, y, candidates)

    st.subheader("Candidate peaks")
    st.dataframe(candidates[["peak_shift", "prominence", "width"]], use_container_width=True)

    st.subheader("Final peaks")
    st.dataframe(final_peaks[["peak_shift", "intensity", "width"]], use_container_width=True)

    st.pyplot(plot_roi(x, y, final_peaks))
