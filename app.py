import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
import streamlit as st

from scipy.signal import savgol_filter, find_peaks, fftconvolve
from scipy.optimize import curve_fit
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from io import StringIO


def process_file(uploaded_file):

    content = uploaded_file.read().decode("utf-8")
    df = pd.read_csv(
        StringIO(content),
        sep=r"\s+|,|\t",
        engine="python",
        header=None
    )

    x = df.iloc[:, 0].values.copy()
    y = df.iloc[:, 1].values.copy()

    rayleigh_region = np.where(x < 100)[0]
    if len(rayleigh_region) == 0:
        rayleigh_idx = np.argmax(y)
    else:
        rayleigh_idx = rayleigh_region[np.argmax(y[rayleigh_region])]
    x_use = x - x[rayleigh_idx]

    wavelet = "sym8"
    level = pywt.dwt_max_level(len(y), wavelet)
    coeffs = pywt.wavedec(y, wavelet, level=level)
    sigma_wav = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma_wav * np.sqrt(2 * np.log(len(y)))
    coeffs_thresh = [coeffs[0]] + [
        pywt.threshold(c, threshold, mode="soft") for c in coeffs[1:]
    ]
    y_smooth = pywt.waverec(coeffs_thresh, wavelet)
    y_smooth = y_smooth[:len(y)]

    def als_baseline(y_in, lam=1e5, p=0.01, n_iter=10):
        L = len(y_in)
        D = diags([1, -2, 1], [0, 1, 2], shape=(L-2, L))
        H = lam * D.T @ D
        w = np.ones(L)
        for _ in range(n_iter):
            W = diags(w, 0, shape=(L, L))
            baseline = spsolve(W + H, w * y_in)
            w = np.where(y_in > baseline, p, 1 - p)
        return baseline

    baseline = als_baseline(y_smooth, lam=1e5, p=0.01)
    signal = y_smooth - baseline
    signal = np.clip(signal, 0, None)

    roi_mask = (x_use >= 0) & (x_use <= 550)
    x_roi = x_use[roi_mask]
    signal = signal[roi_mask]

    mad = np.median(np.abs(signal - np.median(signal)))
    noise_std = 1.4826 * mad
    dynamic_prominence = 4 * noise_std
    dynamic_distance = 8
    dynamic_width = 2

    def lorentzian(x, x0, gamma, A):
        return A * (gamma**2 / ((x - x0)**2 + gamma**2))

    gamma_filter = 10.0
    kernel = lorentzian(np.arange(-50, 51), 0, gamma_filter, 1.0)
    kernel /= kernel.sum()
    signal_enhanced = fftconvolve(signal, kernel, mode="same")

    candidate_peaks, _ = find_peaks(
        signal_enhanced,
        prominence=dynamic_prominence,
        distance=dynamic_distance,
        width=dynamic_width
    )

    filtered_peaks = []
    for p in candidate_peaks:
        peak_height = signal[p]
        half_height = peak_height / 2
        left = p
        while left > 0 and signal[left] > half_height:
            left -= 1
        right = p
        while right < len(signal) - 1 and signal[right] > half_height:
            right += 1
        width_cm = x_roi[right] - x_roi[left]
        if width_cm < 2.0:
            continue
        left_noise = signal[max(0, p-30):max(0, p-10)]
        right_noise = signal[min(len(signal), p+10):min(len(signal), p+30)]
        noise_local = np.concatenate([left_noise, right_noise])
        if len(noise_local) == 0:
            continue
        snr = peak_height / (np.std(noise_local) + 1e-9)
        if snr < 3.0:
            continue
        filtered_peaks.append(p)

    candidate_peaks = np.array(filtered_peaks)

    final_peaks = []
    for p in candidate_peaks:
        try:
            left = max(0, p - 15)
            right = min(len(signal)-1, p + 15)
            x_fit = x_roi[left:right]
            y_fit = signal[left:right]
            p0 = [x_roi[p], 5, signal[p]]
            popt, _ = curve_fit(lorentzian, x_fit, y_fit, p0=p0)
            x0, gamma, A = popt
            final_peaks.append((x0, A))
        except:
            continue

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(x_roi, signal, label="Processed Signal")
    for peak in final_peaks:
        ax.scatter(peak[0], peak[1], color="red", s=100)
    ax.set_title("Final Peaks After Lorentzian Fit")
    ax.set_xlabel("Shifted x-axis")
    ax.set_ylabel("Intensity")
    ax.set_xlim(0, None)
    ax.set_ylim(0, 1000)
    ax.grid()

    return noise_std, dynamic_prominence, candidate_peaks, x_roi, signal, final_peaks, fig


uploaded_files = st.file_uploader(
    "Upload spectra files",
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.write(f"### {uploaded_file.name}")
        try:
            noise_std, dynamic_prominence, candidate_peaks, x_roi, signal, final_peaks, fig = process_file(uploaded_file)

            st.write(f"Estimated Noise STD (MAD) = {noise_std:.4f}")
            st.write(f"Dynamic Prominence = {dynamic_prominence:.4f}")

            st.write("**Candidate Peaks:**")
            for p in candidate_peaks:
                st.write(f"x = {x_roi[p]:.2f} cm⁻¹ , intensity = {signal[p]:.2f}")

            st.write("**Final Peaks After Lorentzian Fit:**")
            for peak in final_peaks:
                st.write(f"x = {peak[0]:.2f} , intensity = {peak[1]:.2f}")

            st.pyplot(fig)
            plt.close(fig)

        except Exception as e:
            st.write(f"Error processing {uploaded_file.name}: {e}")
