import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit

st.set_page_config(page_title="Raman Peak Analyzer", layout="wide")
st.title("Raman Peak Analyzer")


# ============================================================
# LORENTZIAN
# ============================================================

def lorentzian(x, x0, gamma, A):
    return A * (gamma**2 / ((x - x0)**2 + gamma**2))


# ============================================================
# ANALYSIS FUNCTION  (original peak-finding logic preserved)
# ============================================================

def analyze_spectrum(uploaded_file):

    df = pd.read_csv(
        uploaded_file,
        sep=r"\s+|,|\t",
        engine="python",
        header=None
    )

    x = df.iloc[:, 0].values.astype(float)
    y = df.iloc[:, 1].values.astype(float)

    # ── Shift to Rayleigh peak ──────────────────────────────
    max_idx = np.argmax(y)
    x_use = x - x[max_idx]

    # ── Smoothing ───────────────────────────────────────────
    y_smooth = savgol_filter(y, 11, 2)

    # ── Baseline ────────────────────────────────────────────
    baseline = savgol_filter(y_smooth, 301, 3)
    signal   = y_smooth - baseline
    # signal = np.clip(signal, 0, None)

    # ── Noise (original region 700–1200) ────────────────────
    noise_region = signal[(x_use > 700) & (x_use < 1200)]
    if len(noise_region) == 0:
        noise_std = np.std(signal)
    else:
        noise_std = np.std(noise_region)

    dynamic_prominence = 4 * noise_std
    dynamic_height     = 3 * noise_std

    # ── Peak detection (original logic) ─────────────────────
    candidate_peaks, _ = find_peaks(
        signal,
        prominence=dynamic_prominence,
        height=dynamic_height,
        distance=8,
        width=2
    )

    candidate_peaks = np.array([p for p in candidate_peaks if x_use[p] > 50])

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

        width = right - left

        local_region = signal[max(0, p - 20):min(len(signal), p + 20)]
        local_noise  = np.std(local_region)
        snr          = peak_height / (local_noise + 1e-9)

        if snr < 2.5:
            continue
        if width < 3:
            continue

        filtered_peaks.append(p)

    candidate_peaks = np.array(filtered_peaks)

    # ── Lorentzian fit → shift, intensity, FWHM ─────────────
    final_peaks = []
    for p in candidate_peaks:
        try:
            left  = max(0, p - 15)
            right = min(len(signal) - 1, p + 15)

            x_fit = x_use[left:right]
            y_fit = signal[left:right]

            p0 = [x_use[p], 5, signal[p]]

            popt, _ = curve_fit(lorentzian, x_fit, y_fit, p0=p0, maxfev=10000)
            x0, gamma, A = popt

            fwhm = abs(2 * gamma)

            final_peaks.append({
                "shift":     round(x0,   2),
                "intensity": round(A,    2),
                "fwhm":      round(fwhm, 2)
            })

        except Exception:
            pass

    final_peaks.sort(key=lambda p: p["shift"])

    return {
        "sample":   uploaded_file.name,
        "x":        x,
        "x_use":    x_use,
        "y_raw":    y,
        "y_smooth": y_smooth,
        "baseline": baseline,
        "signal":   signal,
        "peaks":    final_peaks
    }


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_files = st.file_uploader(
    "Upload Raman Files (.txt)",
    type=["txt"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.stop()

results = []
with st.spinner("Analyzing files..."):
    for f in uploaded_files:
        results.append(analyze_spectrum(f))

st.success(f"{len(results)} file(s) processed.")


# ============================================================
# PER-SAMPLE SECTIONS
# ============================================================

for result in results:

    st.markdown("---")
    st.subheader(f"Sample: {result['sample']}")

    x        = result["x"]
    x_use    = result["x_use"]
    y_raw    = result["y_raw"]
    y_smooth = result["y_smooth"]
    baseline = result["baseline"]
    signal   = result["signal"]
    peaks    = result["peaks"]

    y_max = max(np.max(y_raw) * 1.05, 600)

    # ── Graph 1: Original Raw Data ────────────────────────────
    st.markdown("### Graph 1: Original Raw Data")

    fig1, ax1 = plt.subplots(figsize=(14, 6))
    ax1.plot(x, y_raw, color='black', linewidth=0.8)
    ax1.set_title("Original Raw Raman Spectrum")
    ax1.set_xlabel("Raman Shift (cm⁻¹)")
    ax1.set_ylabel("Intensity")
    ax1.set_xlim(0, 1000)
    ax1.set_ylim(0, y_max)
    ax1.grid(alpha=0.3)
    fig1.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

    # ── Graph 2: Rayleigh Shifted Raw Data ───────────────────
    st.markdown("### Graph 2: Rayleigh Shifted Raw Data")

    fig2, ax2 = plt.subplots(figsize=(14, 6))
    ax2.plot(x_use, y_raw, color='blue', linewidth=0.8)
    ax2.set_title("Rayleigh Shifted Raw Spectrum")
    ax2.set_xlabel("Raman Shift (cm⁻¹)")
    ax2.set_ylabel("Intensity")
    ax2.set_xlim(0, 1000)
    ax2.set_ylim(0, y_max)
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    # ── Graph 3: Raw + Smoothed Overlay ──────────────────────
    st.markdown("### Graph 3: Raw vs Smoothed Spectrum")

    fig3, ax3 = plt.subplots(figsize=(14, 6))
    ax3.plot(x_use, y_raw,    color='black', linewidth=0.8, alpha=0.6, label='Raw Data')
    ax3.plot(x_use, y_smooth, color='red',   linewidth=2,             label='Smoothed')
    ax3.set_title("Raw vs Smoothed Spectrum")
    ax3.set_xlabel("Raman Shift (cm⁻¹)")
    ax3.set_ylabel("Intensity")
    ax3.set_xlim(0, 1000)
    ax3.set_ylim(0, y_max)
    ax3.legend()
    ax3.grid(alpha=0.3)
    fig3.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

    # ── Graph 4: Baseline Visualization ──────────────────────
    st.markdown("### Graph 4: Baseline Correction")

    fig4, ax4 = plt.subplots(figsize=(14, 6))
    ax4.plot(x_use, y_smooth, color='black',  linewidth=1,   label='Smoothed')
    ax4.plot(x_use, baseline, color='orange', linewidth=2,   label='Baseline')
    ax4.set_title("Baseline Correction")
    ax4.set_xlabel("Raman Shift (cm⁻¹)")
    ax4.set_ylabel("Intensity")
    ax4.set_xlim(0, 1000)
    ax4.set_ylim(0, y_max)
    ax4.legend()
    ax4.grid(alpha=0.3)
    fig4.tight_layout()
    st.pyplot(fig4)
    plt.close(fig4)

    # ── Graph 5: Final Processed Spectrum with Peaks ─────────
    st.markdown("### Graph 5: Final Peaks After Lorentzian Fit")

    sig_max = np.max(signal) * 1.05 if np.max(signal) > 0 else 1

    fig5, ax5 = plt.subplots(figsize=(14, 6))
    ax5.plot(x_use, signal, color='green', linewidth=2, label='Processed Signal')

    for pk in peaks:
        ax5.scatter(pk["shift"], pk["intensity"], color='red', s=100, zorder=5)
        ax5.text(
            pk["shift"],
            pk["intensity"] + sig_max * 0.02,
            f"{pk['shift']:.1f}",
            fontsize=8,
            rotation=90
        )

    ax5.set_title("Final Peaks After Lorentzian Fit")
    ax5.set_xlabel("Raman Shift (cm⁻¹)")
    ax5.set_ylabel("Intensity")
    ax5.set_xlim(0, 1000)
    ax5.set_ylim(0, sig_max)
    ax5.legend()
    ax5.grid(alpha=0.3)
    fig5.tight_layout()
    st.pyplot(fig5)
    plt.close(fig5)

    # ── Peak table for this sample ───────────────────────────
    if peaks:
        st.markdown(f"**Detected Peaks — {result['sample']}**")
        df_peaks = pd.DataFrame(peaks).rename(columns={
            "shift":     "Raman Shift (cm⁻¹)",
            "intensity": "Intensity (a.u.)",
            "fwhm":      "FWHM (cm⁻¹)"
        })
        df_peaks.index = range(1, len(df_peaks) + 1)
        st.dataframe(df_peaks, use_container_width=True)

        csv = df_peaks.to_csv(index=True)
        st.download_button(
            f"Download peak table — {result['sample']}",
            csv,
            file_name=f"peaks_{result['sample']}.csv",
            mime="text/csv",
            key=f"dl_{result['sample']}"
        )
    else:
        st.info("No peaks detected for this sample.")


# ============================================================
# FINAL COMPARISON GRAPH
# ============================================================

st.markdown("---")
st.subheader("Peak Comparison — All Samples")

sample_names = [r["sample"] for r in results]
n_samples    = len(sample_names)

fig2, ax2 = plt.subplots(figsize=(14, max(5, n_samples * 1.4 + 2)))

colors = plt.cm.tab10(np.linspace(0, 0.9, max(n_samples, 1)))

for i, result in enumerate(results):
    shifts = [p["shift"] for p in result["peaks"]]
    y_pos  = [i] * len(shifts)

    ax2.scatter(shifts, y_pos,
                s=120,
                color=colors[i],
                zorder=3,
                edgecolors="black",
                linewidths=0.6)

    for sh in shifts:
        ax2.text(sh, i + 0.18,
                 f"{sh:.2f}",
                 ha="center",
                 va="bottom",
                 fontsize=8,
                 color=colors[i],
                 fontweight="bold")

ax2.set_yticks(range(n_samples))
ax2.set_yticklabels(sample_names, fontsize=10)
ax2.set_xlim(0, 1000)
ax2.xaxis.set_major_locator(ticker.MultipleLocator(100))
ax2.xaxis.set_minor_locator(ticker.MultipleLocator(25))
ax2.tick_params(axis="both", which="major", direction="in", length=5, width=0.8)
ax2.tick_params(axis="both", which="minor", direction="in", length=2.5, width=0.6)
ax2.grid(which="major", axis="x", linestyle="--", linewidth=0.5, color="grey", alpha=0.4)
ax2.grid(which="minor", axis="x", linestyle=":",  linewidth=0.3, color="grey", alpha=0.25)

for y_pos in range(n_samples):
    ax2.axhline(y_pos, color="grey", linewidth=0.4, linestyle="-", alpha=0.25, zorder=1)

ax2.set_xlabel("Raman Shift (cm⁻¹)", fontsize=11)
ax2.set_ylabel("Sample",             fontsize=11)
ax2.set_title("Peak Position Comparison Across Samples",
              fontsize=13, fontweight="bold", pad=12)
ax2.set_ylim(-0.6, n_samples - 0.4)

fig2.tight_layout()
st.pyplot(fig2, use_container_width=True)
plt.close(fig2)
