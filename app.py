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
    y_smooth = savgol_filter(y, 21, 3)

    # ── Baseline ────────────────────────────────────────────
    baseline = savgol_filter(y_smooth, 151, 3)
    signal   = y_smooth - baseline
    signal   = np.clip(signal, 0, None)

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

    candidate_peaks = np.array([p for p in candidate_peaks if 50 < x_use[p] <= 450])

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

    # ── Build Lorentzian fit curves for each peak ────────────
    fit_curves = []
    for pk in final_peaks:
        x0    = pk["shift"]
        gamma = pk["fwhm"] / 2
        A     = pk["intensity"]
        curve = lorentzian(x_use, x0, gamma, A)
        fit_curves.append(curve)

    return {
        "sample":     uploaded_file.name,
        "x_use":      x_use,
        "y_raw":      y,
        "y_smooth":   y_smooth,
        "signal":     signal,
        "peaks":      final_peaks,
        "fit_curves": fit_curves
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

    x_use      = result["x_use"]
    y_raw      = result["y_raw"]
    y_smooth   = result["y_smooth"]
    peaks      = result["peaks"]
    fit_curves = result["fit_curves"]

    # ── Raw + Smooth overlay graph ───────────────────────────
    fig, ax = plt.subplots(figsize=(13, 5))

    ax.plot(x_use, y_raw,
            color="#4C9BE8",
            linewidth=0.8,
            alpha=0.5,
            label="Raw data",
            zorder=2)

    ax.plot(x_use, y_smooth,
            color="#E84C4C",
            linewidth=1.8,
            alpha=0.92,
            label="Smoothed",
            zorder=3)

    # Mark detected peaks with vertical dashed lines + labels
    if peaks:
        for pk in peaks:
            sh  = pk["shift"]
            idx = np.argmin(np.abs(x_use - sh))
            ax.axvline(sh,
                       color="#2CA02C",
                       linewidth=0.9,
                       linestyle="--",
                       alpha=0.7,
                       zorder=1)
            ax.annotate(
                f"{sh:.1f}",
                xy=(sh, y_smooth[idx]),
                xytext=(3, 8),
                textcoords="offset points",
                fontsize=7.5,
                color="#2CA02C",
                rotation=90,
                va="bottom"
            )

    # Draw Lorentzian fit curves
    for i, curve in enumerate(fit_curves):
        label = "Lorentzian fit" if i == 0 else None
        ax.plot(x_use, curve,
                color="#FF8C00",
                linewidth=1.4,
                linestyle="-",
                alpha=0.85,
                label=label,
                zorder=4)

    # Axis range & ticks
    ax.set_xlim(0, 450)
    ax.set_ylim(0, 500)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
    ax.yaxis.set_major_locator(ticker.AutoLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    ax.tick_params(axis="both", which="major", direction="in", length=5, width=0.8)
    ax.tick_params(axis="both", which="minor", direction="in", length=2.5, width=0.6)

    ax.grid(which="major", linestyle="--", linewidth=0.5, color="grey", alpha=0.4)
    ax.grid(which="minor", linestyle=":",  linewidth=0.3, color="grey", alpha=0.25)

    ax.set_xlabel("Raman Shift (cm⁻¹)", fontsize=11)
    ax.set_ylabel("Intensity (a.u.)",    fontsize=11)
    ax.set_title(f"Raman Spectrum — {result['sample']}",
                 fontsize=12, fontweight="bold", pad=10)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.85)

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

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
ax2.set_xlim(0, 450)
ax2.xaxis.set_major_locator(ticker.MultipleLocator(50))
ax2.xaxis.set_minor_locator(ticker.MultipleLocator(10))
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


# ============================================================
# STACKED OVERLAY COMPARISON GRAPH
# ============================================================

st.markdown("---")
st.subheader("Stacked Spectrum Overlay — All Samples")

OFFSET_STEP = 120   # vertical offset between samples

fig3, ax3 = plt.subplots(figsize=(14, max(5, n_samples * 2 + 2)))

colors3 = plt.cm.tab10(np.linspace(0, 0.9, max(n_samples, 1)))

for i, result in enumerate(results):
    x_use3   = result["x_use"]
    signal3  = result["signal"]
    offset   = i * OFFSET_STEP

    mask = (x_use3 >= 0) & (x_use3 <= 450)

    ax3.plot(x_use3[mask], signal3[mask] + offset,
             color=colors3[i],
             linewidth=1.2,
             label=result["sample"],
             zorder=3)

    # Mark peaks
    for pk in result["peaks"]:
        sh = pk["shift"]
        if 0 <= sh <= 450:
            ax3.axvline(sh,
                        color=colors3[i],
                        linewidth=0.7,
                        linestyle="--",
                        alpha=0.5,
                        zorder=2)
            ax3.text(sh, offset + pk["intensity"] + 6,
                     f"{sh:.0f}",
                     ha="center",
                     fontsize=6.5,
                     color=colors3[i])

ax3.set_xlim(0, 450)
ax3.xaxis.set_major_locator(ticker.MultipleLocator(50))
ax3.xaxis.set_minor_locator(ticker.MultipleLocator(10))
ax3.tick_params(axis="both", which="major", direction="in", length=5, width=0.8)
ax3.tick_params(axis="both", which="minor", direction="in", length=2.5, width=0.6)
ax3.grid(which="major", linestyle="--", linewidth=0.5, color="grey", alpha=0.4)
ax3.grid(which="minor", linestyle=":",  linewidth=0.3, color="grey", alpha=0.2)
ax3.set_xlabel("Raman Shift (cm⁻¹)", fontsize=11)
ax3.set_ylabel("Intensity (a.u.) + offset", fontsize=11)
ax3.set_title("Stacked Spectrum Overlay — All Samples",
              fontsize=13, fontweight="bold", pad=12)
ax3.legend(fontsize=8, loc="upper right", framealpha=0.85)
ax3.set_yticks([])   # offsets make absolute y values meaningless

fig3.tight_layout()
st.pyplot(fig3, use_container_width=True)
plt.close(fig3)


# ============================================================
# COMBINED PEAK TABLE + DOWNLOAD
# ============================================================

st.markdown("---")
st.subheader("Combined Peak Table — All Samples")

all_rows = []
for result in results:
    for pk in result["peaks"]:
        all_rows.append({
            "Sample":            result["sample"],
            "Raman Shift (cm⁻¹)": pk["shift"],
            "Intensity (a.u.)":  pk["intensity"],
            "FWHM (cm⁻¹)":       pk["fwhm"]
        })

if all_rows:
    df_all = pd.DataFrame(all_rows)
    df_all.index = range(1, len(df_all) + 1)
    st.dataframe(df_all, use_container_width=True)

    csv_all = df_all.to_csv(index=True)
    st.download_button(
        "Download combined peak table (all samples)",
        csv_all,
        file_name="peaks_all_samples.csv",
        mime="text/csv",
        key="dl_combined"
    )
else:
    st.info("No peaks detected across any sample.")
