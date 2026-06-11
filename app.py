import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit


st.set_page_config(
    page_title="Raman Peak Analyzer",
    layout="wide"
)

st.title("Raman Peak Analyzer")


# ============================================================
# LORENTZIAN
# ============================================================

def lorentzian(x, x0, gamma, A):
    return A * (gamma**2 / ((x - x0)**2 + gamma**2))


# ============================================================
# RAMAN SHIFT DISPLAY RANGE
# ============================================================

X_MIN, X_MAX = 0, 550   # cm⁻¹


# ============================================================
# ANALYSIS FUNCTION
# ============================================================

def analyze_spectrum(uploaded_file):

    df = pd.read_csv(
        uploaded_file,
        sep=r"\s+|,|\t",
        engine="python",
        header=None
    )

    x = df.iloc[:, 0].values
    y = df.iloc[:, 1].values.astype(float)

    # --------------------------------------------------------
    # RAYLEIGH CALIBRATION
    # Shift x-axis so the intensity maximum (Rayleigh peak) = 0.
    # This is the only Rayleigh criterion used — no zeroing, no
    # cutoff mask that could swallow real low-shift Raman peaks.
    # --------------------------------------------------------

    max_idx = np.argmax(y)
    x_use   = x - x[max_idx]       # Rayleigh peak → 0 cm⁻¹

    # Keep the raw y before any processing (for the raw panel)
    y_raw_original = y.copy()

    # --------------------------------------------------------
    # RETAIN ONLY POSITIVE RAMAN SHIFTS  (Stokes side)
    # The Rayleigh peak itself sits at index max_idx = 0 cm⁻¹.
    # Everything at x_use < 0 is anti-Stokes — discard.
    # --------------------------------------------------------

    pos_mask  = x_use >= 0
    x_use     = x_use[pos_mask]
    y         = y[pos_mask]
    y_raw_pos = y_raw_original[pos_mask]

    # --------------------------------------------------------
    # SMOOTHING
    # --------------------------------------------------------

    y_smooth = savgol_filter(y, 21, 3)

    # --------------------------------------------------------
    # BASELINE  (long-window Savitzky–Golay)
    # --------------------------------------------------------

    baseline = savgol_filter(y_smooth, 151, 3)
    signal   = y_smooth - baseline
    signal   = np.clip(signal, 0, None)

    # --------------------------------------------------------
    # NOISE ESTIMATION
    # Use the quiet tail of the Stokes window (450–540 cm⁻¹).
    # This is always inside our data range and contains no
    # strong Raman peaks for most samples, giving a reliable
    # floor for dynamic thresholding.
    # --------------------------------------------------------

    noise_region = signal[(x_use > 450) & (x_use < 540)]

    if len(noise_region) < 5:
        # Fallback: use the lower 20th percentile of the whole
        # signal as a conservative noise floor estimate.
        noise_std = np.percentile(signal[signal > 0], 20) if np.any(signal > 0) else np.std(signal)
    else:
        noise_std = np.std(noise_region)

    # Guard against near-zero noise (flat/featureless spectrum)
    noise_std = max(noise_std, 1e-6)

    dynamic_prominence = 3.5 * noise_std
    dynamic_height     = 2.5 * noise_std

    # --------------------------------------------------------
    # CANDIDATE PEAKS  (your find_peaks algorithm, preserved)
    # --------------------------------------------------------

    candidate_peaks, _ = find_peaks(
        signal,
        prominence=dynamic_prominence,
        height=dynamic_height,
        distance=5,     # tighter than 8 — catches closely spaced peaks
        width=2
    )

    # Restrict to display range; start after a minimal Rayleigh
    # exclusion zone (5 cm⁻¹) to avoid fitting the Rayleigh tail.
    RAYLEIGH_EXCLUSION = 5   # cm⁻¹ — very conservative, only masks the tail

    candidate_peaks = np.array([
        p for p in candidate_peaks
        if RAYLEIGH_EXCLUSION < x_use[p] <= X_MAX
    ])

    # --------------------------------------------------------
    # FALSE PEAK REMOVAL  (SNR + physical width filter)
    # --------------------------------------------------------

    filtered_peaks = []

    for p in candidate_peaks:

        peak_height = signal[p]
        half_height = peak_height / 2

        # Walk left and right to find FWHM indices
        left = p
        while left > 0 and signal[left] > half_height:
            left -= 1

        right = p
        while right < len(signal) - 1 and signal[right] > half_height:
            right += 1

        width_pts = right - left

        # Local noise: window around the peak, excluding the peak
        # itself (±5 pts) to avoid self-contamination
        local_left   = signal[max(0, p - 20):max(0, p - 5)]
        local_right  = signal[min(len(signal), p + 5):min(len(signal), p + 20)]
        local_noise  = np.std(np.concatenate([local_left, local_right])) if (len(local_left) + len(local_right)) > 2 else noise_std
        snr          = peak_height / (local_noise + 1e-9)

        if snr < 2.0:        # relaxed from 2.5 — catch weaker real peaks
            continue
        if width_pts < 2:    # must span at least 2 data points
            continue

        filtered_peaks.append(p)

    candidate_peaks = np.array(filtered_peaks)

    # --------------------------------------------------------
    # LORENTZIAN FIT — returns (position, amplitude, FWHM)
    # --------------------------------------------------------

    final_peaks = []

    for p in candidate_peaks:

        try:
            left  = max(0, p - 20)
            right = min(len(signal) - 1, p + 20)

            x_fit = x_use[left:right]
            y_fit = signal[left:right]

            if len(x_fit) < 5:
                continue

            p0     = [x_use[p], 5.0, signal[p]]
            bounds = (
                [x_use[max(0, p - 10)], 0.5,   0],
                [x_use[min(len(x_use)-1, p + 10)], 80.0,  signal[p] * 5]
            )

            popt, _ = curve_fit(lorentzian, x_fit, y_fit, p0=p0, bounds=bounds, maxfev=2000)

            x0, gamma, A = popt
            fwhm = 2 * abs(gamma)

            # Keep only fits whose centre is inside display range
            if 0 < x0 <= X_MAX:
                final_peaks.append((x0, A, fwhm))

        except Exception:
            # Fit failed — fall back to raw peak position
            final_peaks.append((float(x_use[p]), float(signal[p]), 0.0))

    # De-duplicate: if two fitted peaks land within 5 cm⁻¹ of each
    # other, keep the stronger one (can happen after Lorentzian shift)
    final_peaks.sort(key=lambda pk: pk[0])
    deduped = []
    for pk in final_peaks:
        if deduped and abs(pk[0] - deduped[-1][0]) < 5:
            if pk[1] > deduped[-1][1]:
                deduped[-1] = pk
        else:
            deduped.append(pk)
    final_peaks = deduped

    return {
        "sample": uploaded_file.name,
        "x":      x_use,
        "y_raw":  y_raw_pos,
        "signal": signal,
        "peaks":  final_peaks
    }


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_files = st.file_uploader(
    "Upload Raman Files",
    type=["txt"],
    accept_multiple_files=True
)


if uploaded_files:

    results = []

    with st.spinner("Analyzing Files..."):
        for file in uploaded_files:
            result = analyze_spectrum(file)
            results.append(result)

    st.success(f"{len(results)} files processed.")

    # ========================================================
    # PEAK TABLE  (position | intensity | FWHM)
    # ========================================================

    peak_rows = []

    for result in results:
        sample = result["sample"]
        for i, peak in enumerate(result["peaks"], start=1):
            peak_rows.append({
                "Sample":             sample,
                "Peak Number":        i,
                "Raman Shift (cm⁻¹)": round(peak[0], 2),
                "Intensity":          round(peak[1], 2),
                "FWHM (cm⁻¹)":        round(peak[2], 2)
            })

    peak_df = pd.DataFrame(peak_rows)

    st.subheader("Detected Peaks")
    st.dataframe(peak_df, use_container_width=True)

    # ========================================================
    # CSV DOWNLOAD
    # ========================================================

    csv = peak_df.to_csv(index=False)

    st.download_button(
        "Download Peak Table",
        csv,
        file_name="Peak_Table.csv",
        mime="text/csv"
    )

    # ========================================================
    # OVERLAY GRAPH  (processed signals, 0–550 cm⁻¹)
    # ========================================================

    st.subheader("Overlay Raman Spectra (Processed)")

    fig, ax = plt.subplots(figsize=(12, 6))

    for result in results:
        ax.plot(result["x"], result["signal"], label=result["sample"])

    ax.set_xlim(X_MIN, X_MAX)
    ax.legend()
    ax.grid()
    ax.set_xlabel("Raman Shift (cm⁻¹)")
    ax.set_ylabel("Intensity")

    st.pyplot(fig)
    plt.close(fig)

    # ========================================================
    # RAW + PROCESSED panels — one figure per sample
    # ========================================================

    st.subheader("Raw Data Spectra (per sample)")

    for result in results:

        st.markdown(f"**{result['sample']}**")

        fig_raw, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Top panel: processed signal with peak markers
        axes[0].plot(result["x"], result["signal"], color="steelblue", linewidth=1.2)

        for peak in result["peaks"]:
            axes[0].axvline(peak[0], color="red", linestyle="--", alpha=0.5, linewidth=0.8)

        axes[0].set_xlim(X_MIN, X_MAX)
        axes[0].set_title("Processed Signal (baseline-corrected)")
        axes[0].set_ylabel("Intensity")
        axes[0].set_xlabel("Raman Shift (cm⁻¹)")
        axes[0].grid(alpha=0.4)

        # Bottom panel: raw signal (auto y-scale, no hardcoded limits)
        raw_x = result["x"] if len(result["x"]) == len(result["y_raw"]) else np.arange(len(result["y_raw"]))

        axes[1].plot(raw_x, result["y_raw"], color="darkorange", linewidth=1.0, alpha=0.85)
        axes[1].set_xlim(X_MIN, X_MAX)
        axes[1].set_title("Raw Signal (Stokes side, unprocessed)")
        axes[1].set_ylabel("Intensity")
        axes[1].set_xlabel("Raman Shift (cm⁻¹)")
        axes[1].grid(alpha=0.4)

        plt.tight_layout()
        st.pyplot(fig_raw)
        plt.close(fig_raw)

    # ========================================================
    # PEAK COMPARISON GRAPH — staggered labels
    # ========================================================

    st.subheader("Peak Comparison Graph")

    sample_names = [r["sample"] for r in results]
    y_positions  = list(range(len(sample_names)))

    fig2, ax2 = plt.subplots(figsize=(14, max(6, len(results) * 1.4)))

    for y_idx, result in enumerate(results):

        peaks = [p[0] for p in result["peaks"]]

        ax2.scatter(peaks, [y_idx] * len(peaks), s=80, zorder=3)

        prev_x      = -np.inf
        row         = 0
        row_offsets = [0.18, 0.36, -0.18]

        for peak in sorted(peaks):

            if peak - prev_x < 60:
                row = (row + 1) % len(row_offsets)
            else:
                row = 0

            ax2.text(
                peak,
                y_idx + row_offsets[row],
                f"{peak:.0f}",
                fontsize=8,
                ha="center",
                va="bottom"
            )

            prev_x = peak

    ax2.set_xlim(X_MIN, X_MAX)
    ax2.set_yticks(y_positions)
    ax2.set_yticklabels(sample_names)
    ax2.grid(alpha=0.35)
    ax2.set_xlabel("Raman Shift (cm⁻¹)")
    ax2.set_ylabel("Sample")
    ax2.set_ylim(-0.7, len(results) - 0.3)

    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)
