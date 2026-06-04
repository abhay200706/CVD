import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Raman Peak Analyzer",
    layout="wide",
    page_icon="🔬"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

.stApp {
    background: #0e1117;
    color: #e8eaed;
}

h1, h2, h3 {
    font-family: 'IBM Plex Mono', monospace !important;
}

.main-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
    color: #7DF9C8;
    letter-spacing: -0.02em;
    border-bottom: 2px solid #7DF9C8;
    padding-bottom: 0.5rem;
    margin-bottom: 1.5rem;
}

.sample-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.3rem;
    font-weight: 600;
    color: #FFD166;
    background: rgba(255,209,102,0.08);
    border-left: 4px solid #FFD166;
    padding: 0.6rem 1rem;
    border-radius: 0 8px 8px 0;
    margin: 1.5rem 0 1rem 0;
}

.metric-card {
    background: rgba(125,249,200,0.06);
    border: 1px solid rgba(125,249,200,0.18);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}

.metric-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #7DF9C8;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}

.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.4rem;
    font-weight: 600;
    color: #ffffff;
}

.divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.08);
    margin: 2rem 0;
}

.info-box {
    background: rgba(100,160,255,0.08);
    border: 1px solid rgba(100,160,255,0.2);
    border-radius: 8px;
    padding: 0.8rem 1.2rem;
    font-size: 0.85rem;
    color: #b0c4ff;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LORENTZIAN
# ─────────────────────────────────────────────
def lorentzian(x, x0, gamma, A):
    return A * (gamma**2 / ((x - x0)**2 + gamma**2))


# ─────────────────────────────────────────────
# MATPLOTLIB DARK THEME HELPER
# ─────────────────────────────────────────────
def apply_dark_theme(fig, ax_list):
    fig.patch.set_facecolor('#141820')
    for ax in ax_list:
        ax.set_facecolor('#1a1f2e')
        ax.tick_params(colors='#aab4c8', labelsize=9)
        ax.xaxis.label.set_color('#ccd6f0')
        ax.yaxis.label.set_color('#ccd6f0')
        ax.title.set_color('#e8eaed')
        for spine in ax.spines.values():
            spine.set_edgecolor('#2e3650')
        ax.grid(True, color='#252d42', linewidth=0.6, linestyle='--', alpha=0.7)


# ─────────────────────────────────────────────
# MAIN ANALYSIS
# ─────────────────────────────────────────────
def analyze_spectrum(uploaded_file):
    df = pd.read_csv(
        uploaded_file,
        sep=r"\s+|,|\t",
        engine="python",
        header=None
    )

    x_raw = df.iloc[:, 0].values.astype(float)
    y_raw = df.iloc[:, 1].values.astype(float)

    # --- Shift to Rayleigh peak ---
    max_idx = np.argmax(y_raw)
    x_use = x_raw - x_raw[max_idx]

    # ── SILICON CUTOFF ──────────────────────────────────────────────────────
    # Keep only the range that matters for silicon and common Raman features.
    # The silicon first-order peak is ~521 cm⁻¹; we cut at 600 cm⁻¹ to be
    # safe, and also discard the Rayleigh tail below 50 cm⁻¹.
    RAMAN_MIN = 50      # cm⁻¹  — discard Rayleigh tail
    RAMAN_MAX = 600     # cm⁻¹  — discard everything beyond Si second-order
    roi_mask = (x_use >= RAMAN_MIN) & (x_use <= RAMAN_MAX)

    # Keep raw for plotting (wider view), but analyse only in ROI
    x_plot = x_use          # full shifted axis for display
    y_plot = y_raw          # raw intensity for display

    x_work = x_use[roi_mask]
    y_work = y_raw[roi_mask]

    # --- Smooth (Savitzky-Golay) ---
    wl_smooth = min(21, len(y_work) - (1 if len(y_work) % 2 == 0 else 0))
    if wl_smooth < 5:
        wl_smooth = 5
    y_smooth_work = savgol_filter(y_work, wl_smooth, 3)

    # Build full-length smooth for overlay on raw plot
    wl_full = min(21, len(y_raw) - (1 if len(y_raw) % 2 == 0 else 0))
    if wl_full < 5:
        wl_full = 5
    y_smooth_full = savgol_filter(y_raw, wl_full, 3)

    # --- Baseline ---
    wl_base = min(151, len(y_smooth_work) - (1 if len(y_smooth_work) % 2 == 0 else 0))
    if wl_base < 5:
        wl_base = 5
    baseline = savgol_filter(y_smooth_work, wl_base, 3)
    signal = np.clip(y_smooth_work - baseline, 0, None)

    # --- Noise estimate ---
    noise_region = signal[(x_work > 200) & (x_work < 400)]
    noise_std = np.std(noise_region) if len(noise_region) > 5 else np.std(signal)

    # --- Peak detection ---
    candidate_peaks, _ = find_peaks(
        signal,
        prominence=max(4 * noise_std, signal.max() * 0.05),
        height=3 * noise_std,
        distance=8,
        width=2
    )

    # SNR & width filter
    filtered_peaks = []
    for p in candidate_peaks:
        ph = signal[p]
        half = ph / 2
        l, r = p, p
        while l > 0 and signal[l] > half:
            l -= 1
        while r < len(signal) - 1 and signal[r] > half:
            r += 1
        width_pts = r - l
        local = signal[max(0, p - 20):min(len(signal), p + 20)]
        snr = ph / (np.std(local) + 1e-9)
        if snr >= 2.5 and width_pts >= 3:
            filtered_peaks.append(p)

    # --- Lorentzian fit → extract x0, A, FWHM ---
    final_peaks = []
    for p in filtered_peaks:
        try:
            left  = max(0, p - 20)
            right = min(len(signal) - 1, p + 20)
            xf = x_work[left:right]
            yf = signal[left:right]
            p0 = [x_work[p], 5.0, signal[p]]
            bounds = (
                [x_work[p] - 30, 0.5, 0],
                [x_work[p] + 30, 80,  signal[p] * 3]
            )
            popt, pcov = curve_fit(lorentzian, xf, yf, p0=p0, bounds=bounds, maxfev=5000)
            x0, gamma, A = popt
            perr = np.sqrt(np.diag(pcov))
            fwhm = 2 * abs(gamma)           # Lorentzian FWHM = 2γ
            if 50 < x0 < RAMAN_MAX and A > 0:
                final_peaks.append({
                    "shift": round(x0, 2),
                    "intensity": round(A, 2),
                    "fwhm": round(fwhm, 2),
                    "gamma_err": round(perr[1], 3) if perr[1] < 1e6 else None
                })
        except Exception:
            pass

    # Sort by shift
    final_peaks.sort(key=lambda p: p["shift"])

    return {
        "sample":        uploaded_file.name,
        "x_plot":        x_plot,
        "y_plot":        y_plot,
        "y_smooth_full": y_smooth_full,
        "x_work":        x_work,
        "signal":        signal,
        "peaks":         final_peaks,
        "roi_mask":      roi_mask,
        "raman_max":     RAMAN_MAX,
    }


# ─────────────────────────────────────────────
# PER-SAMPLE PLOT (raw + smooth overlay)
# ─────────────────────────────────────────────
def plot_sample(result, color_raw, color_smooth, color_peaks):
    x_plot        = result["x_plot"]
    y_plot        = result["y_plot"]
    y_smooth_full = result["y_smooth_full"]
    x_work        = result["x_work"]
    signal        = result["signal"]
    peaks         = result["peaks"]
    raman_max     = result["raman_max"]

    fig, axes = plt.subplots(
        1, 2,
        figsize=(14, 4.5),
        gridspec_kw={"width_ratios": [1.6, 1]}
    )
    apply_dark_theme(fig, axes)

    ax_main, ax_zoom = axes

    # ── Left: raw + smooth full spectrum ───────────────────────────────────
    ax_main.plot(x_plot, y_plot,
                 color=color_raw, linewidth=0.8, alpha=0.55, label="Raw data")
    ax_main.plot(x_plot, y_smooth_full,
                 color=color_smooth, linewidth=1.6, alpha=0.9, label="Smoothed")
    ax_main.axvline(raman_max, color='#ff6b6b', linewidth=1, linestyle=':', alpha=0.7,
                    label=f"Analysis limit ({raman_max} cm⁻¹)")
    ax_main.set_xlabel("Raman Shift (cm⁻¹)", fontsize=10)
    ax_main.set_ylabel("Intensity (a.u.)", fontsize=10)
    ax_main.set_title("Full Spectrum: Raw vs Smoothed", fontsize=11, fontweight='bold')
    ax_main.legend(fontsize=8, facecolor='#1a1f2e', edgecolor='#2e3650', labelcolor='#ccd6f0')

    # ── Right: baseline-corrected ROI with detected peaks ──────────────────
    ax_zoom.plot(x_work, signal,
                 color=color_smooth, linewidth=1.5, label="Baseline-corrected")
    ax_zoom.fill_between(x_work, 0, signal, color=color_smooth, alpha=0.12)

    for pk in peaks:
        sh = pk["shift"]
        # find nearest index in x_work
        idx = np.argmin(np.abs(x_work - sh))
        ht  = signal[idx]
        ax_zoom.plot(sh, ht, 'o', color=color_peaks, markersize=7, zorder=5)
        ax_zoom.annotate(
            f"{sh:.1f}",
            xy=(sh, ht),
            xytext=(0, 10),
            textcoords="offset points",
            ha='center',
            fontsize=8,
            color=color_peaks,
            fontfamily='monospace'
        )
        # FWHM bar
        fwhm = pk["fwhm"]
        y_half = ht / 2
        ax_zoom.hlines(y_half, sh - fwhm/2, sh + fwhm/2,
                       color=color_peaks, linewidth=1.5, linestyles='--', alpha=0.7)

    ax_zoom.set_xlabel("Raman Shift (cm⁻¹)", fontsize=10)
    ax_zoom.set_ylabel("Intensity (a.u.)", fontsize=10)
    ax_zoom.set_title("Peak Detection Region (ROI)", fontsize=11, fontweight='bold')
    ax_zoom.legend(fontsize=8, facecolor='#1a1f2e', edgecolor='#2e3650', labelcolor='#ccd6f0')

    fig.tight_layout(pad=2)
    return fig


# ─────────────────────────────────────────────
# COMPARATIVE PLOT
# ─────────────────────────────────────────────
def plot_comparison(all_results, palette):
    fig, ax = plt.subplots(figsize=(14, max(4, len(all_results) * 1.1 + 2)))
    apply_dark_theme(fig, [ax])

    sample_names = [r["sample"] for r in all_results]
    yticks = list(range(len(sample_names)))

    for i, result in enumerate(all_results):
        color = palette[i % len(palette)]
        for pk in result["peaks"]:
            sh = pk["shift"]
            ax.scatter(sh, i, s=120, color=color, zorder=4, edgecolors='white', linewidths=0.5)
            ax.text(
                sh, i + 0.22,
                f"{sh:.2f}",
                ha='center', va='bottom',
                fontsize=7.5, color=color,
                fontfamily='monospace',
                fontweight='bold'
            )

    ax.set_yticks(yticks)
    ax.set_yticklabels(sample_names, fontsize=9)
    ax.set_xlabel("Raman Shift (cm⁻¹)", fontsize=11)
    ax.set_title("Peak Position Comparison Across Samples", fontsize=13, fontweight='bold', color='#e8eaed')
    ax.set_ylim(-0.6, len(sample_names) - 0.4)

    # Light horizontal guide lines
    for y in yticks:
        ax.axhline(y, color='#2e3650', linewidth=0.8, zorder=0)

    fig.tight_layout(pad=2)
    return fig


# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────
st.markdown('<div class="main-title">🔬 Raman Peak Analyzer</div>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
Upload one or more Raman spectra (whitespace / comma / tab delimited .txt files with two columns: 
<b>wavenumber</b> and <b>intensity</b>). Analysis is automatically restricted to the 
<b>50 – 600 cm⁻¹</b> region to avoid artefacts beyond the silicon first-order peak.
</div>
""", unsafe_allow_html=True)

# ── Sidebar controls ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Analysis Settings")
    raman_max_ui = st.slider("Upper Raman cutoff (cm⁻¹)", 400, 800, 600, step=10,
                              help="Peaks above this wavenumber are ignored.")
    raman_min_ui = st.slider("Lower Raman cutoff (cm⁻¹)", 10, 100, 50, step=5,
                              help="Removes Rayleigh tail artefacts.")
    snr_threshold = st.slider("Min SNR for peak acceptance", 1.5, 6.0, 2.5, step=0.5)
    st.markdown("---")
    st.markdown("### 🎨 Plot Style")
    show_raw = st.checkbox("Show raw data in overlay", value=True)

uploaded_files = st.file_uploader(
    "Upload Raman Spectra (.txt)",
    type=["txt"],
    accept_multiple_files=True
)

# Colour palettes
RAW_COLOR    = "#4a90d9"
SMOOTH_COLOR = "#7DF9C8"
PEAK_COLOR   = "#FFD166"
COMP_PALETTE = ["#7DF9C8", "#FFD166", "#FF6B9D", "#A78BFA", "#60A5FA",
                "#F97316", "#34D399", "#FB923C", "#E879F9", "#38BDF8"]

if uploaded_files:
    results = []
    progress = st.progress(0, text="Analyzing spectra…")

    for idx, file in enumerate(uploaded_files):
        result = analyze_spectrum(file)
        # Apply sidebar cutoffs & SNR (re-filter peaks by shift range)
        result["peaks"] = [
            p for p in result["peaks"]
            if raman_min_ui < p["shift"] < raman_max_ui
        ]
        results.append(result)
        progress.progress((idx + 1) / len(uploaded_files), text=f"Processed: {file.name}")

    progress.empty()
    st.success(f"✅  {len(results)} file(s) processed successfully.")

    # ── Per-sample sections ─────────────────────────────────────────────────
    for i, result in enumerate(results):
        color_cycle = COMP_PALETTE[i % len(COMP_PALETTE)]
        st.markdown(f'<div class="sample-header">📄 Sample {i+1}: {result["sample"]}</div>',
                    unsafe_allow_html=True)

        # Summary metrics
        n_peaks = len(result["peaks"])
        if n_peaks:
            dominant = max(result["peaks"], key=lambda p: p["intensity"])
            avg_fwhm = np.mean([p["fwhm"] for p in result["peaks"]])
        else:
            dominant = {"shift": "—", "intensity": "—", "fwhm": "—"}
            avg_fwhm = "—"

        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Peaks Detected</div>
                <div class="metric-value">{n_peaks}</div>
            </div>""", unsafe_allow_html=True)
        with mc2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Dominant Shift</div>
                <div class="metric-value">{dominant['shift']} cm⁻¹</div>
            </div>""", unsafe_allow_html=True)
        with mc3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Peak Intensity</div>
                <div class="metric-value">{dominant['intensity']}</div>
            </div>""", unsafe_allow_html=True)
        with mc4:
            val = f"{avg_fwhm:.2f}" if isinstance(avg_fwhm, float) else avg_fwhm
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Avg FWHM</div>
                <div class="metric-value">{val} cm⁻¹</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Graph
        fig = plot_sample(
            result,
            color_raw    = RAW_COLOR if show_raw else SMOOTH_COLOR,
            color_smooth = color_cycle,
            color_peaks  = PEAK_COLOR
        )
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Peak table
        if result["peaks"]:
            st.markdown("**Peak Parameters (Lorentzian Fit)**")
            peak_df = pd.DataFrame(result["peaks"]).rename(columns={
                "shift":     "Raman Shift (cm⁻¹)",
                "intensity": "Intensity (a.u.)",
                "fwhm":      "FWHM (cm⁻¹)",
                "gamma_err": "γ Fit Error"
            })
            peak_df.index += 1
            st.dataframe(
                peak_df.style
                    .format(precision=2)
                    .background_gradient(subset=["Intensity (a.u.)"], cmap="YlGn"),
                use_container_width=True
            )
        else:
            st.info("No peaks detected for this sample within the selected range.")

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Full peak table download ────────────────────────────────────────────
    st.subheader("📥 Full Peak Table (All Samples)")
    all_rows = []
    for result in results:
        for pk in result["peaks"]:
            all_rows.append({
                "Sample":            result["sample"],
                "Raman Shift (cm⁻¹)": pk["shift"],
                "Intensity (a.u.)":  pk["intensity"],
                "FWHM (cm⁻¹)":       pk["fwhm"],
            })

    if all_rows:
        full_df = pd.DataFrame(all_rows)
        st.dataframe(full_df.style.format(precision=2), use_container_width=True)
        csv_data = full_df.to_csv(index=False)
        st.download_button(
            "⬇️  Download Peak Table (.csv)",
            csv_data,
            file_name="Raman_Peak_Table.csv",
            mime="text/csv"
        )

    # ── Overlay spectra ─────────────────────────────────────────────────────
    st.subheader("📊 Overlay — Baseline-Corrected Spectra")
    fig_ov, ax_ov = plt.subplots(figsize=(13, 5))
    apply_dark_theme(fig_ov, [ax_ov])
    for i, result in enumerate(results):
        col = COMP_PALETTE[i % len(COMP_PALETTE)]
        ax_ov.plot(result["x_work"], result["signal"], color=col,
                   linewidth=1.4, label=result["sample"], alpha=0.85)
    ax_ov.set_xlabel("Raman Shift (cm⁻¹)", fontsize=11)
    ax_ov.set_ylabel("Intensity (a.u.)", fontsize=11)
    ax_ov.legend(fontsize=9, facecolor='#1a1f2e', edgecolor='#2e3650', labelcolor='#ccd6f0')
    fig_ov.tight_layout()
    st.pyplot(fig_ov, use_container_width=True)
    plt.close(fig_ov)

    # ── Comparative peak map ────────────────────────────────────────────────
    st.subheader("🗺️ Peak Position Comparison")
    fig_cmp = plot_comparison(results, COMP_PALETTE)
    st.pyplot(fig_cmp, use_container_width=True)
    plt.close(fig_cmp)

else:
    st.markdown("""
    <div style="text-align:center; padding: 4rem 2rem; color:#5a6478;">
        <div style="font-size:3.5rem; margin-bottom:1rem;">🔬</div>
        <div style="font-family:'IBM Plex Mono',monospace; font-size:1.1rem; color:#7DF9C8;">
            Upload .txt Raman spectra files to begin analysis
        </div>
        <div style="margin-top:0.5rem; font-size:0.9rem;">
            Supports whitespace · comma · tab separated files
        </div>
    </div>
    """, unsafe_allow_html=True)
