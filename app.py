import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import signal
from scipy.optimize import curve_fit
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import pywt
import io
import warnings

warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Raman Peak Analyzer",
    page_icon="🔬",
    layout="wide",
)

# ── Minimal dark-scientific CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Background */
.stApp { background: #0d1117; }
section[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #1e2530; }

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* Top bar title */
.top-bar {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 28px 0 20px 0;
    border-bottom: 1px solid #1e2530;
    margin-bottom: 28px;
}
.top-bar h1 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.35rem;
    font-weight: 600;
    color: #e2e8f0;
    letter-spacing: 0.04em;
    margin: 0;
}
.top-bar .dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #4ade80;
    flex-shrink: 0;
    box-shadow: 0 0 8px #4ade8099;
}
.top-bar .sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #4a5568;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* Upload zone */
div[data-testid="stFileUploader"] {
    background: #111827;
    border: 1px dashed #2d3748;
    border-radius: 6px;
    padding: 8px;
}
div[data-testid="stFileUploader"]:hover {
    border-color: #4a5568;
}

/* Peak card */
.peak-card {
    background: #111827;
    border: 1px solid #1e2530;
    border-left: 3px solid #4ade80;
    border-radius: 4px;
    padding: 14px 18px;
    margin-bottom: 10px;
    font-family: 'IBM Plex Mono', monospace;
}
.peak-card .pk-num {
    font-size: 0.65rem;
    color: #4a5568;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 6px;
}
.peak-card .pk-pos {
    font-size: 1.15rem;
    font-weight: 600;
    color: #e2e8f0;
}
.peak-card .pk-meta {
    font-size: 0.75rem;
    color: #718096;
    margin-top: 4px;
    line-height: 1.7;
}
.peak-card .pk-r2 {
    font-size: 0.75rem;
    color: #4ade80;
    margin-top: 4px;
}

/* Log box */
.log-box {
    background: #080c12;
    border: 1px solid #1e2530;
    border-radius: 4px;
    padding: 16px 18px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.73rem;
    color: #4a5568;
    line-height: 1.9;
    white-space: pre-wrap;
    max-height: 340px;
    overflow-y: auto;
}
.log-box .lg-ok { color: #4ade80; }
.log-box .lg-rej { color: #f87171; }
.log-box .lg-info { color: #60a5fa; }
.log-box .lg-head { color: #a0aec0; }

/* Section label */
.sec-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: #4a5568;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 10px;
    margin-top: 24px;
}

/* Metric pill */
.pill-row { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 20px; }
.pill {
    background: #111827;
    border: 1px solid #1e2530;
    border-radius: 4px;
    padding: 10px 16px;
    font-family: 'IBM Plex Mono', monospace;
}
.pill .pv { font-size: 1.05rem; font-weight: 600; color: #e2e8f0; }
.pill .pl { font-size: 0.65rem; color: #4a5568; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 2px; }

/* Matplotlib figure border */
.stFigure { border: 1px solid #1e2530; border-radius: 4px; }

/* File tab */
.file-tab-active {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #4ade80;
    border-bottom: 2px solid #4ade80;
    padding-bottom: 4px;
}

/* Selectbox, slider */
div[data-baseweb="select"] { background: #111827 !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-bar">
  <div class="dot"></div>
  <div>
    <h1>Raman Peak Analyzer</h1>
    <div class="sub">Lorentzian · ALS Baseline · Wavelet Denoising</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Pipeline functions
# ══════════════════════════════════════════════════════════════════════════════

def load_spectrum(content: bytes, fname: str):
    """Parse spectrum file — handles space/tab/comma delimiters, skips headers."""
    text = content.decode("utf-8", errors="replace")
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for sep in ["\t", ",", "  ", " "]:
            parts = line.split(sep)
            parts = [p.strip() for p in parts if p.strip()]
            if len(parts) >= 2:
                try:
                    x, y = float(parts[0]), float(parts[1])
                    rows.append((x, y))
                    break
                except ValueError:
                    break
    if not rows:
        raise ValueError("Could not parse any numeric rows from file.")
    arr = np.array(rows)
    return arr[:, 0], arr[:, 1]


def rayleigh_align(x, y):
    shift = x[np.argmax(y)]
    return x - shift, shift


def wavelet_denoise(y, wavelet="db4", level=4):
    coeffs = pywt.wavedec(y, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(y)))
    coeffs_t = [pywt.threshold(c, uthresh, mode="soft") for c in coeffs]
    return pywt.waverec(coeffs_t, wavelet)[: len(y)], sigma


def als_baseline(y, lam=1e6, p=0.01, niter=10):
    n = len(y)
    D = diags([1, -2, 1], [0, 1, 2], shape=(n - 2, n))
    H = lam * D.T.dot(D)
    w = np.ones(n)
    for _ in range(niter):
        W = diags(w, 0)
        Z = W + H
        z = spsolve(Z, w * y)
        w = np.where(y > z, p, 1 - p)
    return z


def lorentzian(x, A, x0, gamma):
    return A * gamma**2 / ((x - x0)**2 + gamma**2)


def lorentzian_filter(x, s, gamma=5.0):
    dx = np.mean(np.diff(x))
    half = int(50 / dx)
    kx = np.arange(-half, half + 1) * dx
    kernel = gamma**2 / (kx**2 + gamma**2)
    kernel /= kernel.sum()
    return np.convolve(s, kernel, mode="same")


def run_pipeline(x_raw, y_raw, roi_max=550, lam=1e6, r2_thresh=0.90,
                 min_gamma=1.0, max_gamma=100.0, filter_gamma=5.0,
                 min_prominence=0.05):
    log = []

    def L(msg, kind="info"):
        log.append((kind, msg))

    L(f"Loaded {len(x_raw)} data points", "head")

    # Step 2 — Rayleigh
    x, rayleigh_pos = rayleigh_align(x_raw, y_raw)
    y = y_raw.copy()
    L(f"Rayleigh peak at {rayleigh_pos:.1f} → shifted to 0 cm⁻¹", "info")

    # Step 3 — Wavelet
    y_den, sigma = wavelet_denoise(y)
    L(f"Wavelet db4 / level 4 · noise σ = {sigma:.2f}", "info")

    # Step 4 — ALS baseline
    baseline = als_baseline(y_den, lam=lam)
    y_corr = y_den - baseline
    L(f"ALS baseline removed · λ = {lam:.0e}", "info")

    # Step 5 — ROI
    mask = (x >= 0) & (x <= roi_max)
    xr, yr = x[mask], y_corr[mask]
    L(f"ROI 0–{roi_max} cm⁻¹ · {mask.sum()} points retained", "info")

    if len(xr) < 10:
        L("Insufficient points in ROI — aborting.", "rej")
        return [], log, x, y, y_den, baseline, xr, yr

    # Step 6 — Lorentzian matched filter
    yr_filt = lorentzian_filter(xr, yr, gamma=filter_gamma)
    L(f"Lorentzian matched filter · Γ = {filter_gamma} cm⁻¹", "info")

    # Step 7 — Candidate peaks
    yr_norm = yr_filt - yr_filt.min()
    peak_range = yr_norm.max() - yr_norm.min()
    prom_abs = min_prominence * peak_range if peak_range > 0 else 0
    cand_idx, props = signal.find_peaks(
        yr_norm,
        prominence=prom_abs,
        distance=max(3, int(5 / max(np.mean(np.diff(xr)), 1e-6))),
    )
    L(f"{len(cand_idx)} candidate peaks detected", "head")

    # Steps 8–11 — Fit each candidate
    accepted = []
    dx = np.mean(np.diff(xr))
    win_pts = max(20, int(40 / dx))

    for i, idx in enumerate(cand_idx):
        x0_guess = xr[idx]
        A_guess = yr[idx] if yr[idx] > 0 else yr_filt[idx]
        lo = max(0, idx - win_pts)
        hi = min(len(xr) - 1, idx + win_pts)
        xw, yw = xr[lo:hi], yr[lo:hi]

        if len(xw) < 5:
            L(f"  Cand {i+1} @ {x0_guess:.1f}: rejected — too few points", "rej")
            continue

        try:
            popt, _ = curve_fit(
                lorentzian, xw, yw,
                p0=[max(A_guess, 1e-3), x0_guess, filter_gamma],
                bounds=([0, 0, min_gamma], [np.inf, roi_max, max_gamma]),
                maxfev=4000,
            )
            A_fit, x0_fit, g_fit = popt

            # Convergence / bounds already enforced by curve_fit bounds
            # R²
            y_pred = lorentzian(xw, *popt)
            ss_res = np.sum((yw - y_pred) ** 2)
            ss_tot = np.sum((yw - yw.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            if A_fit <= 0:
                L(f"  Cand {i+1} @ {x0_guess:.1f}: rejected — A ≤ 0", "rej")
                continue
            if not (0 < x0_fit < roi_max):
                L(f"  Cand {i+1} @ {x0_guess:.1f}: rejected — position OOB", "rej")
                continue
            if r2 < r2_thresh:
                L(f"  Cand {i+1} @ {x0_guess:.1f}: rejected — R² = {r2:.2f}", "rej")
                continue

            fwhm = 2 * g_fit
            accepted.append({
                "position": round(x0_fit, 2),
                "amplitude": round(A_fit, 2),
                "gamma": round(g_fit, 2),
                "fwhm": round(fwhm, 2),
                "r2": round(r2, 4),
            })
            L(f"  Cand {i+1} @ {x0_fit:.1f} cm⁻¹  R² = {r2:.3f}  FWHM = {fwhm:.1f}", "ok")

        except RuntimeError:
            L(f"  Cand {i+1} @ {x0_guess:.1f}: rejected — fit did not converge", "rej")

    # Deduplicate (merge peaks closer than 3 cm⁻¹)
    accepted.sort(key=lambda p: p["position"])
    deduped = []
    for pk in accepted:
        if deduped and abs(pk["position"] - deduped[-1]["position"]) < 3:
            if pk["r2"] > deduped[-1]["r2"]:
                deduped[-1] = pk
        else:
            deduped.append(pk)

    L(f"\n{len(deduped)} peaks accepted", "head")
    return deduped, log, x, y, y_den, baseline, xr, yr


# ══════════════════════════════════════════════════════════════════════════════
# Plot
# ══════════════════════════════════════════════════════════════════════════════
DARK_BG = "#0d1117"
PANEL_BG = "#111827"
GRID_C   = "#1e2530"
TEXT_C   = "#a0aec0"
GREEN    = "#4ade80"
BLUE     = "#60a5fa"
ORANGE   = "#fb923c"
RED      = "#f87171"


def make_figure(x, y, y_den, baseline, xr, yr, peaks, roi_max, fname):
    fig, axes = plt.subplots(3, 1, figsize=(10, 9),
                             gridspec_kw={"hspace": 0.55},
                             facecolor=DARK_BG)

    def style_ax(ax, title):
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TEXT_C, labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID_C)
        ax.xaxis.label.set_color(TEXT_C)
        ax.yaxis.label.set_color(TEXT_C)
        ax.set_title(title, color=TEXT_C, fontsize=8.5,
                     fontfamily="monospace", loc="left", pad=6)
        ax.grid(color=GRID_C, linewidth=0.5)

    # Panel 1 — raw + baseline
    ax0 = axes[0]
    xp = x[(x >= 0) & (x <= roi_max)]
    yp = y[(x >= 0) & (x <= roi_max)]
    yb = y_den[(x >= 0) & (x <= roi_max)]
    bl = baseline[(x >= 0) & (x <= roi_max)]
    ax0.plot(xp, yp,   color=BLUE,   lw=0.9, alpha=0.6, label="raw")
    ax0.plot(xp, yb,   color=TEXT_C, lw=0.9, alpha=0.8, label="denoised")
    ax0.plot(xp, bl,   color=ORANGE, lw=1.0, ls="--",   label="ALS baseline")
    ax0.set_xlim(0, roi_max)
    ax0.set_xlabel("Raman Shift (cm⁻¹)", fontsize=8)
    ax0.legend(fontsize=7, facecolor=PANEL_BG, edgecolor=GRID_C,
               labelcolor=TEXT_C, loc="upper right")
    style_ax(ax0, "RAW  ·  DENOISED  ·  BASELINE")

    # Panel 2 — baseline-corrected + Lorentzian fits
    ax1 = axes[1]
    ax1.plot(xr, yr, color=BLUE, lw=0.9, alpha=0.7, label="corrected")
    if peaks:
        xf = np.linspace(0, roi_max, 2000)
        total = np.zeros_like(xf)
        for pk in peaks:
            comp = lorentzian(xf, pk["amplitude"], pk["position"], pk["gamma"])
            total += comp
            ax1.plot(xf, comp, color=GREEN, lw=0.8, alpha=0.5)
        ax1.plot(xf, total, color=GREEN, lw=1.2, label="Lorentzian fits")
        for pk in peaks:
            ax1.axvline(pk["position"], color=GREEN, lw=0.6, alpha=0.4, ls=":")
    ax1.set_xlim(0, roi_max)
    ax1.set_xlabel("Raman Shift (cm⁻¹)", fontsize=8)
    ax1.legend(fontsize=7, facecolor=PANEL_BG, edgecolor=GRID_C,
               labelcolor=TEXT_C, loc="upper right")
    style_ax(ax1, "BASELINE-CORRECTED  ·  LORENTZIAN FITS")

    # Panel 3 — peak markers on corrected
    ax2 = axes[2]
    ax2.plot(xr, yr, color=BLUE, lw=0.9, alpha=0.7)
    if peaks:
        px = [p["position"] for p in peaks]
        py_vals = [lorentzian(p["position"], p["amplitude"], p["position"], p["gamma"])
                   for p in peaks]
        ax2.scatter(px, py_vals, color=GREEN, s=35, zorder=5, label=f"{len(peaks)} peaks")
        for pk in peaks:
            ax2.annotate(
                f"{pk['position']:.0f}",
                xy=(pk["position"], lorentzian(pk["position"], pk["amplitude"], pk["position"], pk["gamma"])),
                xytext=(0, 8), textcoords="offset points",
                ha="center", fontsize=6.5, color=GREEN,
                fontfamily="monospace",
            )
    ax2.set_xlim(0, roi_max)
    ax2.set_xlabel("Raman Shift (cm⁻¹)", fontsize=8)
    ax2.legend(fontsize=7, facecolor=PANEL_BG, edgecolor=GRID_C,
               labelcolor=TEXT_C, loc="upper right")
    style_ax(ax2, f"ACCEPTED PEAKS  ·  {fname}")

    fig.patch.set_facecolor(DARK_BG)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar — parameters
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sec-label">Parameters</div>', unsafe_allow_html=True)
    roi_max = st.slider("ROI upper limit (cm⁻¹)", 100, 1000, 550, 10)
    r2_thresh = st.slider("R² threshold", 0.70, 0.99, 0.90, 0.01)
    filter_gamma = st.slider("Matched filter Γ (cm⁻¹)", 1, 20, 5)
    min_gamma = st.slider("Min peak width Γ (cm⁻¹)", 0.5, 10.0, 1.0, 0.5)
    max_gamma = st.slider("Max peak width Γ (cm⁻¹)", 20, 200, 100, 5)
    lam_exp = st.slider("ALS λ (10^x)", 3, 8, 6)
    lam = 10 ** lam_exp
    min_prom = st.slider("Min peak prominence (%)", 1, 30, 5) / 100

    st.markdown('<div class="sec-label" style="margin-top:28px;">About</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<span style="font-family:\'IBM Plex Mono\',monospace;font-size:0.7rem;'
        'color:#4a5568;line-height:1.8;">Wavelet · ALS · Lorentzian matched filter · '
        'curve_fit convergence · R² gate</span>',
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# Main — upload
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-label">Drop spectrum files</div>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Accepts .txt / .csv / .dat  —  two columns: shift, intensity",
    type=["txt", "csv", "dat", "tsv"],
    accept_multiple_files=True,
    label_visibility="visible",
)

if not uploaded:
    st.markdown(
        '<p style="font-family:\'IBM Plex Mono\',monospace;font-size:0.78rem;'
        'color:#2d3748;margin-top:48px;text-align:center;">'
        'no files loaded yet</p>',
        unsafe_allow_html=True,
    )
    st.stop()

# ── Per-file tab selector ─────────────────────────────────────────────────────
fnames = [f.name for f in uploaded]
if len(fnames) == 1:
    sel_idx = 0
else:
    sel_idx = st.selectbox(
        "File",
        range(len(fnames)),
        format_func=lambda i: fnames[i],
        label_visibility="collapsed",
    )

file = uploaded[sel_idx]
content = file.read()

# ── Run pipeline ──────────────────────────────────────────────────────────────
with st.spinner("Running pipeline…"):
    try:
        x_raw, y_raw = load_spectrum(content, file.name)
        peaks, log, x, y, y_den, baseline, xr, yr = run_pipeline(
            x_raw, y_raw,
            roi_max=roi_max, lam=lam, r2_thresh=r2_thresh,
            min_gamma=min_gamma, max_gamma=max_gamma,
            filter_gamma=filter_gamma, min_prominence=min_prom,
        )
        err = None
    except Exception as e:
        peaks, log, err = [], [], str(e)
        x = y = y_den = baseline = xr = yr = np.array([0, 1])

if err:
    st.error(f"Parse error: {err}")
    st.stop()

# ── Metrics row ───────────────────────────────────────────────────────────────
n_cand = sum(1 for _, m in log if "candidate" in m.lower() and "detected" in m.lower())
total_cand_val = "—"
for _, m in log:
    if "candidate peaks detected" in m:
        total_cand_val = m.split()[0]
        break

st.markdown(f"""
<div class="pill-row">
  <div class="pill"><div class="pv">{len(x_raw)}</div><div class="pl">data points</div></div>
  <div class="pill"><div class="pv">{total_cand_val}</div><div class="pl">candidates</div></div>
  <div class="pill"><div class="pv" style="color:#4ade80">{len(peaks)}</div><div class="pl">accepted peaks</div></div>
  <div class="pill"><div class="pv">{roi_max} cm⁻¹</div><div class="pl">ROI</div></div>
  <div class="pill"><div class="pv">{r2_thresh:.2f}</div><div class="pl">R² min</div></div>
</div>
""", unsafe_allow_html=True)

# ── Figure ────────────────────────────────────────────────────────────────────
fig = make_figure(x, y, y_den, baseline, xr, yr, peaks, roi_max, file.name)
st.pyplot(fig, use_container_width=True)
plt.close(fig)

# ── Two-column: peaks | log ───────────────────────────────────────────────────
col_pk, col_log = st.columns([1, 1], gap="medium")

with col_pk:
    st.markdown('<div class="sec-label">Accepted peaks</div>', unsafe_allow_html=True)
    if not peaks:
        st.markdown(
            '<p style="font-family:\'IBM Plex Mono\',monospace;font-size:0.78rem;'
            'color:#4a5568;">No peaks passed all gates.</p>',
            unsafe_allow_html=True,
        )
    else:
        for i, pk in enumerate(peaks, 1):
            st.markdown(f"""
<div class="peak-card">
  <div class="pk-num">peak {i:02d}</div>
  <div class="pk-pos">{pk['position']} cm⁻¹</div>
  <div class="pk-meta">
    Amplitude &nbsp;{pk['amplitude']}<br>
    Γ (half-width) &nbsp;{pk['gamma']} cm⁻¹<br>
    FWHM &nbsp;{pk['fwhm']} cm⁻¹
  </div>
  <div class="pk-r2">R² = {pk['r2']:.4f}</div>
</div>
""", unsafe_allow_html=True)

        # CSV download
        df = pd.DataFrame(peaks)
        csv_bytes = df.to_csv(index=False).encode()
        st.download_button(
            "↓ Download peaks CSV",
            csv_bytes,
            file_name=f"{file.name.rsplit('.',1)[0]}_peaks.csv",
            mime="text/csv",
        )

with col_log:
    st.markdown('<div class="sec-label">Pipeline log</div>', unsafe_allow_html=True)
    kind_map = {
        "head": '<span class="lg-head">',
        "ok":   '<span class="lg-ok">',
        "rej":  '<span class="lg-rej">',
        "info": '<span class="lg-info">',
    }
    html_log = ""
    for kind, msg in log:
        span = kind_map.get(kind, "<span>")
        html_log += f"{span}{msg}</span>\n"
    st.markdown(f'<div class="log-box">{html_log}</div>', unsafe_allow_html=True)
