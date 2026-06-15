"""
Raman Spectrum Analyzer
Fully automatic — no user-adjustable parameters.
Pipeline: Load → Rayleigh align → Wavelet denoise → ALS baseline →
          ROI extract → Candidate peaks → Lorentzian fit → Validate → Report
"""

import io
import warnings
import textwrap

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import pywt
from scipy import signal
from scipy.optimize import curve_fit
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import streamlit as st

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# FIXED INTERNAL PARAMETERS  (never exposed to user)
# ─────────────────────────────────────────────────────────────────────────────
P = dict(
    # Wavelet denoising
    wavelet        = "db4",
    wavelet_level  = 4,
    # ALS baseline
    als_lam        = 1e6,
    als_p          = 0.001,
    als_niter      = 15,
    # ROI
    roi_lo         = 0.0,
    roi_hi         = 550.0,
    # Matched Lorentzian filter
    filter_gamma   = 8.0,      # cm⁻¹ half-width of the filter kernel
    # Candidate detection
    prominence_frac = 0.03,    # fraction of signal range for prominence
    min_distance_cm = 4.0,     # minimum inter-peak distance in cm⁻¹
    # Lorentzian fit window
    fit_half_window_cm = 50.0,
    # Fit validation
    r2_threshold   = 0.85,
    gamma_min      = 1.0,
    gamma_max      = 100.0,
)

# ─────────────────────────────────────────────────────────────────────────────
# PALETTE & STYLE
# ─────────────────────────────────────────────────────────────────────────────
BG      = "#0B0F19"
PANEL   = "#131929"
BORDER  = "#1E2A40"
ACCENT  = "#38BDF8"   # sky blue — laser-line colour
GREEN   = "#34D399"
AMBER   = "#FBBF24"
RED     = "#F87171"
MUTED   = "#64748B"
TEXT    = "#CBD5E1"
WHITE   = "#F1F5F9"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    PANEL,
    "axes.edgecolor":    BORDER,
    "axes.labelcolor":   MUTED,
    "axes.titlecolor":   TEXT,
    "axes.titlesize":    9,
    "axes.labelsize":    8,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "xtick.labelsize":   7.5,
    "ytick.labelsize":   7.5,
    "grid.color":        BORDER,
    "grid.linewidth":    0.5,
    "legend.fontsize":   7.5,
    "legend.facecolor":  PANEL,
    "legend.edgecolor":  BORDER,
    "lines.linewidth":   1.1,
    "font.family":       "monospace",
})

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Raman Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600&family=Inter:wght@300;400;600&display=swap');

  html, body, [class*="css"] {{
      font-family: 'Inter', sans-serif;
      background: {BG};
      color: {TEXT};
  }}
  .stApp {{ background: {BG}; }}
  #MainMenu, footer, header {{ visibility: hidden; }}

  /* ── header bar ── */
  .app-header {{
      display: flex; align-items: center; gap: 18px;
      border-bottom: 1px solid {BORDER};
      padding: 22px 0 18px;
      margin-bottom: 30px;
  }}
  .app-header .laser-dot {{
      width: 10px; height: 10px; border-radius: 50%;
      background: {ACCENT};
      box-shadow: 0 0 12px {ACCENT}cc, 0 0 24px {ACCENT}66;
      flex-shrink: 0;
  }}
  .app-header h1 {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 1.2rem; font-weight: 600;
      color: {WHITE}; margin: 0; letter-spacing: 0.04em;
  }}
  .app-header .sub {{
      font-size: 0.7rem; color: {MUTED};
      font-family: 'JetBrains Mono', monospace;
      letter-spacing: 0.1em; text-transform: uppercase;
      margin-top: 2px;
  }}

  /* ── upload zone ── */
  div[data-testid="stFileUploader"] {{
      background: {PANEL};
      border: 1px dashed {BORDER};
      border-radius: 6px;
      padding: 6px;
      transition: border-color 0.2s;
  }}
  div[data-testid="stFileUploader"]:hover {{ border-color: {ACCENT}55; }}

  /* ── section label ── */
  .section-label {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.62rem; color: {MUTED};
      letter-spacing: 0.14em; text-transform: uppercase;
      margin: 28px 0 10px;
      display: flex; align-items: center; gap: 8px;
  }}
  .section-label::after {{
      content: ''; flex: 1; height: 1px; background: {BORDER};
  }}

  /* ── stat pills ── */
  .stat-row {{ display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 22px; }}
  .stat-pill {{
      background: {PANEL}; border: 1px solid {BORDER};
      border-radius: 5px; padding: 10px 18px;
      font-family: 'JetBrains Mono', monospace;
      min-width: 110px;
  }}
  .stat-pill .sv {{ font-size: 1.1rem; font-weight: 600; color: {WHITE}; }}
  .stat-pill .sk {{ font-size: 0.62rem; color: {MUTED}; text-transform: uppercase;
                    letter-spacing: 0.09em; margin-top: 3px; }}

  /* ── pipeline log ── */
  .log-scroll {{
      background: #07090F;
      border: 1px solid {BORDER}; border-radius: 5px;
      padding: 16px 18px;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.71rem; line-height: 1.95;
      color: {MUTED};
      max-height: 420px; overflow-y: auto;
      white-space: pre-wrap; word-break: break-all;
  }}
  .log-scroll .ok   {{ color: {GREEN}; }}
  .log-scroll .err  {{ color: {RED}; }}
  .log-scroll .info {{ color: {ACCENT}; }}
  .log-scroll .head {{ color: {WHITE}; font-weight: 600; }}
  .log-scroll .warn {{ color: {AMBER}; }}

  /* ── peak cards ── */
  .peak-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(210px,1fr)); gap: 12px; }}
  .peak-card {{
      background: {PANEL}; border: 1px solid {BORDER};
      border-top: 2px solid {GREEN};
      border-radius: 5px; padding: 14px 16px;
      font-family: 'JetBrains Mono', monospace;
  }}
  .peak-card .pc-n {{ font-size: 0.6rem; color: {MUTED}; letter-spacing: 0.12em;
                       text-transform: uppercase; margin-bottom: 5px; }}
  .peak-card .pc-pos {{ font-size: 1.25rem; font-weight: 600; color: {WHITE}; }}
  .peak-card .pc-unit {{ font-size: 0.65rem; color: {MUTED}; }}
  .peak-card .pc-row {{ font-size: 0.72rem; color: {MUTED}; margin-top: 2px; }}
  .peak-card .pc-r2  {{ font-size: 0.72rem; color: {GREEN}; margin-top: 5px; }}

  /* ── no-peaks message ── */
  .no-peaks {{
      background: {PANEL}; border: 1px solid {BORDER};
      border-radius: 5px; padding: 28px;
      text-align: center;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.78rem; color: {MUTED};
  }}

  /* ── table ── */
  .stDataFrame {{ font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; }}

  /* ── pipeline step badge ── */
  .step-badge {{
      display: inline-block;
      background: {ACCENT}18; border: 1px solid {ACCENT}44;
      color: {ACCENT}; border-radius: 3px;
      padding: 1px 7px; font-size: 0.62rem;
      font-family: 'JetBrains Mono', monospace;
      letter-spacing: 0.08em; margin-right: 6px;
  }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="app-header">
  <div class="laser-dot"></div>
  <div>
    <h1>Raman Spectrum Analyzer</h1>
    <div class="sub">Lorentzian · ALS Baseline · Wavelet · Fully Automatic</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

class Log:
    """Accumulates styled HTML log lines."""
    def __init__(self):
        self._lines = []

    def head(self, msg):   self._lines.append(f'<span class="head">▸ {msg}</span>')
    def info(self, msg):   self._lines.append(f'<span class="info">  {msg}</span>')
    def ok(self, msg):     self._lines.append(f'<span class="ok">  ✓ {msg}</span>')
    def warn(self, msg):   self._lines.append(f'<span class="warn">  ⚠ {msg}</span>')
    def err(self, msg):    self._lines.append(f'<span class="err">  ✗ {msg}</span>')
    def blank(self):       self._lines.append("")

    def html(self):
        return "\n".join(self._lines)


def load_spectrum(content: bytes, log: Log):
    """Step 1 — parse two-column spectrum file."""
    log.head("STEP 1 · Load Spectrum")
    text = content.decode("utf-8", errors="replace")
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "%", ";")):
            continue
        # try multiple separators
        for sep in ("\t", ",", ";", "  ", " "):
            parts = [p.strip() for p in line.split(sep) if p.strip()]
            if len(parts) >= 2:
                try:
                    x, y = float(parts[0]), float(parts[1])
                    rows.append((x, y))
                    break
                except ValueError:
                    continue
    if len(rows) < 10:
        raise ValueError(f"Only {len(rows)} numeric rows found — check file format.")
    arr = np.array(rows)
    x, y = arr[:, 0], arr[:, 1]
    log.info(f"Points loaded    : {len(x)}")
    log.info(f"x range          : {x.min():.2f} – {x.max():.2f} cm⁻¹")
    log.info(f"Intensity range  : {y.min():.2f} – {y.max():.2f}")
    log.ok("Spectrum loaded successfully")
    log.blank()
    return x, y


def rayleigh_align(x, y, log: Log):
    """Step 2 — shift x so argmax(y) → 0."""
    log.head("STEP 2 · Rayleigh Alignment")
    idx_max = np.argmax(y)
    x_R = x[idx_max]
    x_new = x - x_R
    log.info(f"Rayleigh peak at : {x_R:.3f} cm⁻¹  (index {idx_max})")
    log.info(f"Shift applied    : {-x_R:+.3f} cm⁻¹")
    log.ok("x-axis zeroed at Rayleigh peak")
    log.blank()
    return x_new, x_R


def wavelet_denoise(y, log: Log):
    """Step 3 — soft-threshold wavelet denoising."""
    log.head("STEP 3 · Wavelet Denoising")
    wv    = P["wavelet"]
    level = P["wavelet_level"]
    coeffs = pywt.wavedec(y, wv, level=level)
    # MAD estimate of noise sigma from finest-scale detail
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(max(len(y), 2)))
    coeffs_t = [coeffs[0]] + [
        pywt.threshold(c, uthresh, mode="soft") for c in coeffs[1:]
    ]
    y_den = pywt.waverec(coeffs_t, wv)[: len(y)]
    log.info(f"Wavelet          : {wv}")
    log.info(f"Decomposition    : level {level}")
    log.info(f"Noise σ (MAD)    : {sigma:.4f}")
    log.info(f"Universal thresh : {uthresh:.4f}")
    log.ok("Denoising complete")
    log.blank()
    return y_den, sigma


def als_baseline(y, log: Log):
    """Step 4 — asymmetric least squares baseline."""
    log.head("STEP 4 · ALS Baseline Correction")
    lam   = P["als_lam"]
    p     = P["als_p"]
    niter = P["als_niter"]
    n = len(y)
    D = diags([1, -2, 1], [0, 1, 2], shape=(n - 2, n))
    H = lam * D.T.dot(D)
    w = np.ones(n)
    z = y.copy()
    for _ in range(niter):
        W = diags(w, 0, shape=(n, n))
        Z = W + H
        z = spsolve(Z, w * y)
        w = np.where(y > z, p, 1 - p)
    corrected = y - z
    log.info(f"λ (smoothness)   : {lam:.0e}")
    log.info(f"p (asymmetry)    : {p}")
    log.info(f"Iterations       : {niter}")
    log.info(f"Baseline min/max : {z.min():.2f} / {z.max():.2f}")
    log.info(f"Corrected range  : {corrected.min():.2f} / {corrected.max():.2f}")
    log.ok("Baseline removed")
    log.blank()
    return z, corrected


def extract_roi(x, y_den, y_corr, baseline, log: Log):
    """Step 5 — keep 0–550 cm⁻¹."""
    log.head("STEP 5 · ROI Extraction  [0 – 550 cm⁻¹]")
    lo, hi = P["roi_lo"], P["roi_hi"]
    mask = (x >= lo) & (x <= hi)
    if mask.sum() < 5:
        log.err("Fewer than 5 points in ROI — check Rayleigh alignment")
        raise ValueError("ROI contains too few points.")
    log.info(f"Points retained  : {mask.sum()}  of  {len(x)}")
    log.ok(f"ROI extracted: {x[mask].min():.2f} – {x[mask].max():.2f} cm⁻¹")
    log.blank()
    return x[mask], y_den[mask], y_corr[mask], baseline[mask]


def lorentzian_kernel(x_span, gamma):
    k = gamma**2 / (x_span**2 + gamma**2)
    return k / k.sum()


def find_candidates(xr, yr_corr, log: Log):
    """Step 6 — liberal candidate generation for high recall."""
    log.head("STEP 6 · Candidate Peak Generation")
    dx   = float(np.median(np.diff(xr)))

    # Matched Lorentzian filter
    g    = P["filter_gamma"]
    half = max(3, int(5 * g / dx))
    kx   = np.arange(-half, half + 1) * dx
    kern = lorentzian_kernel(kx, g)
    yr_mf = np.convolve(yr_corr, kern, mode="same")
    log.info(f"Matched filter Γ : {g} cm⁻¹  (kernel pts: {len(kern)})")

    # Prominence threshold
    sig_range = float(yr_mf.max() - yr_mf.min())
    prom_abs  = P["prominence_frac"] * sig_range
    min_dist  = max(2, int(P["min_distance_cm"] / dx))
    log.info(f"Prominence floor : {prom_abs:.4f}  ({P['prominence_frac']*100:.0f}% of range)")
    log.info(f"Min distance     : {min_dist} pts  ({P['min_distance_cm']} cm⁻¹)")

    # Multi-scale detection
    cand_set = set()
    for extra_smooth in [0, 1, 2]:
        y_tmp = yr_mf.copy()
        if extra_smooth > 0:
            y_tmp = np.convolve(y_tmp,
                                np.ones(extra_smooth * 2 + 1) / (extra_smooth * 2 + 1),
                                mode="same")
        idxs, _ = signal.find_peaks(y_tmp, prominence=prom_abs * 0.5,
                                     distance=max(2, min_dist // 2))
        cand_set.update(idxs.tolist())

    # Also detect on raw corrected spectrum
    idxs_raw, _ = signal.find_peaks(yr_corr, prominence=prom_abs * 0.3,
                                     distance=max(2, min_dist // 2))
    cand_set.update(idxs_raw.tolist())

    cands = np.array(sorted(cand_set))
    log.info(f"Candidates found : {len(cands)}")
    if len(cands):
        positions = ", ".join(f"{xr[i]:.1f}" for i in cands)
        # wrap long line
        for chunk in textwrap.wrap(f"Positions (cm⁻¹): {positions}", width=72):
            log.info(chunk)
    log.blank()
    return cands, yr_mf


def lorentzian(x, A, x0, gamma):
    return A * gamma**2 / ((x - x0)**2 + gamma**2)


def fit_and_validate(xr, yr_corr, cand_idx, log: Log):
    """Steps 7 & 8 — fit Lorentzian to each candidate, then validate."""
    log.head("STEP 7 · Lorentzian Fitting  &  STEP 8 · Validation")
    dx      = float(np.median(np.diff(xr)))
    half_w  = max(10, int(P["fit_half_window_cm"] / dx))
    r2_thr  = P["r2_threshold"]
    g_min   = P["gamma_min"]
    g_max   = P["gamma_max"]
    roi_hi  = P["roi_hi"]

    accepted  = []
    rejected  = []

    for rank, idx in enumerate(cand_idx, 1):
        x0g  = xr[idx]
        Ag   = float(yr_corr[idx]) if yr_corr[idx] > 0 else 1.0
        lo   = max(0, idx - half_w)
        hi   = min(len(xr), idx + half_w)
        xw   = xr[lo:hi]
        yw   = yr_corr[lo:hi]

        reason = None
        popt   = None

        if len(xw) < 6:
            reason = "insufficient points in fit window"
        else:
            try:
                popt, _ = curve_fit(
                    lorentzian, xw, yw,
                    p0=[Ag, x0g, P["filter_gamma"]],
                    bounds=([0.0, 0.0, g_min], [np.inf, roi_hi, g_max]),
                    maxfev=8000,
                )
                A_f, x0_f, g_f = popt

                # Compute R²
                y_pred = lorentzian(xw, *popt)
                ss_res = np.sum((yw - y_pred)**2)
                ss_tot = np.sum((yw - yw.mean())**2)
                r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

                # Validate
                if A_f <= 0:
                    reason = "A ≤ 0 (unphysical amplitude)"
                elif not (g_min < g_f < g_max):
                    reason = f"unphysical width  Γ = {g_f:.2f} cm⁻¹  (allowed {g_min}–{g_max})"
                elif not (0.0 < x0_f <= roi_hi):
                    reason = f"position {x0_f:.2f} cm⁻¹ outside ROI"
                elif r2 < r2_thr:
                    reason = f"R² too low  ({r2:.3f} < {r2_thr})"

            except RuntimeError:
                reason = "fit did not converge"
            except Exception as e:
                reason = f"fit error: {e}"

        if reason is None and popt is not None:
            A_f, x0_f, g_f = popt
            fwhm = 2 * g_f
            y_pred = lorentzian(xw, *popt)
            ss_res = np.sum((yw - y_pred)**2)
            ss_tot = np.sum((yw - yw.mean())**2)
            r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            accepted.append(dict(
                position=round(float(x0_f), 2),
                amplitude=round(float(A_f), 3),
                gamma=round(float(g_f), 3),
                fwhm=round(float(fwhm), 3),
                r2=round(float(r2), 4),
            ))
            log.ok(
                f"Cand {rank:>2} @ {x0g:>7.2f} cm⁻¹  →  "
                f"x₀={x0_f:.2f}  A={A_f:.2f}  Γ={g_f:.2f}  FWHM={fwhm:.2f}  R²={r2:.4f}"
            )
        else:
            rejected.append(dict(position=float(x0g), reason=reason))
            log.err(f"Cand {rank:>2} @ {x0g:>7.2f} cm⁻¹  →  Rejected: {reason}")

    # Deduplicate accepted (merge within 3 cm⁻¹, keep higher R²)
    accepted.sort(key=lambda p: p["position"])
    deduped = []
    for pk in accepted:
        if deduped and abs(pk["position"] - deduped[-1]["position"]) < 3.0:
            if pk["r2"] > deduped[-1]["r2"]:
                deduped[-1] = pk
        else:
            deduped.append(pk)

    log.blank()
    log.head(f"STEP 9 · Final Peak Table")
    log.info(f"Candidates tested : {len(cand_idx)}")
    log.info(f"Rejected          : {len(rejected)}")
    log.ok(  f"Accepted peaks    : {len(deduped)}")
    log.blank()
    return deduped, rejected


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def make_plots(x_raw, y_raw, x, y, y_den, baseline, xr, yr_corr,
               yr_mf, cand_idx, peaks, fname):
    """Build the 8-panel figure."""
    fig = plt.figure(figsize=(14, 18), facecolor=BG)
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.35)

    def ax_style(ax, title, xlabel="Raman Shift (cm⁻¹)", ylabel="Intensity"):
        ax.set_title(title, fontsize=8.5, color=TEXT, loc="left",
                     fontfamily="monospace", pad=6)
        ax.set_xlabel(xlabel, fontsize=7.5)
        ax.set_ylabel(ylabel, fontsize=7.5)
        ax.grid(True, alpha=0.5)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)

    roi_lo, roi_hi = P["roi_lo"], P["roi_hi"]

    # ── 1. Raw spectrum ──────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(x_raw, y_raw, color=ACCENT, lw=0.9, alpha=0.85)
    ax1.axvline(x_raw[np.argmax(y_raw)], color=AMBER, lw=1, ls="--", alpha=0.7,
                label=f"Rayleigh @ {x_raw[np.argmax(y_raw)]:.1f}")
    ax1.legend()
    ax_style(ax1, "① Raw Spectrum")

    # ── 2. Rayleigh-aligned ──────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(x, y, color=ACCENT, lw=0.9, alpha=0.85)
    ax2.axvline(0, color=AMBER, lw=1, ls="--", alpha=0.7, label="0 cm⁻¹")
    ax2.axvspan(roi_lo, roi_hi, alpha=0.06, color=GREEN, label="ROI")
    ax2.legend()
    ax_style(ax2, "② Rayleigh-Aligned Spectrum")

    # ── 3. Denoised ─────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    mask_roi = (x >= roi_lo) & (x <= roi_hi)
    ax3.plot(x[mask_roi], y[mask_roi],     color=ACCENT, lw=0.8, alpha=0.45, label="aligned")
    ax3.plot(x[mask_roi], y_den[mask_roi], color=WHITE,  lw=1.0, alpha=0.9,  label="denoised")
    ax3.legend()
    ax_style(ax3, "③ Wavelet Denoised  (ROI)")

    # ── 4. ALS baseline ─────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(xr, yr_corr + baseline[mask_roi][: len(xr)], color=WHITE, lw=0.9,
             alpha=0.6, label="denoised")
    ax4.plot(xr, baseline[mask_roi][: len(xr)], color=AMBER, lw=1.2,
             ls="--", label="ALS baseline")
    ax4.legend()
    ax_style(ax4, "④ ALS Baseline  (ROI)")

    # ── 5. Baseline-corrected ────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(xr, yr_corr, color=ACCENT, lw=1.0)
    ax5.axhline(0, color=BORDER, lw=0.8, ls=":")
    ax_style(ax5, "⑤ Baseline-Corrected Spectrum")

    # ── 6. Candidate peaks ───────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(xr, yr_mf, color=ACCENT, lw=0.9, alpha=0.7, label="matched filter")
    if len(cand_idx):
        ax6.scatter(xr[cand_idx], yr_mf[cand_idx],
                    color=AMBER, s=30, zorder=5, label=f"{len(cand_idx)} candidates")
    ax6.legend()
    ax_style(ax6, "⑥ Candidate Peaks  (matched-filter output)")

    # ── 7. Lorentzian fits ───────────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[3, 0])
    ax7.plot(xr, yr_corr, color=ACCENT, lw=0.9, alpha=0.5, label="corrected")
    if peaks:
        xf    = np.linspace(roi_lo, roi_hi, 3000)
        total = np.zeros_like(xf)
        for pk in peaks:
            comp  = lorentzian(xf, pk["amplitude"], pk["position"], pk["gamma"])
            total += comp
            ax7.plot(xf, comp, color=GREEN, lw=0.7, alpha=0.45)
        ax7.plot(xf, total, color=GREEN, lw=1.4, label="sum of fits")
    ax7.legend()
    ax_style(ax7, "⑦ Lorentzian Fits")

    # ── 8. Accepted peaks ────────────────────────────────────────────────────
    ax8 = fig.add_subplot(gs[3, 1])
    ax8.plot(xr, yr_corr, color=ACCENT, lw=0.9, alpha=0.6)
    if peaks:
        for pk in peaks:
            ax8.axvline(pk["position"], color=GREEN, lw=0.7, alpha=0.4, ls=":")
        px = [pk["position"] for pk in peaks]
        py = [lorentzian(pk["position"], pk["amplitude"],
                          pk["position"], pk["gamma"]) for pk in peaks]
        ax8.scatter(px, py, color=GREEN, s=40, zorder=6,
                    label=f"{len(peaks)} accepted")
        for pk in peaks:
            yv = lorentzian(pk["position"], pk["amplitude"],
                            pk["position"], pk["gamma"])
            ax8.annotate(
                f"{pk['position']:.0f}",
                xy=(pk["position"], yv),
                xytext=(0, 7), textcoords="offset points",
                ha="center", fontsize=6.2, color=GREEN,
                fontfamily="monospace",
            )
    ax8.legend()
    ax_style(ax8, f"⑧ Accepted Peaks  ·  {fname}")

    fig.suptitle("", fontsize=1)   # keeps tight_layout happy
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT MAIN
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-label">Upload spectrum file</div>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Two-column file · Raman Shift & Intensity · .txt  .csv  .dat",
    type=["txt", "csv", "dat", "tsv"],
    accept_multiple_files=False,
    label_visibility="visible",
)

if not uploaded:
    st.markdown(f"""
    <div style="text-align:center; padding: 70px 0 50px;
                font-family:'JetBrains Mono',monospace;
                font-size:0.8rem; color:{MUTED};">
      drop a spectrum file above to begin analysis
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── RUN PIPELINE ─────────────────────────────────────────────────────────────
content = uploaded.read()
fname   = uploaded.name
log     = Log()

with st.spinner("Running analysis pipeline…"):
    try:
        x_raw, y_raw                     = load_spectrum(content, log)
        x, x_R                           = rayleigh_align(x_raw, y_raw, log)
        y_den, sigma                      = wavelet_denoise(y_raw, log)
        baseline, y_corr                  = als_baseline(y_den, log)
        xr, yr_den, yr_corr, bl_roi       = extract_roi(x, y_den, y_corr, baseline, log)
        cand_idx, yr_mf                   = find_candidates(xr, yr_corr, log)
        peaks, rejected                   = fit_and_validate(xr, yr_corr, cand_idx, log)
        pipeline_ok = True
    except Exception as exc:
        log.err(f"Pipeline aborted: {exc}")
        pipeline_ok = False

# ── STAT PILLS ───────────────────────────────────────────────────────────────
if pipeline_ok:
    n_pts  = len(x_raw)
    n_cand = len(cand_idx)
    n_acc  = len(peaks)
    n_rej  = len(rejected)
    st.markdown(f"""
    <div class="stat-row">
      <div class="stat-pill"><div class="sv">{n_pts:,}</div><div class="sk">data points</div></div>
      <div class="stat-pill"><div class="sv">{n_cand}</div><div class="sk">candidates</div></div>
      <div class="stat-pill"><div class="sv" style="color:{GREEN}">{n_acc}</div><div class="sk">accepted peaks</div></div>
      <div class="stat-pill"><div class="sv" style="color:{RED}">{n_rej}</div><div class="sk">rejected</div></div>
      <div class="stat-pill"><div class="sv">{sigma:.2f}</div><div class="sk">noise σ</div></div>
      <div class="stat-pill"><div class="sv">{x_R:.1f} cm⁻¹</div><div class="sk">Rayleigh shift</div></div>
    </div>
    """, unsafe_allow_html=True)

# ── VISUALISATIONS ────────────────────────────────────────────────────────────
if pipeline_ok:
    st.markdown('<div class="section-label">Step 10 · Visualizations</div>',
                unsafe_allow_html=True)
    fig = make_plots(x_raw, y_raw, x, y_raw, y_den, baseline,
                     xr, yr_corr, yr_mf, cand_idx, peaks, fname)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ── PEAK CARDS + TABLE ────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Accepted peaks</div>', unsafe_allow_html=True)

if not pipeline_ok or not peaks:
    st.markdown(f'<div class="no-peaks">No peaks passed all validation gates — '
                f'try a different file or check that the spectrum is within 0–550 cm⁻¹ after Rayleigh alignment.</div>',
                unsafe_allow_html=True)
else:
    # Cards
    cards_html = '<div class="peak-grid">'
    for i, pk in enumerate(peaks, 1):
        cards_html += f"""
        <div class="peak-card">
          <div class="pc-n">peak {i:02d}</div>
          <div class="pc-pos">{pk['position']} <span class="pc-unit">cm⁻¹</span></div>
          <div class="pc-row">Amplitude &nbsp;{pk['amplitude']}</div>
          <div class="pc-row">Γ  &nbsp;{pk['gamma']} cm⁻¹</div>
          <div class="pc-row">FWHM &nbsp;{pk['fwhm']} cm⁻¹</div>
          <div class="pc-r2">R² = {pk['r2']:.4f}</div>
        </div>"""
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)

    # Table
    st.markdown('<div class="section-label">Peak table</div>', unsafe_allow_html=True)
    df = pd.DataFrame(peaks).rename(columns={
        "position": "Position (cm⁻¹)",
        "amplitude": "Amplitude",
        "gamma": "Γ (cm⁻¹)",
        "fwhm": "FWHM (cm⁻¹)",
        "r2": "R²",
    })
    df.index = [f"Peak {i}" for i in range(1, len(df) + 1)]
    st.dataframe(df, use_container_width=True)

    # Download
    csv_bytes = df.to_csv().encode()
    st.download_button(
        "↓ Export peaks as CSV",
        csv_bytes,
        file_name=f"{fname.rsplit('.', 1)[0]}_raman_peaks.csv",
        mime="text/csv",
    )

# ── PIPELINE LOG ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Pipeline log</div>', unsafe_allow_html=True)
st.markdown(f'<div class="log-scroll">{log.html()}</div>', unsafe_allow_html=True)
