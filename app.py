"""
Raman Peak Analyzer — Discovery Mode
Upload → Process → Inspect. No rejection gates. No thresholds visible.
Every detected peak is shown. Lorentzian fit info is supplementary only.
"""

import io
import warnings
import textwrap

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pywt
from scipy import signal
from scipy.optimize import curve_fit
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import streamlit as st

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# FIXED INTERNAL PARAMETERS — never shown to the user
# ─────────────────────────────────────────────────────────────────────────────
_P = dict(
    wavelet       = "db4",
    wavelet_level = 4,
    als_lam       = 1e6,
    als_p         = 0.001,
    als_niter     = 15,
    roi_lo        = 0.0,
    roi_hi        = 550.0,
    filter_gamma  = 8.0,
    prom_frac     = 0.025,
    min_dist_cm   = 3.5,
    fit_half_cm   = 55.0,
    gamma_min     = 1.0,
    gamma_max     = 120.0,
)

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR TOKENS
# ─────────────────────────────────────────────────────────────────────────────
BG     = "#080C14"
PANEL  = "#0F1620"
BORDER = "#1A2235"
ACCENT = "#5B9CF6"
GREEN  = "#3DD68C"
AMBER  = "#F5A623"
RED    = "#E05C5C"
MUTED  = "#4A607A"
TEXT   = "#A8BCCF"
WHITE  = "#E8F0F8"

plt.rcParams.update({
    "figure.facecolor": BG,   "axes.facecolor":  PANEL,
    "axes.edgecolor":   BORDER,"axes.labelcolor": MUTED,
    "axes.titlecolor":  TEXT,  "axes.titlesize":  8.5,
    "axes.labelsize":   8,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "grid.color": BORDER, "grid.linewidth": 0.45,
    "legend.fontsize": 7.5, "legend.facecolor": PANEL,
    "legend.edgecolor": BORDER,
    "lines.linewidth": 1.1, "font.family": "monospace",
})

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG + CSS
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Raman Analyzer", page_icon="🔬",
                   layout="centered", initial_sidebar_state="collapsed")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {{
  font-family: 'JetBrains Mono', monospace;
  background: {BG}; color: {TEXT};
}}
.stApp {{ background: {BG}; }}
#MainMenu, footer, header {{ visibility: hidden; }}
section[data-testid="stSidebar"] {{ display: none; }}

/* ── wordmark ── */
.wordmark {{
  padding: 32px 0 24px;
  border-bottom: 1px solid {BORDER};
  margin-bottom: 32px;
}}
.wordmark h1 {{
  font-size: 1.05rem; font-weight: 600;
  color: {WHITE}; letter-spacing: 0.06em;
  margin: 0 0 4px;
}}
.wordmark .sub {{
  font-size: 0.62rem; color: {MUTED};
  letter-spacing: 0.13em; text-transform: uppercase;
}}
.wordmark .dot {{
  display: inline-block; width: 7px; height: 7px;
  border-radius: 50%; background: {ACCENT};
  margin-right: 10px; margin-bottom: -1px;
  box-shadow: 0 0 8px {ACCENT}bb;
}}

/* ── upload ── */
div[data-testid="stFileUploader"] {{
  background: {PANEL}; border: 1px dashed {BORDER};
  border-radius: 5px; padding: 4px;
}}

/* ── divider label ── */
.divlabel {{
  font-size: 0.6rem; color: {MUTED};
  letter-spacing: 0.16em; text-transform: uppercase;
  display: flex; align-items: center; gap: 10px;
  margin: 28px 0 14px;
}}
.divlabel::before, .divlabel::after {{
  content:''; flex:1; height:1px; background:{BORDER};
}}

/* ── processing log ── */
.logbox {{
  background: #05080E;
  border: 1px solid {BORDER}; border-radius: 5px;
  padding: 14px 16px;
  font-size: 0.7rem; line-height: 2.0;
  color: {MUTED};
  white-space: pre-wrap;
}}
.logbox .lv   {{ color:{ACCENT}; }}
.logbox .lok  {{ color:{GREEN};  }}
.logbox .lwarn{{ color:{AMBER};  }}

/* ── peak table ── */
.ptable {{
  width: 100%; border-collapse: collapse;
  font-size: 0.75rem;
}}
.ptable thead tr {{
  border-bottom: 1px solid {BORDER};
  color: {MUTED}; font-size: 0.62rem; letter-spacing: 0.1em;
  text-transform: uppercase;
}}
.ptable thead th {{ padding: 0 14px 10px; text-align: left; font-weight: 400; }}
.ptable tbody tr {{ border-bottom: 1px solid {BORDER}22; }}
.ptable tbody tr:hover {{ background: {PANEL}; }}
.ptable tbody td {{ padding: 9px 14px; color: {TEXT}; }}
.ptable tbody td:first-child {{ color: {MUTED}; font-size: 0.65rem; }}
.ptable .shift {{ color: {WHITE}; font-weight: 500; }}
.ptable .r2hi  {{ color: {GREEN}; }}
.ptable .r2lo  {{ color: {AMBER}; }}
.ptable .r2no  {{ color: {MUTED}; }}

/* ── download btn ── */
.stDownloadButton button {{
  background: {PANEL} !important;
  border: 1px solid {BORDER} !important;
  color: {TEXT} !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.72rem !important;
  border-radius: 4px !important;
  padding: 6px 18px !important;
  margin-top: 10px;
}}
.stDownloadButton button:hover {{ border-color: {ACCENT}55 !important; }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class Log:
    def __init__(self):
        self._lines = []
    def info(self, m): self._lines.append(f'<span class="lv">  {m}</span>')
    def ok(self, m):   self._lines.append(f'<span class="lok">✓ {m}</span>')
    def warn(self, m): self._lines.append(f'<span class="lwarn">⚠ {m}</span>')
    def html(self):    return "\n".join(self._lines)


def _parse(content: bytes, log: Log):
    text = content.decode("utf-8", errors="replace")
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line[0] in "#%;":
            continue
        for sep in ("\t", ",", ";", "  ", " "):
            parts = [p.strip() for p in line.split(sep) if p.strip()]
            if len(parts) >= 2:
                try:
                    rows.append((float(parts[0]), float(parts[1])))
                    break
                except ValueError:
                    continue
    if len(rows) < 10:
        raise ValueError(f"Only {len(rows)} numeric rows — check file format.")
    a = np.array(rows)
    x, y = a[:, 0], a[:, 1]
    log.ok(f"Loaded {len(x):,} points  ·  x {x.min():.1f} – {x.max():.1f} cm⁻¹")
    return x, y


def _rayleigh(x, y, log: Log):
    xR = x[np.argmax(y)]
    log.ok(f"Rayleigh peak at {xR:.2f} cm⁻¹  →  shifted to 0")
    return x - xR, xR


def _denoise(y, log: Log):
    wv, lv = _P["wavelet"], _P["wavelet_level"]
    c   = pywt.wavedec(y, wv, level=lv)
    sig = np.median(np.abs(c[-1])) / 0.6745
    thr = sig * np.sqrt(2 * np.log(max(len(y), 2)))
    ct  = [c[0]] + [pywt.threshold(ci, thr, mode="soft") for ci in c[1:]]
    yd  = pywt.waverec(ct, wv)[: len(y)]
    log.ok(f"Wavelet denoised  ({wv}  L{lv})  ·  noise σ = {sig:.3f}")
    return yd


def _baseline(y, log: Log):
    lam, p, n = _P["als_lam"], _P["als_p"], _P["als_niter"]
    N = len(y)
    D = diags([1, -2, 1], [0, 1, 2], shape=(N-2, N))
    H = lam * D.T.dot(D)
    w = np.ones(N);  z = y.copy()
    for _ in range(n):
        W = diags(w, 0, shape=(N, N))
        z = spsolve(W + H, w * y)
        w = np.where(y > z, p, 1-p)
    log.ok("ALS baseline removed  (λ = 1e6)")
    return z, y - z


def _roi(x, y_den, y_corr, bl, log: Log):
    lo, hi = _P["roi_lo"], _P["roi_hi"]
    m = (x >= lo) & (x <= hi)
    if m.sum() < 5:
        raise ValueError("ROI has fewer than 5 points after Rayleigh shift.")
    log.ok(f"ROI selected  0 – 550 cm⁻¹  ·  {m.sum()} points retained")
    return x[m], y_den[m], y_corr[m], bl[m]


def _lorentzian(x, A, x0, g):
    return A * g**2 / ((x - x0)**2 + g**2)


def _detect(xr, yr, log: Log):
    dx   = float(np.median(np.diff(xr)))
    g    = _P["filter_gamma"]
    half = max(4, int(6 * g / dx))
    kx   = np.arange(-half, half + 1) * dx
    kern = g**2 / (kx**2 + g**2);  kern /= kern.sum()
    yr_mf = np.convolve(yr, kern, mode="same")

    prom  = _P["prom_frac"] * max(yr_mf.max() - yr_mf.min(), 1e-9)
    mdist = max(2, int(_P["min_dist_cm"] / dx))

    cands = set()
    for src, pf in [(yr_mf, 1.0), (yr_mf, 0.5), (yr, 0.3)]:
        idx, _ = signal.find_peaks(src, prominence=prom * pf,
                                   distance=max(2, mdist // 2))
        cands.update(idx.tolist())

    cands = np.array(sorted(cands))
    log.ok(f"{len(cands)} candidate peaks found between 0 – 550 cm⁻¹")
    return cands, yr_mf


def _fit_one(xr, yr, idx):
    dx   = float(np.median(np.diff(xr)))
    half = max(10, int(_P["fit_half_cm"] / dx))
    lo, hi = max(0, idx - half), min(len(xr), idx + half)
    xw, yw = xr[lo:hi], yr[lo:hi]
    x0g = xr[idx];  Ag = max(float(yr[idx]), 1e-3)
    try:
        popt, _ = curve_fit(
            _lorentzian, xw, yw,
            p0=[Ag, x0g, _P["filter_gamma"]],
            bounds=([0, 0, _P["gamma_min"]], [np.inf, _P["roi_hi"], _P["gamma_max"]]),
            maxfev=8000,
        )
        A, x0, g = popt
        yp   = _lorentzian(xw, *popt)
        ss_r = np.sum((yw - yp)**2)
        ss_t = np.sum((yw - yw.mean())**2)
        r2   = float(1 - ss_r / ss_t) if ss_t > 0 else 0.0
        return dict(converged=True, A=A, x0=x0, gamma=g, fwhm=2*g, r2=r2)
    except Exception:
        return dict(converged=False)


def run_pipeline(content: bytes):
    log = Log()
    x_raw, y_raw           = _parse(content, log)
    x, xR                  = _rayleigh(x_raw, y_raw, log)
    y_den                  = _denoise(y_raw, log)
    bl, y_corr             = _baseline(y_den, log)
    xr, yr_den, yr, bl_roi = _roi(x, y_den, y_corr, bl, log)
    cands, yr_mf            = _detect(xr, yr, log)

    peaks = []
    for rank, idx in enumerate(cands, 1):
        fit = _fit_one(xr, yr, idx)
        peaks.append(dict(
            n   = rank,
            pos = round(float(xr[idx]), 2),
            amp = round(float(yr[idx]), 2),
            **( dict(r2=round(fit["r2"], 3), fwhm=round(fit["fwhm"], 2), converged=True)
                if fit["converged"] else
                dict(r2=None, fwhm=None, converged=False) )
        ))

    return dict(
        log=log, x_raw=x_raw, y_raw=y_raw,
        x=x, xR=xR, y_den=y_den, bl=bl, y_corr=y_corr,
        xr=xr, yr=yr, yr_mf=yr_mf, cands=cands, peaks=peaks,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def _style_ax(ax, title, xl="Raman Shift (cm⁻¹)", yl="Intensity"):
    ax.set_title(title, fontsize=8, color=TEXT, loc="left",
                 fontfamily="monospace", pad=7)
    ax.set_xlabel(xl, fontsize=7.5)
    ax.set_ylabel(yl, fontsize=7.5)
    ax.grid(True, alpha=0.55)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)


def plot_graph1(x_raw, y_raw, x, xR):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), facecolor=BG)
    fig.subplots_adjust(wspace=0.32, left=0.07, right=0.97, top=0.88, bottom=0.16)

    ax = axes[0]
    ax.plot(x_raw, y_raw, color=ACCENT, lw=0.95, alpha=0.85)
    ax.axvline(xR, color=AMBER, lw=1.1, ls="--", alpha=0.8,
               label=f"Rayleigh @ {xR:.1f} cm⁻¹")
    ax.legend(fontsize=7)
    _style_ax(ax, "Raw Spectrum")

    ax2 = axes[1]
    m = (x >= 0) & (x <= 550)
    ax2.plot(x[m], y_raw[m], color=ACCENT, lw=0.95, alpha=0.85)
    ax2.axvline(0, color=AMBER, lw=1.0, ls="--", alpha=0.7, label="0 cm⁻¹")
    ax2.set_xlim(0, 550)
    ax2.legend(fontsize=7)
    _style_ax(ax2, "Rayleigh-Aligned  ·  0 – 550 cm⁻¹")
    return fig


def plot_graph2(xr, yr, peaks):
    fig, ax = plt.subplots(figsize=(11, 4.2), facecolor=BG)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.87, bottom=0.14)

    ax.plot(xr, yr, color=ACCENT, lw=1.0, alpha=0.75, zorder=2)

    if peaks:
        # vertical drop lines coloured by R²
        for p in peaks:
            c = GREEN if (p["r2"] is not None and p["r2"] >= 0.85) else \
                AMBER  if (p["r2"] is not None) else RED
            ax.plot([p["pos"], p["pos"]], [0, p["amp"]],
                    color=c, lw=0.75, alpha=0.5, zorder=3)

        px = np.array([p["pos"] for p in peaks])
        py = np.array([p["amp"] for p in peaks])
        ax.scatter(px, py, color=WHITE, s=30, zorder=5,
                   linewidths=0.6, edgecolors=BORDER)

        # staggered labels to reduce overlap
        sorted_peaks = sorted(peaks, key=lambda k: k["pos"])
        offsets = [14, 28, 44, 60]
        last_x  = -999.0
        bump    = 0
        for i, p in enumerate(sorted_peaks):
            if p["pos"] - last_x < 14:
                bump = (bump + 1) % len(offsets)
            else:
                bump = 0
            oy = offsets[bump]
            ax.annotate(
                f"{p['pos']:.0f}",
                xy=(p["pos"], p["amp"]),
                xytext=(0, oy), textcoords="offset points",
                ha="center", fontsize=6.5, color=WHITE,
                fontfamily="monospace", zorder=6,
                arrowprops=dict(arrowstyle="-", color=MUTED,
                                lw=0.5, alpha=0.45),
            )
            last_x = p["pos"]

    ax.set_xlim(0, 550)
    ypad = (yr.max() - yr.min()) * 0.15 if len(yr) else 1
    ax.set_ylim(bottom=yr.min() - ypad * 0.3)
    n = len(peaks)
    _style_ax(ax, f"Detected Peaks  ·  0 – 550 cm⁻¹  ·  {n} peak{'s' if n!=1 else ''} found")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# PEAK TABLE HTML
# ─────────────────────────────────────────────────────────────────────────────

def render_table(peaks):
    rows = ""
    for p in peaks:
        if p["r2"] is None:
            r2_html = f'<span class="r2no">—</span>'
            fwhm_s  = "—"
        elif p["r2"] >= 0.85:
            r2_html = f'<span class="r2hi">{p["r2"]:.3f}</span>'
            fwhm_s  = f'{p["fwhm"]:.2f}'
        else:
            r2_html = f'<span class="r2lo">{p["r2"]:.3f}</span>'
            fwhm_s  = f'{p["fwhm"]:.2f}'

        rows += f"""
        <tr>
          <td>{p['n']:02d}</td>
          <td class="shift">{p['pos']:.2f}</td>
          <td>{p['amp']:.1f}</td>
          <td>{fwhm_s}</td>
          <td>{r2_html}</td>
        </tr>"""

    return f"""
    <table class="ptable">
      <thead><tr>
        <th>#</th><th>Raman Shift (cm⁻¹)</th>
        <th>Intensity</th><th>FWHM (cm⁻¹)</th>
        <th>Lorentzian R²</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>"""


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(f"""
<div class="wordmark">
  <h1><span class="dot"></span>Raman Peak Analyzer</h1>
  <div class="sub">Upload · Auto-process · Inspect every peak</div>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Two-column file  ·  Raman Shift & Intensity  ·  .txt  .csv  .dat",
    type=["txt", "csv", "dat", "tsv"],
    label_visibility="visible",
)

if not uploaded:
    st.markdown(f"""
    <div style="text-align:center; padding:70px 0 50px;
         font-size:0.75rem; color:{MUTED};">
      drop a spectrum file above — analysis runs automatically
    </div>""", unsafe_allow_html=True)
    st.stop()

with st.spinner("Processing…"):
    try:
        R  = run_pipeline(uploaded.read())
        ok = True
    except Exception as exc:
        st.error(f"Error: {exc}")
        ok = False

if not ok:
    st.stop()

# ── LOG ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="divlabel">Processing log</div>', unsafe_allow_html=True)
st.markdown(f'<div class="logbox">{R["log"].html()}</div>', unsafe_allow_html=True)

# ── PEAK TABLE ───────────────────────────────────────────────────────────────
st.markdown('<div class="divlabel">Peak table</div>', unsafe_allow_html=True)

if not R["peaks"]:
    st.markdown(
        f'<p style="color:{MUTED};font-size:0.78rem;">'
        "No candidate peaks found in 0 – 550 cm⁻¹ region.</p>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(render_table(R["peaks"]), unsafe_allow_html=True)
    df = pd.DataFrame([
        {"Peak": p["n"], "Raman Shift (cm⁻¹)": p["pos"],
         "Intensity": p["amp"],
         "FWHM (cm⁻¹)": p["fwhm"] if p["fwhm"] else "",
         "Lorentzian R²": p["r2"]  if p["r2"]  else ""}
        for p in R["peaks"]
    ])
    st.download_button(
        "↓ Export as CSV",
        df.to_csv(index=False).encode(),
        file_name=f"{uploaded.name.rsplit('.',1)[0]}_peaks.csv",
        mime="text/csv",
    )

# ── GRAPH 1 ──────────────────────────────────────────────────────────────────
st.markdown('<div class="divlabel">Graph 1 · Raw + Rayleigh Alignment</div>',
            unsafe_allow_html=True)
fig1 = plot_graph1(R["x_raw"], R["y_raw"], R["x"], R["xR"])
st.pyplot(fig1, use_container_width=True)
plt.close(fig1)

# ── GRAPH 2 ──────────────────────────────────────────────────────────────────
st.markdown('<div class="divlabel">Graph 2 · Detected Peaks  (0 – 550 cm⁻¹)</div>',
            unsafe_allow_html=True)
fig2 = plot_graph2(R["xr"], R["yr"], R["peaks"])
st.pyplot(fig2, use_container_width=True)
plt.close(fig2)
