"""
Robust Raman Peak Detection Pipeline
-------------------------------------
Workflow:
  Raw Spectrum
    → Rayleigh Removal
    → Savitzky-Golay Smoothing
    → arPLS Baseline Correction
    → Derivative Peak Candidate Detection
    → Prominence + Width Filtering
    → Lorentzian Fitting
    → Output / Plot

Usage:
    python raman_peak_detect.py <your_file.csv> [options]

Options:
    --col-x        Column index for Raman shift (default: 0)
    --col-y        Column index for intensity   (default: 1)
    --rayleigh-gap Gap (cm⁻¹) after Rayleigh peak to start (default: 80)
    --sg-window    Savitzky-Golay window length  (default: 11)
    --sg-poly      Savitzky-Golay poly order     (default: 3)
    --prominence   Noise multiples for prominence threshold (default: 4)
    --max-width    Max peak width in points (for fluorescence rejection, default: 80)
    --min-width    Min peak width in points (default: 3)
    --distance     Min distance between peaks in points (default: 10)
    --als-lam      arPLS lambda smoothness (default: 1e5)
    --als-ratio    arPLS convergence ratio (default: 0.001)
    --no-plot      Suppress plot output
    --output       Save results to CSV (e.g. peaks_out.csv)
"""

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks, peak_widths
from scipy.optimize import curve_fit
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1.  arPLS Baseline Correction
#     (Asymmetrically Reweighted Penalized Least Squares)
#     Much more accurate than a wide Savitzky-Golay baseline.
# ─────────────────────────────────────────────
def arPLS_baseline(y, lam=1e5, ratio=0.001, max_iter=200):
    N = len(y)
    D = diags([1, -2, 1], [0, 1, 2], shape=(N - 2, N))
    H = lam * D.T.dot(D)
    w = np.ones(N)
    for _ in range(max_iter):
        W = diags(w, 0)
        Z = W + H
        z = spsolve(Z, w * y)
        d = y - z
        d_neg = d[d < 0]
        m = d_neg.mean() if len(d_neg) else 0.0
        s = d_neg.std()  if len(d_neg) else 1.0
        w_new = 1.0 / (1.0 + np.exp(2.0 * (d - (2.0 * s - m)) / s))
        if np.linalg.norm(w_new - w) / np.linalg.norm(w) < ratio:
            break
        w = w_new
    return z


# ─────────────────────────────────────────────
# 2.  Lorentzian model for peak fitting
# ─────────────────────────────────────────────
def lorentzian(x, amp, cen, wid):
    return amp * (wid**2) / ((x - cen)**2 + wid**2)


def fit_lorentzian(x, y_corr, peak_idx):
    """Fit a Lorentzian around a detected peak index.
    Returns (center_cm, fwhm_cm, amplitude, r_squared) or None on failure."""
    half_win = 30  # points either side
    lo = max(0, peak_idx - half_win)
    hi = min(len(x) - 1, peak_idx + half_win)
    xw, yw = x[lo:hi+1], y_corr[lo:hi+1]

    amp0  = yw.max()
    cen0  = x[peak_idx]
    wid0  = (x[hi] - x[lo]) / 6.0

    try:
        popt, _ = curve_fit(
            lorentzian, xw, yw,
            p0=[amp0, cen0, wid0],
            bounds=([0, xw[0], 0.5], [np.inf, xw[-1], (xw[-1]-xw[0])]),
            maxfev=5000
        )
        amp, cen, wid = popt
        y_fit = lorentzian(xw, *popt)
        ss_res = np.sum((yw - y_fit)**2)
        ss_tot = np.sum((yw - yw.mean())**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        fwhm = 2 * abs(wid)
        return cen, fwhm, amp, r2
    except Exception:
        return None


# ─────────────────────────────────────────────
# 3.  Derivative-based candidate detection
# ─────────────────────────────────────────────
def derivative_candidates(y_corr):
    dy  = np.gradient(y_corr)
    d2y = np.gradient(dy)
    candidates = []
    for i in range(1, len(dy) - 1):
        if dy[i - 1] > 0 and dy[i + 1] < 0 and d2y[i] < 0:
            candidates.append(i)
    return np.array(candidates, dtype=int)


# ─────────────────────────────────────────────
# 4.  Main pipeline
# ─────────────────────────────────────────────
def detect_peaks(filepath, col_x=0, col_y=1,
                 rayleigh_gap=80, sg_window=11, sg_poly=3,
                 prominence_mult=4, max_width=80, min_width=3,
                 distance=10, als_lam=1e5, als_ratio=0.001,
                 show_plot=True, output_csv=None):

    # ── Load ──────────────────────────────────
    df = pd.read_csv(filepath, header=None, comment='#', sep=None,
                     engine='python', skip_blank_lines=True)
    # Try to drop header row if first row is non-numeric
    try:
        float(df.iloc[0, col_x])
    except (ValueError, TypeError):
        df = df.iloc[1:].reset_index(drop=True)

    x_raw = df.iloc[:, col_x].astype(float).values
    y_raw = df.iloc[:, col_y].astype(float).values

    # Sort ascending by shift
    order = np.argsort(x_raw)
    x_raw, y_raw = x_raw[order], y_raw[order]

    # ── Rayleigh Removal ──────────────────────
    ray_idx = np.argmax(y_raw)
    ray_center = x_raw[ray_idx]
    mask = x_raw > ray_center + rayleigh_gap
    if mask.sum() < 20:
        raise ValueError(
            f"Too few points after Rayleigh removal at {ray_center:.1f} cm⁻¹ "
            f"+ {rayleigh_gap} cm⁻¹ gap. Check col_x / col_y."
        )
    x = x_raw[mask]
    y = y_raw[mask]

    print(f"Rayleigh peak at {ray_center:.1f} cm⁻¹ → keeping {len(x)} points "
          f"from {x[0]:.1f} to {x[-1]:.1f} cm⁻¹")

    # ── Savitzky-Golay Smoothing ──────────────
    win = sg_window if sg_window % 2 == 1 else sg_window + 1
    win = min(win, len(y) - 1 if (len(y) - 1) % 2 == 1 else len(y) - 2)
    y_smooth = savgol_filter(y, window_length=win, polyorder=sg_poly)

    # ── arPLS Baseline Correction ─────────────
    baseline = arPLS_baseline(y_smooth, lam=als_lam, ratio=als_ratio)
    y_corr   = y_smooth - baseline
    # Zero-floor (negative values are artefacts below noise)
    y_corr   = np.clip(y_corr, 0, None)

    # ── Noise Estimate (MAD-based, robust) ────
    noise = np.median(np.abs(y_corr - np.median(y_corr))) / 0.6745

    print(f"Estimated noise (σ): {noise:.2f} counts")

    # ── Derivative Candidates ─────────────────
    deriv_cands = derivative_candidates(y_corr)

    # ── find_peaks filter on candidates ───────
    prom_thresh = prominence_mult * noise
    if len(deriv_cands) == 0:
        peaks = np.array([], dtype=int)
        props = {'prominences': np.array([])}
    else:
        y_cand = np.zeros_like(y_corr)
        y_cand[deriv_cands] = y_corr[deriv_cands]
        peaks, props = find_peaks(
            y_corr,
            prominence=prom_thresh,
            width=min_width,
            distance=distance
        )
        # Keep only peaks that were also derivative candidates (within ±3 pts)
        def near_cand(p):
            return np.any(np.abs(deriv_cands - p) <= 3)
        peaks = np.array([p for p in peaks if near_cand(p)], dtype=int)
        if len(peaks):
            _, props_arr = find_peaks(y_corr, prominence=prom_thresh)
            # Recompute prominences for kept peaks
            _, proms_all = find_peaks(y_corr[peaks] if len(peaks) else y_corr)
            proms = np.array([
                props['prominences'][list(
                    find_peaks(y_corr, prominence=prom_thresh, width=min_width,
                               distance=distance)[0]).index(p)]
                if p in find_peaks(y_corr, prominence=prom_thresh,
                                   width=min_width, distance=distance)[0]
                else y_corr[p]
                for p in peaks
            ])
            props = {'prominences': proms}

    # ── Width Filtering (reject fluorescence humps) ──
    if len(peaks):
        widths_pts = peak_widths(y_corr, peaks, rel_height=0.5)[0]
        good = widths_pts < max_width
        peaks = peaks[good]
        if len(props['prominences']) == len(good):
            props['prominences'] = props['prominences'][good]
        else:
            props['prominences'] = y_corr[peaks]

    print(f"Prominence threshold: {prom_thresh:.2f} | "
          f"Candidate peaks after all filters: {len(peaks)}")

    # ── Lorentzian Fitting ────────────────────
    results = []
    for i, p in enumerate(peaks):
        fit = fit_lorentzian(x, y_corr, p)
        prom_val = props['prominences'][i] if i < len(props['prominences']) else y_corr[p]
        if fit:
            cen, fwhm, amp, r2 = fit
            results.append({
                'shift_raw_cm':   round(x[p], 2),
                'shift_fit_cm':   round(cen,  2),
                'intensity':      round(y_corr[p], 2),
                'amplitude_fit':  round(amp,  2),
                'fwhm_cm':        round(fwhm, 2),
                'prominence':     round(prom_val, 2),
                'r2_lorentzian':  round(r2,   4),
            })
        else:
            results.append({
                'shift_raw_cm':   round(x[p], 2),
                'shift_fit_cm':   round(x[p], 2),
                'intensity':      round(y_corr[p], 2),
                'amplitude_fit':  None,
                'fwhm_cm':        None,
                'prominence':     round(prom_val, 2),
                'r2_lorentzian':  None,
            })

    df_out = pd.DataFrame(results)

    # ── Print Table ───────────────────────────
    print("\n" + "="*75)
    print(f"{'#':>3}  {'Raw shift':>12}  {'Fit shift':>12}  "
          f"{'Intensity':>10}  {'FWHM':>8}  {'R²':>6}")
    print("-"*75)
    for i, r in df_out.iterrows():
        fwhm_s = f"{r['fwhm_cm']:8.2f}" if r['fwhm_cm'] is not None else "    N/A "
        r2_s   = f"{r['r2_lorentzian']:6.4f}" if r['r2_lorentzian'] is not None else "   N/A"
        print(f"{i+1:>3}  {r['shift_raw_cm']:>12.2f}  {r['shift_fit_cm']:>12.2f}  "
              f"{r['intensity']:>10.2f}  {fwhm_s}  {r2_s}")
    print("="*75)
    print(f"Total peaks detected: {len(df_out)}\n")

    # ── Save CSV ──────────────────────────────
    if output_csv:
        df_out.to_csv(output_csv, index=False)
        print(f"Peaks saved to: {output_csv}")

    # ── Plot ──────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 12),
                             gridspec_kw={'hspace': 0.45})
    fig.suptitle("Raman Peak Detection Pipeline", fontsize=13, fontweight='bold')

    # Panel 1: Raw + Rayleigh boundary
    ax = axes[0]
    ax.plot(x_raw, y_raw, color='steelblue', lw=0.8, label='Raw spectrum')
    ax.axvline(ray_center + rayleigh_gap, color='red', ls='--', lw=1,
               label=f'Rayleigh cutoff ({ray_center+rayleigh_gap:.0f} cm⁻¹)')
    ax.set_title("Raw Spectrum + Rayleigh Removal")
    ax.set_xlabel("Raman Shift (cm⁻¹)")
    ax.set_ylabel("Intensity (counts)")
    ax.legend(fontsize=8)
    ax.set_xlim(x_raw[0], x_raw[-1])

    # Panel 2: Smoothed + arPLS baseline
    ax = axes[1]
    ax.plot(x, y_smooth,   color='steelblue', lw=0.9, label='Smoothed')
    ax.plot(x, baseline,   color='orange',    lw=1.2, ls='--', label='arPLS baseline')
    ax.set_title("Smoothed Spectrum + arPLS Baseline")
    ax.set_xlabel("Raman Shift (cm⁻¹)")
    ax.set_ylabel("Intensity (counts)")
    ax.legend(fontsize=8)
    ax.set_xlim(x[0], x[-1])

    # Panel 3: Baseline-corrected + detected peaks
    ax = axes[2]
    ax.plot(x, y_corr, color='steelblue', lw=0.9, label='Baseline-corrected')
    ax.axhline(prom_thresh, color='gray', ls=':', lw=0.8, label=f'Prom. threshold ({prom_thresh:.1f})')
    for r in results:
        px = r['shift_fit_cm']
        py = r['intensity']
        ax.axvline(px, color='red', lw=0.6, alpha=0.4)
        ax.annotate(f"{px:.0f}", xy=(px, py),
                    xytext=(0, 6), textcoords='offset points',
                    ha='center', fontsize=6.5, color='red', rotation=90)
    ax.scatter(df_out['shift_fit_cm'], df_out['intensity'],
               color='red', zorder=5, s=22, label='Detected peaks')
    ax.set_title(f"Baseline-Corrected + Detected Peaks ({len(df_out)} peaks)")
    ax.set_xlabel("Raman Shift (cm⁻¹)")
    ax.set_ylabel("Corrected Intensity (counts)")
    ax.legend(fontsize=8)
    ax.set_xlim(x[0], x[-1])

    plt.savefig("/mnt/user-data/outputs/raman_peaks_plot.png", dpi=150,
                bbox_inches='tight')
    print("Plot saved: raman_peaks_plot.png")
    plt.close()

    return df_out


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robust Raman Peak Detector")
    parser.add_argument("file",           help="Input CSV file path")
    parser.add_argument("--col-x",        type=int,   default=0)
    parser.add_argument("--col-y",        type=int,   default=1)
    parser.add_argument("--rayleigh-gap", type=float, default=80)
    parser.add_argument("--sg-window",    type=int,   default=11)
    parser.add_argument("--sg-poly",      type=int,   default=3)
    parser.add_argument("--prominence",   type=float, default=4)
    parser.add_argument("--max-width",    type=float, default=80)
    parser.add_argument("--min-width",    type=float, default=3)
    parser.add_argument("--distance",     type=int,   default=10)
    parser.add_argument("--als-lam",      type=float, default=1e5)
    parser.add_argument("--als-ratio",    type=float, default=0.001)
    parser.add_argument("--no-plot",      action="store_true")
    parser.add_argument("--output",       type=str,   default=None)
    args = parser.parse_args()

    detect_peaks(
        filepath      = args.file,
        col_x         = args.col_x,
        col_y         = args.col_y,
        rayleigh_gap  = args.rayleigh_gap,
        sg_window     = args.sg_window,
        sg_poly       = args.sg_poly,
        prominence_mult = args.prominence,
        max_width     = args.max_width,
        min_width     = args.min_width,
        distance      = args.distance,
        als_lam       = args.als_lam,
        als_ratio     = args.als_ratio,
        show_plot     = not args.no_plot,
        output_csv    = args.output,
    )
