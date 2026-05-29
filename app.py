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

    return A * (
        gamma**2 /
        ((x - x0)**2 + gamma**2)
    )


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
    y = df.iloc[:, 1].values

    # --------------------------------------------------------
    # SHIFT TO RAYLEIGH PEAK
    # --------------------------------------------------------

    max_idx = np.argmax(y)

    x_use = x - x[max_idx]

    # --------------------------------------------------------
    # SMOOTHING
    # --------------------------------------------------------

    y_smooth = savgol_filter(
        y,
        21,
        3
    )

    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    baseline = savgol_filter(
        y_smooth,
        151,
        3
    )

    signal = y_smooth - baseline

    signal = np.clip(signal, 0, None)

    # --------------------------------------------------------
    # NOISE
    # --------------------------------------------------------

    noise_region = signal[
        (x_use > 700) &
        (x_use < 1200)
    ]

    if len(noise_region) == 0:
        noise_std = np.std(signal)
    else:
        noise_std = np.std(noise_region)

    dynamic_prominence = 4 * noise_std

    dynamic_height = 3 * noise_std

    # --------------------------------------------------------
    # PEAK DETECTION
    # --------------------------------------------------------

    candidate_peaks, properties = find_peaks(

        signal,

        prominence=dynamic_prominence,

        height=dynamic_height,

        distance=8,

        width=2
    )

    candidate_peaks = np.array([

        p for p in candidate_peaks

        if x_use[p] > 50
    ])

    filtered_peaks = []

    for p in candidate_peaks:

        peak_height = signal[p]

        half_height = peak_height / 2

        left = p

        while left > 0 and signal[left] > half_height:
            left -= 1

        right = p

        while right < len(signal)-1 and signal[right] > half_height:
            right += 1

        width = right - left

        local_region = signal[
            max(0, p-20):
            min(len(signal), p+20)
        ]

        local_noise = np.std(local_region)

        snr = peak_height / (local_noise + 1e-9)

        if snr < 2.5:
            continue

        if width < 3:
            continue

        filtered_peaks.append(p)

    candidate_peaks = np.array(filtered_peaks)

    # --------------------------------------------------------
    # LORENTZIAN FIT
    # --------------------------------------------------------

    final_peaks = []

    for p in candidate_peaks:

        try:

            left = max(0, p-15)

            right = min(
                len(signal)-1,
                p+15
            )

            x_fit = x_use[left:right]

            y_fit = signal[left:right]

            p0 = [
                x_use[p],
                5,
                signal[p]
            ]

            popt, _ = curve_fit(
                lorentzian,
                x_fit,
                y_fit,
                p0=p0
            )

            x0, gamma, A = popt

            final_peaks.append(
                (x0, A)
            )

        except:
            pass

    return {

        "sample":
            uploaded_file.name,

        "x":
            x_use,

        "signal":
            signal,

        "peaks":
            final_peaks
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

    st.success(
        f"{len(results)} files processed."
    )

    # ========================================================
    # PEAK TABLE
    # ========================================================

    peak_rows = []

    for result in results:

        sample = result["sample"]

        for i, peak in enumerate(
            result["peaks"],
            start=1
        ):

            peak_rows.append({

                "Sample":
                    sample,

                "Peak Number":
                    i,

                "Raman Shift":
                    round(
                        peak[0],
                        2
                    ),

                "Intensity":
                    round(
                        peak[1],
                        2
                    )
            })

    peak_df = pd.DataFrame(
        peak_rows
    )

    st.subheader(
        "Detected Peaks"
    )

    st.dataframe(
        peak_df,
        use_container_width=True
    )

    # ========================================================
    # CSV DOWNLOAD
    # ========================================================

    csv = peak_df.to_csv(
        index=False
    )

    st.download_button(

        "Download Peak Table",

        csv,

        file_name=
        "Peak_Table.csv",

        mime=
        "text/csv"
    )

    # ========================================================
    # OVERLAY GRAPH
    # ========================================================

    st.subheader(
        "Overlay Raman Spectra"
    )

    fig, ax = plt.subplots(
        figsize=(12,6)
    )

    for result in results:

        ax.plot(

            result["x"],

            result["signal"],

            label=result["sample"]
        )

    ax.legend()

    ax.grid()

    ax.set_xlabel(
        "Raman Shift"
    )

    ax.set_ylabel(
        "Intensity"
    )

    st.pyplot(fig)

    # ========================================================
    # PEAK COMPARISON GRAPH
    # ========================================================

    st.subheader(
        "Peak Comparison Graph"
    )

    fig2, ax2 = plt.subplots(
        figsize=(14,8)
    )

    for result in results:

        sample = result["sample"]

        peaks = [

            p[0]

            for p in result["peaks"]
        ]

        ax2.scatter(

            peaks,

            [sample] * len(peaks),

            s=80
        )

        for peak in peaks:

            ax2.text(

                peak,

                sample,

                f"{peak:.0f}",

                fontsize=8
            )

    ax2.grid()

    ax2.set_xlabel(
        "Raman Shift"
    )

    ax2.set_ylabel(
        "Sample"
    )

    st.pyplot(fig2)
