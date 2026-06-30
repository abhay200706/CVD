import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io

st.title("Rayleigh Corrected Raman Data")

uploaded_files = st.file_uploader(
    "Upload your Raman spectroscopy .txt files",
    type=["txt"],
    accept_multiple_files=True
)

if uploaded_files:
    corrected_results = {}  # filename -> corrected_data (numpy array)

    for file in uploaded_files:
        data = np.loadtxt(file)
        raman_shift = data[:, 0]
        intensity = data[:, 1]

        # Rayleigh shift: shift the raman shift value at max intensity to zero
        max_idx = np.argmax(intensity)
        shift_value = raman_shift[max_idx]
        corrected_shift = raman_shift - shift_value

        corrected_data = np.column_stack((corrected_shift, intensity))
        corrected_results[file.name] = corrected_data

        st.subheader(file.name)

        # Individual plot: original vs corrected, overlapping
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(raman_shift, intensity, label="Original", alpha=0.7)
        ax.plot(corrected_shift, intensity, label="Rayleigh Corrected", alpha=0.7)
        ax.set_xlim(-50, 550)
        ax.set_ylim(300, 1200)
        ax.set_xlabel("Raman Shift (cm-1)")
        ax.set_ylabel("Intensity")
        ax.set_title(file.name)
        ax.legend()
        st.pyplot(fig)

        # Individual download button for this file's corrected data
        txt_buffer = io.StringIO()
        np.savetxt(txt_buffer, corrected_data, fmt="%.4f")
        out_name = file.name.replace(".txt", "_rayleigh_corrected.txt")
        st.download_button(
            label=f"Download corrected file: {out_name}",
            data=txt_buffer.getvalue(),
            file_name=out_name,
            mime="text/plain",
            key=out_name
        )

        st.markdown("---")

    # Final combined plot: all corrected files overlapping
    st.subheader("All Rayleigh Corrected Spectra (Overlapping)")
    fig_all, ax_all = plt.subplots(figsize=(10, 6))
    for name, corrected_data in corrected_results.items():
        ax_all.plot(corrected_data[:, 0], corrected_data[:, 1], label=name)
    ax_all.set_xlim(-50, 550)
    ax_all.set_ylim(300, 1200)
    ax_all.set_xlabel("Raman Shift (cm-1)")
    ax_all.set_ylabel("Intensity")
    ax_all.set_title("Overlapping Rayleigh Corrected Raman Spectra")
    ax_all.legend(fontsize=7)
    st.pyplot(fig_all)
