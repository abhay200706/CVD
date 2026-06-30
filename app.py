import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
import zipfile

st.title("Rayleigh Corrected Raman Data")

uploaded_files = st.file_uploader(
    "Upload your Raman spectroscopy .txt files",
    type=["txt"],
    accept_multiple_files=True
)

if uploaded_files:
    corrected_results = {}  # filename -> corrected_data (numpy array)

    fig, ax = plt.subplots(figsize=(10, 6))

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

        ax.plot(corrected_shift, intensity, label=file.name)

    ax.set_xlim(-50, 550)
    ax.set_ylim(300, 1200)
    ax.set_xlabel("Raman Shift (cm-1)")
    ax.set_ylabel("Intensity")
    ax.set_title("Overlapping Rayleigh Corrected Raman Spectra")
    ax.legend(fontsize=7)
    st.pyplot(fig)

    # Zip all corrected txt files together for download
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        for name, corrected_data in corrected_results.items():
            txt_buffer = io.StringIO()
            np.savetxt(txt_buffer, corrected_data, fmt="%.4f")
            out_name = name.replace(".txt", "_rayleigh_corrected.txt")
            zf.writestr(out_name, txt_buffer.getvalue())

    zip_buffer.seek(0)
    st.download_button(
        label="Download all Rayleigh corrected txt files (zip)",
        data=zip_buffer,
        file_name="rayleigh_corrected_data.zip",
        mime="application/zip"
    )
