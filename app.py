import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# Folder containing your input .txt files (Raman shift, Intensity columns)
input_folder = "input_data"
output_folder = "rayleigh_corrected_data"

os.makedirs(output_folder, exist_ok=True)

# Get all txt files in the input folder
files = glob.glob(os.path.join(input_folder, "*.txt"))

plt.figure(figsize=(10, 6))

for file in files:
    data = np.loadtxt(file)
    raman_shift = data[:, 0]
    intensity = data[:, 1]

    # Rayleigh shift: shift the raman shift value at max intensity to zero
    max_idx = np.argmax(intensity)
    shift_value = raman_shift[max_idx]
    corrected_shift = raman_shift - shift_value

    # Save corrected data
    filename = os.path.basename(file)
    output_path = os.path.join(output_folder, filename.replace(".txt", "_rayleigh_corrected.txt"))
    corrected_data = np.column_stack((corrected_shift, intensity))
    np.savetxt(output_path, corrected_data, fmt="%.4f")

    # Plot
    plt.plot(corrected_shift, intensity, label=filename)

plt.xlim(-50, 550)
plt.ylim(300, 1200)
plt.xlabel("Raman Shift (cm-1)")
plt.ylabel("Intensity")
plt.title("Overlapping Rayleigh Corrected Raman Spectra")
plt.legend(fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "overlapping_plot.png"), dpi=300)
plt.show()

print("All files converted to Rayleigh corrected txt files in:", output_folder)
