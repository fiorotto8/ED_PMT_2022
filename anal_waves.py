import os
import re
import PMT
from PMT import PMTwave
from tqdm import tqdm
import ROOT
from collections import defaultdict
import numpy as np
import uproot

def classify_files(directory):
    files = [f for f in os.listdir(directory) if 'C1' in f]
    classified_files = []

    for file in files:
        drift_match = re.search(r'(\d+)drift', file)
        volt_match = re.search(r'(\d+)V', file)

        driftE = int(drift_match.group(1)) if drift_match else 1000
        Volt = int(volt_match.group(1)) if volt_match else None

        classified_files.append({
            'filename': file,
            'driftE': driftE,
            'Volt': Volt
        })

    return classified_files

# Example usage
directory = 'waves/'
classified_files = classify_files(directory)

ROOT.gROOT.SetBatch(True)
main=ROOT.TFile("main.root","RECREATE")

# Dictionary to store deltaT values for each (driftE, Volt) combination
deltaT_dict = defaultdict(list)

# Loop over all files with tqdm for progress tracking
counter = 0
for file_info in tqdm(classified_files, desc="Processing files"):
    wf = PMTwave(os.path.join(directory, file_info['filename']))
    minPeak = wf.minPeakTime
    maxPeak = wf.maxPeakTime
    if minPeak is not None and maxPeak is not None:
        deltaT = maxPeak - minPeak
        # Append deltaT to the corresponding (driftE, Volt) group
        deltaT_dict[(file_info['driftE'], file_info['Volt'])].append(deltaT)
        if counter % 1 == 0:
            wf.plot_zoomed_waveform_with_filtered()
    else:
        continue
        #wf.plot_zoomed_waveform_with_filtered()
    counter += 1

# Create histograms for each (driftE, Volt) combination
for (driftE, Volt), deltaT_values in deltaT_dict.items():
    if deltaT_values:  # Only create a histogram if there are values
        hist_name = f"deltaT_drift{driftE}_volt{Volt}"
        PMT.hist(deltaT_values, hist_name, channels=20)

print("All histograms have been created and saved to main.root.")


# Prepare data for TTree
tree_driftE = []
tree_volt = []
tree_deltaT = []

# Loop over the deltaT_dict to prepare data for the TTree
for (driftE, Volt), deltaT_values in deltaT_dict.items():
    for deltaT in deltaT_values:
        tree_driftE.append(driftE)
        tree_volt.append(Volt)
        tree_deltaT.append(deltaT)

# Convert lists to NumPy arrays
tree_data = {
    "driftE": np.array(tree_driftE, dtype=np.int32),
    "Volt": np.array(tree_volt, dtype=np.int32),
    "deltaT": np.array(tree_deltaT, dtype=np.float64),
}

# Write the TTree to a ROOT file
with uproot.recreate("output_tree.root") as file:
    file["deltaTTree"] = tree_data

print("TTree written to output_tree.root.")