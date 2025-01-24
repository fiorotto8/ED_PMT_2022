import ROOT
import numpy as np
from collections import defaultdict
import PMT 

# Example: Read the TTree data from the output ROOT file
import uproot

file_name = "output_tree.root"
tree_name = "deltaTTree"

# Open the file and read the TTree
with uproot.open(file_name) as file:
    tree = file[tree_name]
    data = tree.arrays(library="np")

# Extract branches
driftE = data["driftE"]
volt = data["Volt"]
deltaT = data["deltaT"]

# Group deltaT by (driftE, Volt)
grouped_data = defaultdict(list)
for dE, v, dt in zip(driftE, volt, deltaT):
    grouped_data[(dE, v)].append(dt)

# Prepare data for TGraphs
means_by_driftE = defaultdict(list)  # {driftE: [(volt, mean, error)]}
means_by_volt = defaultdict(list)    # {volt: [(driftE, mean, error)]}

for (dE, v), deltaT_values in grouped_data.items():
    mean = np.mean(deltaT_values)
    error = np.std(deltaT_values) / np.sqrt(len(deltaT_values))  # Statistical error
    means_by_driftE[dE].append((v, mean, error))
    means_by_volt[v].append((dE, mean, error))

# Create the output ROOT file
output_file = ROOT.TFile("deltaT_means.root", "RECREATE")

# Plot mean deltaT vs Volt for each driftE
for dE, points in means_by_driftE.items():
    points.sort()  # Sort by Volt
    volts, means, errors = zip(*points)
    PMT.grapherr(
        volts,
        means,
        [0] * len(volts),  # No error in x
        errors,
        "Voltage (V)",
        "Mean deltaT (ns)",
        name=f"deltaT_vs_Volt_driftE_{dE}",
        color=4,
        markerstyle=22,
        markersize=2,
        write=True
    )

# Plot mean deltaT vs driftE for each Volt
for v, points in means_by_volt.items():
    points.sort()  # Sort by driftE
    driftEs, means, errors = zip(*points)
    PMT.grapherr(
        driftEs,
        means,
        [0] * len(driftEs),  # No error in x
        errors,
        "Drift Field (V/cm)",
        "Mean deltaT (ns)",
        name=f"deltaT_vs_driftE_Volt_{v}",
        color=2,
        markerstyle=21,
        markersize=2,
        write=True
    )

# Write and close the output ROOT file
output_file.Write()
output_file.Close()

print("TGraphs written to deltaT_means.root.")