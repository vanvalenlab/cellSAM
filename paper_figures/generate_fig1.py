import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

name_map = {
    "Gendarme_BriFi": "BriFiSeg",
    "cellpose": "Cellpose",
    "ep_phase_microscopy_all": "Phase400",
    "H_and_E": "H&E",
    "tissuenet_wholecell": "TissueNet",
    "YeaZ": "YeaZ",
    "YeastNet": "YeastNet",
    "dsb_fixed": "DSB",
    "deepbacs": "DeepBacs",
    "omnipose": "OmniPose",
}


datasets = [
    'Gendarme_BriFi',
    'H_and_E',
    'YeaZ',
    'YeastNet',
    'cellpose',
    'deepbacs',
    'dsb_fixed',
    'ep_phase_microscopy_all',
    'omnipose',
    'tissuenet_wholecell',
]

# colors for the plots
c1 = "#fdbb84"
c2 = "#e34a33"
c3 = '#deebf7'
c4 = '#3182bd'

# define paths to results
cellsam_path = Path('/home/ulisrael/cellSAM/paper_figures/eval_results/cellsam')
cellpose_path = Path('/home/ulisrael/cellSAM/paper_figures/eval_results/cellpose')

cellpose_generalist_path = cellpose_path / 'general'
cellsam_generalist_path = cellsam_path / 'general'

# read in results for cellpose generalist
cp_generalist_dict = {}
for file in cellpose_generalist_path.glob("*.txt"):
    try:
        data = np.loadtxt(file)
        cp_generalist_dict[file.stem] = data
    except Exception as e:
        print(f"Error reading {file.name}: {e}")

# read in results for cellsam generalist
cs_generalist_dict = {}
for file in cellsam_generalist_path.glob("*.txt"):
    try:
        data = np.loadtxt(file)
        cs_generalist_dict[file.stem] = data
    except Exception as e:
        print(f"Error reading {file.name}: {e}")


cp_means = []; cs_means = []
cp_sems = []; cs_sems = []

for ds in datasets:
    cp_data = cp_generalist_dict[ds]
    cs_data = cs_generalist_dict[ds]
    # 1 - mean for the bar
    cp_m = 1 - np.mean(cp_data)
    cs_m = 1 - np.mean(cs_data)
    # standard error of the mean for the error bar
    cp_sem = np.std(cp_data, ddof=1) / np.sqrt(len(cp_data))
    cs_sem = np.std(cs_data, ddof=1) / np.sqrt(len(cs_data))

    cp_means.append(cp_m)
    cs_means.append(cs_m)
    cp_sems.append(cp_sem)
    cs_sems.append(cs_sem)

# Plot as a bar chart
x = np.arange(len(datasets))
width = 0.35  # width of each bar

fig, ax = plt.subplots(figsize=(8, 5))

# Plot CP bars slightly left, CS bars slightly right
bars_cp = ax.bar(x - width/2, cp_means, width, 
                 edgecolor='black', linewidth=1,
                 yerr=cp_sems, capsize=5, label='CellPose', color=c2)
bars_cs = ax.bar(x + width/2, cs_means, width, 
                 edgecolor='black', linewidth=1,
                 yerr=cs_sems, capsize=5, label='CellSam', color=c4)

ax.set_xticks(x)
ax.set_xticklabels([name_map[ds] for ds in datasets], rotation=45, ha='right')
ax.set_ylabel('Mean Error (1 - F1)')
# ax.set_title('Generalist Model Comparison of Mean Error')
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, 1.15),
    ncol=2,
    prop={'size': 14},
    frameon=False
)
plt.tight_layout()
fig.savefig("mean_error_dataset_comparison_cp_reg.svg", format="svg", dpi=300)
plt.show()


dataset_agg_map = {
    "Tissue": ["tissuenet_wholecell"],
    "Cell Culture": ["cellpose", "ep_phase_microscopy_all", "Gendarme_BriFi"],
    "H&E": ["H_and_E"],
    "Bacteria": ["deepbacs", "omnipose"],
    "Yeast": ["YeaZ", "YeastNet"],
    "Nuclear": ["dsb_fixed"],
}

group_names = list(dataset_agg_map.keys())
cp_group_means = []
cp_group_sems  = []
cs_group_means = []
cs_group_sems  = []

for group in group_names:
    # Get all datasets that belong to this group
    datasets_for_group = dataset_agg_map[group]
    
    # Gather all F1 arrays and concatenate them
    cp_all = np.concatenate([cp_generalist_dict[ds] for ds in datasets_for_group])
    cs_all = np.concatenate([cs_generalist_dict[ds] for ds in datasets_for_group])
    
    # Compute (1 - mean(F1)) for the group
    cp_mean = 1 - np.mean(cp_all)
    cs_mean = 1 - np.mean(cs_all)
    cp_group_means.append(cp_mean)
    cs_group_means.append(cs_mean)
    
    # Standard error of the mean (SEM) for the group
    cp_sem = np.std(cp_all, ddof=1) / np.sqrt(len(cp_all))
    cs_sem = np.std(cs_all, ddof=1) / np.sqrt(len(cs_all))
    cp_group_sems.append(cp_sem)
    cs_group_sems.append(cs_sem)

# Now plot side‐by‐side bars for the groups
x = np.arange(len(group_names))
width = 0.35
plt.rcParams['svg.fonttype'] = 'none' 
# plt.rcParams['ps.fonttype'] = 42 
# plt.rcParams['font.family'] = 'Arial'  # Set a standard font (adjust as needed)
fig, ax = plt.subplots(figsize=(8,5))

bars_cp = ax.bar(
    x - width/2, cp_group_means, width,
    yerr=cp_group_sems, edgecolor='black', linewidth=1, capsize=5, 
    label='CellPose', color=c2
)
bars_cs = ax.bar(
    x + width/2, cs_group_means, width,
    yerr=cs_group_sems, edgecolor='black', linewidth=1, capsize=5, 
    label='CellSam', color=c4
)

ax.set_xticks(x)
ax.set_xticklabels(group_names, rotation=45, ha='right')
ax.set_ylabel('Mean Error (1 - F1)')
# ax.set_title('Grouped Comparison of Mean Error')
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, 1.15),
    ncol=2,
    prop={'size': 14},
    frameon=False
)
plt.tight_layout()
# save figure as vector
# export to edit in illustrator

# fig.savefig("mean_error_general_grouped_comparison_cp_reg.svg", format="svg", dpi=300)
plt.show()