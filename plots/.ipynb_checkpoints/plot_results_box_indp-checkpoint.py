import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from statannotations.Annotator import Annotator
# Update fonts to bold and use LaTeX
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'


# ____________________________________________________________________________________________________

def find_csv_files(root_dir):
    csv_files = []

    for foldername, subfolders, filenames in os.walk(root_dir):
        subfolders[:] = [folder for folder in subfolders if "archive" not in folder.lower()]
        for filename in filenames:
            if filename.endswith('.csv'):
                csv_files.append(os.path.join(foldername, filename))

    return csv_files


data = pd.DataFrame()
path = ["../results/AttUNet_NoAug", "../results/AttUNet_single", "../results/AttUNet"]
for p in path:
    csv_files = find_csv_files(p)
    for fname in csv_files:
        data_ = pd.read_csv(fname, on_bad_lines='skip')
        data = pd.concat([data, data_])

data['Method'] = data['Method'].str.replace(r'\bAttUNet_checkpoint-28\b', 'Attention U-Net (No Aug)')
data['Method'] = data['Method'].str.replace(r'\bAttUNet_checkpoint-236\b', 'Attention U-Net (Single)')
data['Method'] = data['Method'].str.replace(r'\bAttUNet_checkpoint-232\b', 'Attention U-Net (Single)')
data['Method'] = data['Method'].str.replace(r'\bAttUNet_checkpoint-288\b', 'Attention U-Net (Single)')
data['Method'] = data['Method'].str.replace(r'\bAttUNet_checkpoint-264\b', 'Attention U-Net (All)')

save_path = './figures'

# ____________________________________________________________________________________________________
sns.set_theme(font='sans-serif', font_scale=1)
sns.set_style("whitegrid")
palette = sns.color_palette("deep")

pairs = [
    [('fMRI-ON', 'Attention U-Net (All)'), ('fMRI-ON', 'Attention U-Net (No Aug)')],
    [('fMRI-ON', 'Attention U-Net (All)'), ('fMRI-ON', 'Attention U-Net (Single)')],
    [('ge1_5T', 'Attention U-Net (All)'), ('ge1_5T', 'Attention U-Net (No Aug)')],
    [('ge1_5T', 'Attention U-Net (All)'), ('ge1_5T', 'Attention U-Net (Single)')],
    [('phillips1_5T', 'Attention U-Net (All)'), ('phillips1_5T', 'Attention U-Net (No Aug)')],
    [('phillips1_5T', 'Attention U-Net (All)'), ('phillips1_5T', 'Attention U-Net (Single)')],
    [('siemens1_5T', 'Attention U-Net (All)'), ('siemens1_5T', 'Attention U-Net (No Aug)')],
    [('siemens1_5T', 'Attention U-Net (All)'), ('siemens1_5T', 'Attention U-Net (Single)')],
    [('walthamtrio', 'Attention U-Net (All)'), ('walthamtrio', 'Attention U-Net (No Aug)')],
    [('walthamtrio', 'Attention U-Net (All)'), ('walthamtrio', 'Attention U-Net (Single)')],

]


subcat_order = ["otherscanners", "fMRI-ON"]
states_order = ["Attention U-Net (No Aug)", "Attention U-Net (Single)", "Attention U-Net (All)"]
# ____________________________________________________________________________________________________
fig, axes = plt.subplots(2, 1, figsize=(6, 8))

data_subjwise4 = data[(data['Modality'] == 'otherscanners') | (data['Modality'] == 'fMRI-ON')]

hue_plot_params = {
    'data': data_subjwise4,
    'x': 'Type',
    'y': 'Dice',
    # "order": subcat_order,
    "hue": "Method",
    "hue_order": states_order,
    # "palette": states_palette
}

k0 = sns.boxplot(
    ax=axes[0],
    **hue_plot_params,
    palette=palette,
    flierprops=dict(
        markerfacecolor='0.50',
        markersize=1.5
    ),
    showmeans=True,
    meanprops={
        "marker": "D",
        "markerfacecolor": "yellow",
        "markeredgecolor": "black",
        "markersize": "3"
    }
)

annotator = Annotator(axes[0], pairs, **hue_plot_params)
annotator.configure(test="t-test_paired").apply_and_annotate()


# k0.set_ylabel(r'\textbf{' + k0.get_ylabel() + '}')
# # k0.set_xlabel(r'\textbf{' + k0.get_xlabel() + '}')
# k0.set_xlabel('')
# k0.set(ylim=(0.3, 1))
# k0.yaxis.grid(True, linestyle='--', linewidth=0.5, zorder=0)
# k0.set_xticklabels(k0.get_xticklabels(), rotation=0, ha="center", fontsize=7)

# k0.get_legend().remove()

hue_plot_params2 = {
    'data': data_subjwise4,
    'x': 'Type',
    'y': 'IoU',
    # "order": subcat_order,
    "hue": "Method",
    "hue_order": states_order,
    # "palette": states_palette
}

k1 = sns.boxplot(
    ax=axes[1],
    **hue_plot_params2,
    palette=palette,
    flierprops=dict(
        markerfacecolor='0.50',
        markersize=1.5
    ),
    showmeans=True,
    meanprops={
        "marker": "D",
        "markerfacecolor": "yellow",
        "markeredgecolor": "black",
        "markersize": "3"
    }
)

annotator = Annotator(axes[1], pairs, **hue_plot_params2)
annotator.configure(test="t-test_paired").apply_and_annotate()

# k1.set(ylim=(0.1, 1))

k1.set_ylabel(r'\textbf{' + k1.get_ylabel() + '}')
k1.set_xlabel(r'\textbf{' + k1.get_xlabel() + '}')
k1.yaxis.grid(True, linestyle='--', linewidth=0.5, zorder=0)
k1.set_xticklabels(k1.get_xticklabels(), rotation=0, ha="center", fontsize=7)


plt.tight_layout()

plt.savefig(os.path.join(save_path, 'diceplot_alb.pdf'), bbox_inches='tight', pad_inches=0)
plt.show()
