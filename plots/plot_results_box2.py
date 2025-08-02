import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob

from statannotations.Annotator import Annotator

# # Update fonts to bold and use LaTeX
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
# plt.rcParams['font.weight'] = 'bold'
# plt.rcParams['axes.labelweight'] = 'bold'
# plt.rcParams['axes.titleweight'] = 'bold'


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

# ____________________________________________________________________________________________________
data = data.rename(columns={'Modality': 'MRI Sequence'})
data = data.rename(columns={'Dice': 'DSC'})

data['Method'] = data['Method'].str.replace(r'\bAttUNet_checkpoint-28\b', 'Attention U-Net (All, No Aug)')
data['Method'] = data['Method'].str.replace(r'\bAttUNet_checkpoint-236\b', 'Attention U-Net (Single, Aug)')
data['Method'] = data['Method'].str.replace(r'\bAttUNet_checkpoint-232\b', 'Attention U-Net (Single, Aug)')
data['Method'] = data['Method'].str.replace(r'\bAttUNet_checkpoint-288\b', 'Attention U-Net (Single, Aug)')
data['Method'] = data['Method'].str.replace(r'\bAttUNet_checkpoint-264\b', 'Attention U-Net (All, Aug)')

data['Type'] = data['Type'].str.replace('ge1_5T', 'T2W-Ge1.5T')
data['Type'] = data['Type'].str.replace('phillips1_5T', 'T2W-Phillips1.5T')
data['Type'] = data['Type'].str.replace('siemens1_5T', 'T2W-Siemens1.5T')
data['Type'] = data['Type'].str.replace('walthamtrio', 'T2W-Siemens3T-SiteW')

data['Type'] = data['Type'].str.replace('B0', 'DWI-B0')
data['Type'] = data['Type'].str.replace('B1', 'DWI-B1')

data['Type'] = data['Type'].str.replace('fMRI-ON', 'fMRI-SiteD')

save = True
save_path = './figures'

hue_order = ["Attention U-Net (Single, Aug)", "Attention U-Net (All, No Aug)", "Attention U-Net (All, Aug)"]


pairs_all = [
    [('T2W', 'Attention U-Net (All, Aug)'), ('T2W', 'Attention U-Net (Single, Aug)')],
    [('T2W', 'Attention U-Net (All, Aug)'), ('T2W', 'Attention U-Net (All, No Aug)')],
    [('DWI', 'Attention U-Net (All, Aug)'), ('DWI', 'Attention U-Net (Single, Aug)')],
    [('DWI', 'Attention U-Net (All, Aug)'), ('DWI', 'Attention U-Net (All, No Aug)')],
    [('fMRI', 'Attention U-Net (All, Aug)'), ('fMRI', 'Attention U-Net (Single, Aug)')],
    [('fMRI', 'Attention U-Net (All, Aug)'), ('fMRI', 'Attention U-Net (All, No Aug)')],

]

pairs_indp = [
    [('fMRI-SiteD', 'Attention U-Net (All, Aug)'), ('fMRI-SiteD', 'Attention U-Net (All, No Aug)')],
    [('fMRI-SiteD', 'Attention U-Net (All, Aug)'), ('fMRI-SiteD', 'Attention U-Net (Single, Aug)')],
    [('T2W-Ge1.5T', 'Attention U-Net (All, Aug)'), ('T2W-Ge1.5T', 'Attention U-Net (All, No Aug)')],
    [('T2W-Ge1.5T', 'Attention U-Net (All, Aug)'), ('T2W-Ge1.5T', 'Attention U-Net (Single, Aug)')],
    [('T2W-Phillips1.5T', 'Attention U-Net (All, Aug)'), ('T2W-Phillips1.5T', 'Attention U-Net (All, No Aug)')],
    [('T2W-Phillips1.5T', 'Attention U-Net (All, Aug)'), ('T2W-Phillips1.5T', 'Attention U-Net (Single, Aug)')],
    [('T2W-Siemens1.5T', 'Attention U-Net (All, Aug)'), ('T2W-Siemens1.5T', 'Attention U-Net (All, No Aug)')],
    [('T2W-Siemens1.5T', 'Attention U-Net (All, Aug)'), ('T2W-Siemens1.5T', 'Attention U-Net (Single, Aug)')],
    [('T2W-Siemens3T-SiteW', 'Attention U-Net (All, Aug)'), ('T2W-Siemens3T-SiteW', 'Attention U-Net (All, No Aug)')],
    [('T2W-Siemens3T-SiteW', 'Attention U-Net (All, Aug)'), ('T2W-Siemens3T-SiteW', 'Attention U-Net (Single, Aug)')],

]



# ____________________________________________________________________________________________________

fig, axes = plt.subplots(2, 2, figsize=(10, 7))
sns.set_theme(font='sans-serif', font_scale=1)
sns.set_style("whitegrid")
palette = sns.color_palette("deep")

# ____________________________________________________________________________________________________
# Dice 0, IoU 0
data_ = data[(data['MRI Sequence'] != 'otherscanners') & (data['MRI Sequence'] != 'fMRI-ON')]
x_order = ["T2W", "DWI", "fMRI"]
plot_params_dice = {
    'data': data_,
    'x': 'MRI Sequence',
    'y': 'DSC',
    "order": x_order,
    "hue": "Method",
    "hue_order": hue_order,
    "palette": palette,
    "flierprops": dict(
        markerfacecolor='0.50',
        markersize=1.5
    ),
    "showmeans": True,
    "meanprops": {
        "marker": "D",
        "markerfacecolor": "yellow",
        "markeredgecolor": "black",
        "markersize": "3"
    }
}

plot_params_iou = {
    'data': data_,
    'x': 'MRI Sequence',
    'y': 'IoU',
    "order": x_order,
    "hue": "Method",
    "hue_order": hue_order,
    "palette": palette,
    "flierprops": dict(
        markerfacecolor='0.50',
        markersize=1.5
    ),
    "showmeans": True,
    "meanprops": {
        "marker": "D",
        "markerfacecolor": "yellow",
        "markeredgecolor": "black",
        "markersize": "3"
    }
}

g00 = sns.boxplot(ax=axes[0, 0], **plot_params_dice)
annotator = Annotator(axes[0, 0], pairs_all, **plot_params_dice)
annotator.configure(test="t-test_paired").apply_and_annotate()
g00.set(ylim=(0.4, 1.2))
g00.set_xlabel('')
g00.set_ylabel(r'\textbf{' + g00.get_ylabel() + '}')

g10 = sns.boxplot(ax=axes[1, 0], **plot_params_iou)
g10.set(ylim=(0.3, 1))
g10.set_ylabel(r'\textbf{' + g10.get_ylabel() + '}')
g10.set_xlabel(r'\textbf{' + g10.get_xlabel() + '}')

# ____________________________________________________________________________________________________
# Dice 1, IoU 1
data_ = data[(data['MRI Sequence'] == 'otherscanners') | (data['MRI Sequence'] == 'fMRI-ON')]
plot_params_dice = {
    'data': data_,
    'x': 'Type',
    'y': 'DSC',
    # "order": x_order,
    "hue": "Method",
    "hue_order": hue_order,
    "palette": palette,
    "flierprops": dict(
        markerfacecolor='0.50',
        markersize=1.5
    ),
    "showmeans": True,
    "meanprops": {
        "marker": "D",
        "markerfacecolor": "yellow",
        "markeredgecolor": "black",
        "markersize": "3"
    }
}

plot_params_iou = {
    'data': data_,
    'x': 'Type',
    'y': 'IoU',
    # "order": x_order,
    "hue": "Method",
    "hue_order": hue_order,
    "palette": palette,
    "flierprops": dict(
        markerfacecolor='0.50',
        markersize=1.5
    ),
    "showmeans": True,
    "meanprops": {
        "marker": "D",
        "markerfacecolor": "yellow",
        "markeredgecolor": "black",
        "markersize": "3"
    }
}

g01 = sns.boxplot(ax=axes[0, 1], **plot_params_dice)
annotator = Annotator(axes[0, 1], pairs_indp, **plot_params_dice)
annotator.configure(test="t-test_paired").apply_and_annotate()
g01.set(ylim=(0.4, 1.2))
g01.set_ylabel('')
g01.set_xlabel('')
g01.set_xticklabels(g01.get_xticklabels(), rotation=0, ha="center", fontsize=10)

g11 = sns.boxplot(ax=axes[1, 1], **plot_params_iou)
g11.set(ylim=(0.3, 1))
g11.set_ylabel('')
g11.set_xlabel(r'\textbf{' + g11.get_xlabel() + '}')
g11.set_xticklabels(g11.get_xticklabels(), rotation=0, ha="center", fontsize=10)

# ____________________________________________________________________________________________________
for ax in [g00, g01, g10, g11]:
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, zorder=0)
    ax.get_legend().remove()

handles, labels = g00.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.525, -0.00))
plt.tight_layout()
plt.subplots_adjust(bottom=0.125)

if save:
    plt.savefig(os.path.join(save_path, 'diceplot_alb_nl.pdf'), bbox_inches='tight', pad_inches=0)

plt.show()
