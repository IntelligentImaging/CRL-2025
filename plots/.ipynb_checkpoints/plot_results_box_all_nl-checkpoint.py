import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from statannotations.Annotator import Annotator

# # Update fonts to bold
# plt.rc('font', family='serif')
# plt.rcParams['font.weight'] = 'bold'
# plt.rcParams['axes.labelweight'] = 'bold'
# plt.rcParams['axes.titleweight'] = 'bold'

def find_csv_files(root_dir):
    csv_files = []
    for foldername, subfolders, filenames in os.walk(root_dir):
        subfolders[:] = [folder for folder in subfolders if "archive" not in folder.lower()]
        for filename in filenames:
            if filename.endswith('.csv'):
                csv_files.append(os.path.join(foldername, filename))
    return csv_files

data = pd.DataFrame()
path = ["../results/AttUNet2", "../results/BasicUNet2", "../results/DynUNet2"]
for p in path:
    csv_files = find_csv_files(p)
    for fname in csv_files:
        data_ = pd.read_csv(fname, on_bad_lines='skip')
        data = pd.concat([data, data_])

data = data.rename(columns={'Modality': 'MRI Sequence'})
data = data.rename(columns={'Dice': 'ASD'})
data = data.rename(columns={'IoU': 'HD95'})

data['Method'] = data['Method'].str.replace('BasicUNet_checkpoint-284', 'U-Net')
data['Method'] = data['Method'].str.replace('AttUNet_checkpoint-264', 'Attention U-Net')
data['Method'] = data['Method'].str.replace('DynUNet_checkpoint-288', 'Dynamic U-Net')

data['Type'] = data['Type'].str.replace('abnormality', 'T2W-Abnormality')
data['Type'] = data['Type'].str.replace('artifacts', 'T2W-Artifacts')
data['Type'] = data['Type'].str.replace('t2', 'T2W-Typical')
data['Type'] = data['Type'].str.replace('twins', 'T2W-Twins')

data['Type'] = data['Type'].str.replace('ge1_5T', 'T2W-Ge1.5T')
data['Type'] = data['Type'].str.replace('phillips1_5T', 'T2W-Phillips1.5T')
data['Type'] = data['Type'].str.replace('siemens1_5T', 'T2W-Siemens1.5T')
data['Type'] = data['Type'].str.replace('walthamtrio', 'T2W-Siemens3T-SiteW')

data['Type'] = data['Type'].str.replace('B0', 'DWI-B0')
data['Type'] = data['Type'].str.replace('B1', 'DWI-B1')

save = True
save_path = './figures'

hue_order = ["U-Net", "Dynamic U-Net", "Attention U-Net"]

pairs = [
    [('T2W', 'Attention U-Net'), ('T2W', 'U-Net')],
    [('T2W', 'Attention U-Net'), ('T2W', 'Dynamic U-Net')],
    [('DWI', 'Attention U-Net'), ('DWI', 'U-Net')],
    [('DWI', 'Attention U-Net'), ('DWI', 'Dynamic U-Net')],
    [('fMRI', 'Attention U-Net'), ('fMRI', 'U-Net')],
    [('fMRI', 'Attention U-Net'), ('fMRI', 'Dynamic U-Net')],
]

fig, axes = plt.subplots(2, 3, figsize=(18, 7))
sns.set_theme(font='sans-serif', font_scale=1)
sns.set_style("whitegrid")
palette = sns.color_palette("deep")

# Dice 0, HD95 0
data_ = data[(data['MRI Sequence'] != 'otherscanners') & (data['MRI Sequence'] != 'fMRI-ON')]
x_order = ["T2W", "DWI", "fMRI"]
plot_params_dice = {
    'data': data_,
    'x': 'MRI Sequence',
    'y': 'ASD',
    "order": x_order,
    "hue": "Method",
    "hue_order": hue_order,
    "palette": palette,
    "flierprops": dict(markerfacecolor='0.50', markersize=1.5),
    "showmeans": True,
    "meanprops": {"marker": "D", "markerfacecolor": "yellow", "markeredgecolor": "black", "markersize": "3"}
}

plot_params_iou = {
    'data': data_,
    'x': 'MRI Sequence',
    'y': 'HD95',
    "order": x_order,
    "hue": "Method",
    "hue_order": hue_order,
    "palette": palette,
    "flierprops": dict(markerfacecolor='0.50', markersize=1.5),
    "showmeans": True,
    "meanprops": {"marker": "D", "markerfacecolor": "yellow", "markeredgecolor": "black", "markersize": "3"}
}

g00 = sns.boxplot(ax=axes[0, 0], **plot_params_dice)
g00.set(ylim=(0.0, 40.0))
annotator = Annotator(axes[0, 0], pairs, **plot_params_dice)
annotator.configure(test="t-test_paired").apply_and_annotate()

g00.set_xlabel('')
g00.set_ylabel(g00.get_ylabel())

g10 = sns.boxplot(ax=axes[1, 0], **plot_params_iou)
# g10.set(ylim=(0.3, 1))
g10.set_ylabel(g10.get_ylabel())
g10.set_xlabel(g10.get_xlabel())

# Dice 1, HD95 1
data_ = data[data['MRI Sequence'] == 'T2W']
x_order = ["T2W-Typical", "T2W-Abnormality", "T2W-Artifacts", "T2W-Twins"]
plot_params_dice = {
    'data': data_,
    'x': 'Type',
    'y': 'ASD',
    "order": x_order,
    "hue": "Method",
    "hue_order": hue_order,
    "palette": palette,
    "flierprops": dict(markerfacecolor='0.50', markersize=1.5),
    "showmeans": True,
    "meanprops": {"marker": "D", "markerfacecolor": "yellow", "markeredgecolor": "black", "markersize": "3"}
}

plot_params_iou = {
    'data': data_,
    'x': 'Type',
    'y': 'HD95',
    "order": x_order,
    "hue": "Method",
    "hue_order": hue_order,
    "palette": palette,
    "flierprops": dict(markerfacecolor='0.50', markersize=1.5),
    "showmeans": True,
    "meanprops": {"marker": "D", "markerfacecolor": "yellow", "markeredgecolor": "black", "markersize": "3"}
}

g01 = sns.boxplot(ax=axes[0, 1], **plot_params_dice)
g01.set(ylim=(0.0, 40.0))
g01.set_ylabel('')
g01.set_xlabel('')
g01.set_xticklabels(g01.get_xticklabels(), rotation=0, ha="center", fontsize=10)

g11 = sns.boxplot(ax=axes[1, 1], **plot_params_iou)
# g11.set(ylim=(0.3, 1))
g11.set_ylabel('')
g11.set_xlabel(g11.get_xlabel())
g11.set_xticklabels(g11.get_xticklabels(), rotation=0, ha="center", fontsize=10)

# Dice 2, HD95 2
data_ = data[data['MRI Sequence'] == 'DWI']
x_order = ["DWI-B0", "DWI-B1"]
plot_params_dice = {
    'data': data_,
    'x': 'Type',
    'y': 'ASD',
    "order": x_order,
    "hue": "Method",
    "hue_order": hue_order,
    "palette": palette,
    "flierprops": dict(markerfacecolor='0.50', markersize=1.5),
    "showmeans": True,
    "meanprops": {"marker": "D", "markerfacecolor": "yellow", "markeredgecolor": "black", "markersize": "3"}
}

plot_params_iou = {
    'data': data_,
    'x': 'Type',
    'y': 'HD95',
    "order": x_order,
    "hue": "Method",
    "hue_order": hue_order,
    "palette": palette,
    "flierprops": dict(markerfacecolor='0.50', markersize=1.5),
    "showmeans": True,
    "meanprops": {"marker": "D", "markerfacecolor": "yellow", "markeredgecolor": "black", "markersize": "3"}
}

g02 = sns.boxplot(ax=axes[0, 2], **plot_params_dice)
# g02.set(ylim=(0.7, 1))
g02.set_ylabel('')
g02.set_xlabel('')

g12 = sns.boxplot(ax=axes[1, 2], **plot_params_iou)
# g12.set(ylim=(0.5, 1))
g12.set_ylabel('')
g12.set_xlabel(g12.get_xlabel())

for ax in [g00, g01, g10, g11, g02, g12]:
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, zorder=0)
    ax.get_legend().remove()

handles, labels = g00.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.525, -0.00))
plt.tight_layout()
plt.subplots_adjust(bottom=0.125)

if save:
    plt.savefig(os.path.join(save_path, 'diceplot_nl.pdf'), bbox_inches='tight', pad_inches=0)

plt.show()