import matplotlib

# matplotlib.use('TkAgg')
from skimage import measure
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

img_paths = [
    "../datasets/t2mri_dataset/data/abnormality/lvm0090s1/fetus_4.nii.gz",
    "../datasets/t2mri_dataset/data/artifacts/f1131s1/f1131s1_fetus_4.nii.gz",
    "../datasets/t2mri_dataset/data/t2/FCB050s1/fetus_16.nii.gz",
    "../datasets/t2mri_dataset/data/twins/f0663s1/fetus_B_11.nii.gz",
    "../datasets/t2mri_dataset/data/otherscanners/ge1_5T/f0612s1/fetus_2.nii.gz",
    "../datasets/t2mri_dataset/data/otherscanners/phillips1_5T/f0521s1/fetus_301.nii.gz",
    "../datasets/t2mri_dataset/data/otherscanners/siemens1_5T/f0049s1/fetus_3.nii.gz",
    "../datasets/t2mri_dataset/data/otherscanners/walthamtrio/f1150s1/fetus_005.nii.gz",

    # "../datasets/dmri_dataset/data/B0/0853s2_18_vol_0000.nii.gz",
    # "../datasets/dmri_dataset/data/B0/0954s1_15_vol_0000.nii.gz",
    # "../datasets/dmri_dataset/data/B1/1003s1_19_vol_0001.nii.gz",
    # "../datasets/dmri_dataset/data/B1/1193s1_14_vol_0005.nii.gz",
    #
    # "../datasets/fmri_dataset/dMRIs3D/f1107s1/rs-fMRI_120_time_pts_39/vol_0010.nii.gz",
    # "../datasets/fmri_dataset/dMRIs3D/f1145s1/rs-fMRI_80_time_pts_61/vol_0010.nii.gz",
    # "../datasets/fmri_dataset/dMRIs3D/f1200s2/rs-fMRI_80_time_pts_36/vol_0010.nii.gz",
    # # "../datasets/fmri_dataset/dMRIs3D/f1315s1/rs-fMRI_80_time_pts_51/vol_0010.nii.gz",
]

gt_path = [
    "../datasets/t2mri_dataset/data/abnormality/lvm0090s1/goodmask/maskfetus_4_final.nii.gz",
    "../datasets/t2mri_dataset/data/artifacts/f1131s1/goodmask/maskf1131s1_fetus_4_final.nii.gz",
    "../datasets/t2mri_dataset/data/t2/FCB050s1/goodmask/maskfetus_16_final.nii.gz",
    "../datasets/t2mri_dataset/data/twins/f0663s1/goodmask/maskfetus_B_11_final.nii.gz",
    "../datasets/t2mri_dataset/data/otherscanners/ge1_5T/f0612s1/goodmask/maskfetus_2_final.nii.gz",
    "../datasets/t2mri_dataset/data/otherscanners/phillips1_5T/f0521s1/goodmask/maskfetus_301_final.nii.gz",
    "../datasets/t2mri_dataset/data/otherscanners/siemens1_5T/f0049s1/goodmask/maskfetus_3_final.nii.gz",
    "../datasets/t2mri_dataset/data/otherscanners/walthamtrio/f1150s1/goodmask/maskfetus_005_final.nii.gz",

    # "../datasets/dmri_dataset/data/B0/goodmask/0853s2_18_vol_0000_mask_final.nii.gz",
    # "../datasets/dmri_dataset/data/B0/goodmask/0954s1_15_vol_0000_mask_final.nii.gz",
    # "../datasets/dmri_dataset/data/B1/goodmask/1003s1_19_vol_0001_mask_final.nii.gz",
    # "../datasets/dmri_dataset/data/B1/goodmask/1193s1_14_vol_0005_mask_final.nii.gz",
    #
    # "../datasets/fmri_dataset/dMRIs3D/f1107s1/rs-fMRI_120_time_pts_39/vol_0010_mask_final.nii.gz",
    # "../datasets/fmri_dataset/dMRIs3D/f1145s1/rs-fMRI_80_time_pts_61/vol_0010_mask_final.nii.gz",
    # "../datasets/fmri_dataset/dMRIs3D/f1200s2/rs-fMRI_80_time_pts_36/vol_0010_mask_final.nii.gz",
    # # "../datasets/fmri_dataset/dMRIs3D/f1315s1/rs-fMRI_80_time_pts_51/vol_0010_mask_final.nii.gz",
]

UNet_pred_path = [
    "../results/BasicUNet/t2mri_dataset/data/abnormality/lvm0090s1/fetus_4_pred.nii.gz",
    "../results/BasicUNet/t2mri_dataset/data/artifacts/f1131s1/f1131s1_fetus_4_pred.nii.gz",
    "../results/BasicUNet/t2mri_dataset/data/t2/FCB050s1/fetus_16_pred.nii.gz",
    "../results/BasicUNet/t2mri_dataset/data/twins/f0663s1/fetus_B_11_pred.nii.gz",
    "../results/BasicUNet/t2mri_dataset/data/otherscanners/ge1_5T/f0612s1/fetus_2_pred.nii.gz",
    "../results/BasicUNet/t2mri_dataset/data/otherscanners/phillips1_5T/f0521s1/fetus_301_pred.nii.gz",
    "../results/BasicUNet/t2mri_dataset/data/otherscanners/siemens1_5T/f0049s1/fetus_3_pred.nii.gz",
    "../results/BasicUNet/t2mri_dataset/data/otherscanners/walthamtrio/f1150s1/fetus_005_pred.nii.gz",
    #
    #     "../results/BasicUNet/dmri_dataset/data/B0/0853s2_18_vol_0000_pred.nii.gz",
    #     "../results/BasicUNet/dmri_dataset/data/B0/0954s1_15_vol_0000_pred.nii.gz",
    #     "../results/BasicUNet/dmri_dataset/data/B1/1003s1_19_vol_0005_pred.nii.gz",
    #     "../results/BasicUNet/dmri_dataset/data/B1/1193s1_14_vol_0005_pred.nii.gz",
    #
    #     "../results/BasicUNet/fmri_dataset/dMRIs3D/f1107s1/rs-fMRI_120_time_pts_39/vol_0010_pred.nii.gz",
    #     "../results/BasicUNet/fmri_dataset/dMRIs3D/f1145s1/rs-fMRI_80_time_pts_61/vol_0010_pred.nii.gz",
    #     "../results/BasicUNet/fmri_dataset/dMRIs3D/f1200s2/rs-fMRI_80_time_pts_36/vol_0010_pred.nii.gz",
    #     # "../results/BasicUNet/fmri_dataset/dMRIs3D/f1315s1/rs-fMRI_80_time_pts_51/vol_0010_pred.nii.gz",
]

DynNet_pred_path = [
    "../results/DynUNet/t2mri_dataset/data/abnormality/lvm0090s1/fetus_4_pred.nii.gz",
    "../results/DynUNet/t2mri_dataset/data/artifacts/f1131s1/f1131s1_fetus_4_pred.nii.gz",
    "../results/DynUNet/t2mri_dataset/data/t2/FCB050s1/fetus_16_pred.nii.gz",
    "../results/DynUNet/t2mri_dataset/data/twins/f0663s1/fetus_B_11_pred.nii.gz",
    "../results/DynUNet/t2mri_dataset/data/otherscanners/ge1_5T/f0612s1/fetus_2_pred.nii.gz",
    "../results/DynUNet/t2mri_dataset/data/otherscanners/phillips1_5T/f0521s1/fetus_301_pred.nii.gz",
    "../results/DynUNet/t2mri_dataset/data/otherscanners/siemens1_5T/f0049s1/fetus_3_pred.nii.gz",
    "../results/DynUNet/t2mri_dataset/data/otherscanners/walthamtrio/f1150s1/fetus_005_pred.nii.gz",

    # "../results/DynUNet/dmri_dataset/data/B0/0853s2_18_vol_0000_pred.nii.gz",
    # "../results/DynUNet/dmri_dataset/data/B0/0954s1_15_vol_0000_pred.nii.gz",
    # "../results/DynUNet/dmri_dataset/data/B1/1003s1_19_vol_0005_pred.nii.gz",
    # "../results/DynUNet/dmri_dataset/data/B1/1193s1_14_vol_0005_pred.nii.gz",
    #
    # "../results/DynUNet/fmri_dataset/dMRIs3D/f1107s1/rs-fMRI_120_time_pts_39/vol_0010_pred.nii.gz",
    # "../results/DynUNet/fmri_dataset/dMRIs3D/f1145s1/rs-fMRI_80_time_pts_61/vol_0010_pred.nii.gz",
    # "../results/DynUNet/fmri_dataset/dMRIs3D/f1200s2/rs-fMRI_80_time_pts_36/vol_0010_pred.nii.gz",
    # # "../results/DynUNet/fmri_dataset/dMRIs3D/f1315s1/rs-fMRI_80_time_pts_51/vol_0010_pred.nii.gz",
]

AttUNet_pred_path = [
    "../results/AttUNet/t2mri_dataset/data/abnormality/lvm0090s1/fetus_4_pred.nii.gz",
    "../results/AttUNet/t2mri_dataset/data/artifacts/f1131s1/f1131s1_fetus_4_pred.nii.gz",
    "../results/AttUNet/t2mri_dataset/data/t2/FCB050s1/fetus_16_pred.nii.gz",
    "../results/AttUNet/t2mri_dataset/data/twins/f0663s1/fetus_B_11_pred.nii.gz",
    "../results/AttUNet/t2mri_dataset/data/otherscanners/ge1_5T/f0612s1/fetus_2_pred.nii.gz",
    "../results/AttUNet/t2mri_dataset/data/otherscanners/phillips1_5T/f0521s1/fetus_301_pred.nii.gz",
    "../results/AttUNet/t2mri_dataset/data/otherscanners/siemens1_5T/f0049s1/fetus_3_pred.nii.gz",
    "../results/AttUNet/t2mri_dataset/data/otherscanners/walthamtrio/f1150s1/fetus_005_pred.nii.gz",

    # "../results/AttUNet/dmri_dataset/data/B0/0853s2_18_vol_0000_pred.nii.gz",
    # "../results/AttUNet/dmri_dataset/data/B0/0954s1_15_vol_0000_pred.nii.gz",
    # "../results/AttUNet/dmri_dataset/data/B1/1003s1_19_vol_0005_pred.nii.gz",
    # "../results/AttUNet/dmri_dataset/data/B1/1193s1_14_vol_0005_pred.nii.gz",
    #
    # "../results/AttUNet/fmri_dataset/dMRIs3D/f1107s1/rs-fMRI_120_time_pts_39/vol_0010_pred.nii.gz",
    # "../results/AttUNet/fmri_dataset/dMRIs3D/f1145s1/rs-fMRI_80_time_pts_61/vol_0010_pred.nii.gz",
    # "../results/AttUNet/fmri_dataset/dMRIs3D/f1200s2/rs-fMRI_80_time_pts_36/vol_0010_pred.nii.gz",
    # # "../results/AttUNet/fmri_dataset/dMRIs3D/f1315s1/rs-fMRI_80_time_pts_51/vol_0010_pred.nii.gz",
]

slice_number = [
    17,
    16,
    11,
    3,

    13,
    13,
    7,
    14,

    # 22,
    # 15,
    # 12,
    # 11,
    #
    # 20,
    # 19,
    # 18,
    # # 10,
]

titles = [
    r"\textbf{'T2W-Abnormality'}",
    r"\textbf{'T2W-Artifacts'}",
    r"\textbf{'T2W-Representative'}",
    r"\textbf{'T2W-Twins'}",

    r"\textbf{'T2W-Ge1.5T'}",
    r"\textbf{'T2W-Phillips1.5T'}",
    r"\textbf{'T2W-Siemens1.5T'}",
    r"\textbf{'T2W-Siemens1.5T-SiteW'}",
    #
    # r"\textbf{'DWI-B0'}",
    # r"\textbf{'DWI-B0'}",
    # r"\textbf{'DWI-B1'}",
    # r"\textbf{'DWI-B1'}",
    #
    # r"\textbf{'fMRI'}",
    # r"\textbf{'fMRI'}",
    # r"\textbf{'fMRI'}",
]

titles_ = [
    r"\textbf{U-Net}",
    r"\textbf{DynU-Net}",
    r"\textbf{Att-U-Net}",
]

plt.interactive(False)
cmap_mask = matplotlib.colors.ListedColormap(['none', 'red'])
cmap_gt = matplotlib.colors.ListedColormap(['none', 'blue'])

fig, axes = plt.subplots(nrows=8, ncols=3, sharex=False, sharey=False, constrained_layout=False, figsize=(8, 19))

pred_path = UNet_pred_path + DynNet_pred_path + AttUNet_pred_path
img_paths = img_paths * 3
gt_path = gt_path * 3
slice_number = slice_number * 3
titles = titles * 3

c = 0
t = 0
for j, col in enumerate(axes.T):  # Using axes.T to transpose and iterate over columns
    for i, cell in enumerate(col):
        selected_slice = nib.load(img_paths[c]).get_fdata()[:, :, slice_number[c]]
        pred = nib.load(pred_path[c]).get_fdata()[:, :, slice_number[c]]
        gt = nib.load(gt_path[c]).get_fdata()[:, :, slice_number[c]]

        contours_pred = measure.find_contours(pred, 0.5)
        contours_gt = measure.find_contours(gt, 0.5)

        cell.imshow(selected_slice, aspect="auto", origin='lower', cmap='gray')
        # cell.imshow(pred, alpha=1, cmap=cmap_gt)
         

        cell.get_xaxis().set_ticks([])
        cell.get_yaxis().set_ticks([])
        cell.margins(x=0)

        cell.text(0.05, 0.95, titles[c], color='white', ha='left', va='top', transform=cell.transAxes)

        cell.axis('off')
        c += 1

        if i == 0 and (j == 0 or j == 1 or j == 2):
            cell.set_title(titles_[t], fontsize=20)
            t += 1

#
# fig.text(0.2, 0.75, r'\textbf{Normal}', fontsize=15, va='center', rotation='vertical')
# fig.text(0.2, 0.25, r'\textbf{Challenging}', fontsize=15, va='center', rotation='vertical')
plt.tight_layout()
# plt.subplots_adjust(pad=-5.0)
plt.subplots_adjust(wspace=0.01, hspace=0.04)
save_path = './figs/'
plt.savefig('temp1.pdf', bbox_inches='tight', pad_inches=0)
plt.show()
