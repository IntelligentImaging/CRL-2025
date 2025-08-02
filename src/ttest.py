import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from scipy.stats import ttest_ind, ttest_rel


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
path = ["../results"] #AttUNet", "../results/AttUNet_single"]  # ["../results/AttUNet_NoAug", "../results/AttUNet_single", "../results/AttUNet"]
for p in path:
    csv_files = find_csv_files(p)
    for fname in csv_files:
        data_ = pd.read_csv(fname, on_bad_lines='skip')
        data = pd.concat([data, data_])

data = data[(data['Modality'] != 'otherscanners') & (data['Modality'] != 'fMRI-ON')]

data['Method'] = data['Method'].str.replace(r'\bBasicUNet_checkpoint-284\b', 'U-Net')
data['Method'] = data['Method'].str.replace(r'\bAttUNet_checkpoint-264\b', 'Attention U-Net')
data['Method'] = data['Method'].str.replace(r'\bDynUNet_checkpoint-288\b', 'Dynamic U-Net')
data['Method'] = data['Method'].str.replace(r'\bAttUNet_checkpoint-28\b', 'Attention U-Net (No Aug)')
data['Method'] = data['Method'].str.replace(r'\bAttUNet_checkpoint-236\b', 'Attention U-Net (Single)')
data['Method'] = data['Method'].str.replace(r'\bAttUNet_checkpoint-232\b', 'Attention U-Net (Single)')
data['Method'] = data['Method'].str.replace(r'\bAttUNet_checkpoint-288\b', 'Attention U-Net (Single)')

dice_unet = data[data['Method'] == 'U-Net']['Dice']
dice_attention_unet = data[data['Method'] == 'Attention U-Net']['Dice']
dice_dynunet = data[data['Method'] == 'Dynamic U-Net']['Dice']
dice_attention_unet_no_aug = data[data['Method'] == 'Attention U-Net (No Aug)']['Dice']
dice_attention_unet_single = data[data['Method'] == 'Attention U-Net (Single)']['Dice']

######
# For comparing U-Net and Attention U-Net:
t_statistic, p_value = ttest_rel(dice_unet, dice_attention_unet)
print(f"Comparison between U-Net and Attention U-Net: t = {t_statistic:.2f}, p = {p_value:.5f}")

# For comparing U-Net and DynU-Net:
t_statistic, p_value = ttest_rel(dice_unet, dice_dynunet)
print(f"Comparison between U-Net and DynU-Net: t = {t_statistic:.2f}, p = {p_value:.5f}")

# For comparing Attention U-Net and DynU-Net:
t_statistic, p_value = ttest_rel(dice_attention_unet, dice_dynunet)
print(f"Comparison between Attention U-Net and DynU-Net: t = {t_statistic:.2f}, p = {p_value:.5f}")

# For comparing Attention U-Net and Attention U-Net (NoAug):
t_statistic, p_value = ttest_rel(dice_attention_unet, dice_attention_unet_no_aug)
print(f"Comparison between Attention U-Net and Attention U-Net (NoAug): t = {t_statistic:.2f}, p = {p_value:.5f}")

# For comparing Attention U-Net and Attention U-Net (single):
t_statistic, p_value = ttest_rel(dice_attention_unet, dice_attention_unet_single)
print(f"Comparison between Attention U-Net and Attention U-Net (single): t = {t_statistic:.2f}, p = {p_value:.5f}")

#
# import os
# import pandas as pd
# import numpy as np
# from scipy.stats import ttest_rel
#
# input_directory = "../results_FeTA"
# output_summary_file = "summary_statistics.csv"
# output_stats_file = "umamba_vs_stats.csv"
#
# summary_data = []
#
# def process_metrics(metric_str):
#     values = eval(metric_str)
#     return np.mean(values), np.std(values), values
#
# all_metrics_data = []
# for csv_file in os.listdir(input_directory):
#     if csv_file.endswith(".csv"):
#         method_name = os.path.splitext(csv_file)[0]
#         file_path = os.path.join(input_directory, csv_file)
#         data = pd.read_csv(file_path)
#         for index, row in data.iterrows():
#             dice_mean, dice_std, dice_values = process_metrics(row["Dice"])
#             iou_mean, iou_std, iou_values = process_metrics(row["IoU"])
#             hd_mean, hd_std, hd_values = process_metrics(row["HD"])
#             summary_data.append({
#                 "Method": row["Method"],
#                 "Modality": row["Modality"],
#                 "Type": row["Type"],
#                 "Subject": row["Subject"],
#                 "Dice Mean": dice_mean,
#                 "Dice Std": dice_std,
#                 "IoU Mean": iou_mean,
#                 "IoU Std": iou_std,
#                 "HD Mean": hd_mean,
#                 "HD Std": hd_std
#             })
#             all_metrics_data.append({
#                 "Method": row["Method"],
#                 "Modality": row["Modality"],
#                 "Type": row["Type"],
#                 "Metric": "Dice",
#                 "Values": dice_values
#             })
#             all_metrics_data.append({
#                 "Method": row["Method"],
#                 "Modality": row["Modality"],
#                 "Type": row["Type"],
#                 "Metric": "IoU",
#                 "Values": iou_values
#             })
#             all_metrics_data.append({
#                 "Method": row["Method"],
#                 "Modality": row["Modality"],
#                 "Type": row["Type"],
#                 "Metric": "HD",
#                 "Values": hd_values
#             })
#
# summary_df = pd.DataFrame(summary_data)
# summary_df.to_csv(output_summary_file, index=False)
#
# all_metrics_expanded = []
# for row in all_metrics_data:
#     for value in row["Values"]:
#         all_metrics_expanded.append({
#             "Method": row["Method"],
#             "Modality": row["Modality"],
#             "Type": row["Type"],
#             "Metric": row["Metric"],
#             "Value": value
#         })
#
# metrics_df = pd.DataFrame(all_metrics_expanded)
#
# stats_results = []
# for metric in metrics_df["Metric"].unique():
#     metric_data = metrics_df[metrics_df["Metric"] == metric]
#     umamba_data = metric_data[metric_data["Method"] == "UMAMBA"]["Value"]
#     other_methods = metric_data[metric_data["Method"] != "UMAMBA"]["Method"].unique()
#     for method in other_methods:
#         method_data = metric_data[metric_data["Method"] == method]["Value"]
#         t_stat, p_value = ttest_rel(umamba_data, method_data)
#         stats_results.append({
#             "Metric": metric,
#             "UMAMBA_vs": method,
#             "T-Statistic": t_stat,
#             "P-Value": p_value
#         })
#
# stats_df = pd.DataFrame(stats_results)
# stats_df.to_csv(output_stats_file, index=False)
#
# print(f"Summary statistics saved to {output_summary_file}")
# print(f"UMAMBA statistical analysis results saved to {output_stats_file}")
