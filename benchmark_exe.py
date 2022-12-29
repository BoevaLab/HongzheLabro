import torch
import scvi
import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
import scipy

all_adver_best = best_result_selection("all", "adver")
lung_adver_best = best_result_selection("lung", "adver")
pancreas_adver_best = best_result_selection("pancreas", "adver")
all_general_scvi_best = best_result_selection("all", "general_scvi")
lung_general_scvi_best = best_result_selection("lung", "general_scvi")
pancreas_general_scvi_best = best_result_selection("pancreas", "general_scvi")
all_No_adver_best = best_result_selection("all", "No_adver")
lung_No_adver_best = best_result_selection("lung", "No_adver")
pancreas_No_adver_best = best_result_selection("pancreas", "No_adver")

score_pd = compare_method(all_adver_best, all_general_scvi_best, all_No_adver_best, lung_adver_best, lung_general_scvi_best, lung_No_adver_best,
pancreas_adver_best, pancreas_general_scvi_best, pancreas_No_adver_best, ["adver_all","general_scvi_all", "No_adver_all", "adver_lung","general_scvi_lung","No_adver_lung",
"adver_pancreas","general_scvi_pancreas","No_adver_pancreas"], True, "grid_search_result.csv")

score_pd_T = score_pd.T

ax = score_pd_T.plot.barh(figsize=(12,15))
ax.invert_yaxis() 
ax.legend(bbox_to_anchor=(1, 1))
for container in ax.containers:
    ax.bar_label(container, fmt = '%.3f')

score_adver = mean_std(score_pd, "adver")
score_general_scvi = mean_std(score_pd, "general_scvi")
score_no_adver = mean_std(score_pd, "no_adver")

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.scatter(score_no_adver[0], score_no_adver[2])
plt.errorbar(score_no_adver[0], score_no_adver[2], xerr = score_no_adver[1],yerr=score_no_adver[3], label = 'batch_rep_no_adver')

ax.scatter(score_adver[0], score_adver[2])
plt.errorbar(score_adver[0], score_adver[2], xerr = score_adver[1],yerr=score_adver[3], label = 'batch_rep_adver')

ax.scatter(score_general_scvi[0], score_general_scvi[2])
plt.errorbar(score_general_scvi[0], score_general_scvi[2], xerr = score_general_scvi[1],yerr=score_general_scvi[3], label = 'general_scvi')
plt.legend(loc='upper left')
plt.xlabel('Bio conservation')
plt.ylabel('Batch correction')
plt.show()

hetero_plot("removed_batches_all_general_scvi.csv", 
"general_scvi_all_batch_0.csv", 
"general_scvi_all_batch_1.csv", 
"general_scvi_all_batch_2.csv", 
"general_scvi_all_batch_3.csv",
dataset = 'all',
method = 'general_scvi',
title = 'general_scvi_all')