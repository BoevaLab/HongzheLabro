import torch
import scvi
import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
import scipy

grid_search_list = grid_search_list_generator(layers = [1,3,5], 
num_latent = [5,10,15], 
lr =[1e-3,1e-4,1e-5])

col_name = []
for i in grid_search_list:
    col_name.append("VAE layers of %s, num latents of %s, lr of %s" % (i[0],i[1],i[2]))

idata = sc.read_h5ad("Immune_ALL_human.h5ad")

mdata = malignant_cell_collection(idata, malignant_cell_incices = [1,5], label_key = 'final_annotation')

model_preprocessing(mdata, 'batch')

score = hyperparameter_tuning_general_scvi(mdata, layer_key = 'counts', batch_key= 'batch', label_key= 'final_annotation', group_key= 'final_annotation', max_clusters = 8, grid_search_list = grid_search_list)

score_pd = convert_scorelist_into_df(score, col_name, True,'grid_general_scvi_all.csv')


ldata = sc.read_h5ad("Lung_atlas_public.h5ad")
mldata = malignant_cell_collection(ldata, malignant_cell_incices = [0,2], label_key = 'cell_type')
model_preprocessing(mldata, 'batch')
score_lung = hyperparameter_tuning_general_scvi(mldata, layer_key = 'counts', batch_key= 'batch', label_key= 'cell_type', group_key= 'cell_type', max_clusters = 8, grid_search_list = grid_search_list)
score_lung_pd = convert_scorelist_into_df(score_lung, col_name, True,'grid_general_scvi_lung.csv')

pdata = sc.read_h5ad("human_pancreas_norm_complexBatch.h5ad")

mpdata = malignant_cell_collection(pdata, malignant_cell_incices = [0,2], label_key = 'celltype')
model_preprocessing(mpdata, 'tech')
score_pancreas = hyperparameter_tuning_general_scvi(mpdata, layer_key = 'counts', batch_key= 'tech', label_key= 'celltype', group_key= 'celltype', max_clusters = 8, grid_search_list = grid_search_list)
score_pancreas_pd = convert_scorelist_into_df(score_pancreas, col_name, True,'grid_general_scvi_pancreas.csv')