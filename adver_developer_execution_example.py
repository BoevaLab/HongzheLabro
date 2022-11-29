python adver_developer.py

# Need to download the corresponding h5ad beforehand
idata = sc.read_h5ad("Immune_ALL_human.h5ad")

ndata, mdata = data_spliting_del(idata, label_key = 'final_annotation', non_malignant_cell_indices = [0,2,3,4,6,7,8,9,10,11,12,13,14,15], malignant_cell_indices = [1,5])

model_imm = train_model_firstVAE(ndata, 'counts', 'final_annotation')latent_imm = get_latent_UMAP(model_imm, ndata, 'batch', 'final_annotation', "X_adverVAE_1")

mdata, latent_id_imm = fetch_batch_information(ndata, mdata, 'batch', latent_imm)

sec_model_imm = second_VAE(mdata, latent_id_imm)

plot_reconstruction_loss_and_elbo(sec_model_imm)

get_latent_secUMAP(sec_model_imm, mdata, 'batch', 'final_annotation', 'X_secVAE_1')

kbet_collection = kbet_for_different_cell_type(mdata, latent_key = 'X_secVAE_1', batch_key = 'batch', label_key = 'final_annotation')
print(kbet_collection)

kbet_rni_asw(adata = mdata, latent_key = 'X_secVAE_1', batch_key = 'batch', label_key = 'final_annotation', group_key = 'final_annotation', max_clusters = 8)

