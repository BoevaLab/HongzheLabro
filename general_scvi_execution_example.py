python general_scvi.py

idata = sc.read_h5ad("Immune_ALL_human.h5ad")

mdata = malignant_cell_collection(idata, malignant_cell_incices = [1,5], label_key = 'final_annotation')

model_preprocessing(mdata)

model = model_train(mdata, layer_key = 'counts', batch_key = 'batch', n_layers = 5, n_hidden = 512, n_latent = 10)

plot_reconstruction_loss_and_elbo(model)

get_latent_UMAP(model, mdata, 'batch', 'final_annotation', "X_VAE")

kbet_collection = kbet_for_different_cell_type(mdata, latent_key = 'X_VAE', batch_key = 'batch', label_key = 'final_annotation')
print(kbet_collection)

kbet_rni_asw(adata = mdata, latent_key = 'X_VAE', batch_key = 'batch', label_key = 'final_annotation', group_key = 'final_annotation', max_clusters = 8)
