import torch
import scvi
import anndata as ad
import scanpy as sc
import pandas as pd
import scipy

def malignant_cell_collection(adata, malignant_cell_incices, label_key):
    mdata = adata[adata.obs[label_key].isin(adata.obs[label_key].unique()[[i for i in malignant_cell_incices]])]
    return mdata

def model_preprocessing(adata):
    #adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=10e4)
    sc.pp.log1p(adata)
    adata.raw = adata
    sc.pp.highly_variable_genes(
    adata,
    n_top_genes=2000,
    subset=True,
    layer="counts",
    batch_key='batch'
)

def model_train(adata, layer_key, batch_key, n_layers, n_hidden, n_latent):
    adata = adata.copy()
    scvi.model.SCVI.setup_anndata(adata, layer=layer_key, batch_key = batch_key)
    model = scvi.model.SCVI(adata, n_layers = n_layers, n_hidden=n_hidden, n_latent=n_latent)
    model.train(max_epochs=100, validation_size=0.1, check_val_every_n_epoch=5, early_stopping=True, 
    early_stopping_monitor='elbo_validation', early_stopping_patience = 5)
    return model

def plot_reconstruction_loss_and_elbo(model):
    train_recon_loss = model.history['reconstruction_loss_train']
    elbo_train = model.history['elbo_train']
    elbo_val = model.history['elbo_validation']
    val_recon_loss = model.history['reconstruction_loss_validation']
    ax = train_recon_loss.plot()
    elbo_train.plot(ax = ax)
    elbo_val.plot(ax = ax)
    val_recon_loss.plot(ax = ax)

def get_latent_UMAP(model, adata, batch_key, label_key):
    latent = model.get_latent_representation()
    adata.obsm["X_VAE"] = latent
    sc.pp.neighbors(adata, use_rep="X_VAE", n_neighbors=20)
    sc.tl.umap(adata, min_dist=0.3)
    sc.pl.umap(adata, color = [label_key, batch_key])

def kbet_for_different_cell_type(adata, latent_key, batch_key, label_key):
    k = len(adata.obs[label_key].unique())
    kbet_collection = []
    for i in range(k):
        kbet_collection.append(adata.obs[label_key].unique()[i])
        kbet_score = kbet(adata[adata.obs[label_key].isin(adata.obs[label_key].unique()[[i]])], latent_key = latent_key, batch_key = batch_key, label_key = label_key)
        kbet_collection.append(kbet_score)
    return kbet_collection

def kbet_rni_asw(adata, latent_key, batch_key, label_key, group_key, max_clusters):
    bdata = adata.copy()
    ari_score_collection = []
    k = np.linspace(2, max_clusters, max_clusters-1)
    for i in k:
        cdata = _binary_search_leiden_resolution(bdata, k = int(i), start = 0.1, end = 0.9, key_added ='final_annotation', random_state = 0, directed = False, 
        use_weights = False, _epsilon = 1e-3)
        if cdata is None:
            ari_score_collection.append(0)
            continue
        adata.obs['cluster_{}'.format(int(i))] = cdata.obs['final_annotation']
        ari_score_collection.append(compute_ari(adata, group_key = group_key, cluster_key = 'cluster_{}'.format(int(i))))


    # Note, all keys should come from the columns in adata.obs
    ari_score = {f"maximum ARI_score with {int(k[np.argmax(ari_score_collection)])} clusters": np.max(ari_score_collection)}
    sc.pl.umap(adata, color = ['cluster_{}'.format(int(k[np.argmax(ari_score_collection)]))])
    kbet_score = kbet(adata, latent_key=latent_key, batch_key=batch_key, label_key=label_key)
    asw_score = compute_asw(adata, group_key = group_key, latent_key = latent_key)
    return (kbet_score, ari_score, asw_score)