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
    
    if "counts" not in adata.layers.keys():
        adata.layers["counts"] = adata.X.copy()

    if scipy.sparse.issparse(adata.layers['counts']):
        if np.any([(k%1) for k in adata.layers['counts'].todense().ravel()]):
            adata.layers['counts'] = np.round(adata.layers['counts'].todense())
    else:
        if np.any([(k%1) for k in adata.layers['counts'].ravel()]):
            adata.layers['counts'] = np.round(adata.layers['counts'])
            
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
    kBET_scores = {"cell_type": [], "kBET": []}
    for i in range(k):
        kBET_scores["cell_type"].append(adata.obs[label_key].unique()[i])
        kbet_score = kbet(adata[adata.obs[label_key].isin(adata.obs[label_key].unique()[[i]])], latent_key = latent_key, batch_key = batch_key, label_key = label_key)
        kBET_scores["kBET"].append(kbet_score)
    return kBET_scores

from sklearn.metrics.cluster import silhouette_samples, silhouette_score

def silhouette_batch(
    adata,
    batch_key,
    group_key,
    latent_key,
    metric="euclidean",
    return_all=False,
    scale=True,
    verbose=True,
):
    """Batch ASW
    Modified average silhouette width (ASW) of batch
    This metric measures the silhouette of a given batch.
    It assumes that a silhouette width close to 0 represents perfect overlap of the batches, thus the absolute value of
    the silhouette width is used to measure how well batches are mixed.
    For all cells :math:`i` of a cell type :math:`C_j`, the batch ASW of that cell type is:
    .. math::
        batch \\, ASW_j = \\frac{1}{|C_j|} \\sum_{i \\in C_j} |silhouette(i)|
    The final score is the average of the absolute silhouette widths computed per cell type :math:`M`.
    .. math::
        batch \\, ASW = \\frac{1}{|M|} \\sum_{i \\in M} batch \\, ASW_j
    For a scaled metric (which is the default), the absolute ASW per group is subtracted from 1 before averaging, so that
    0 indicates suboptimal label representation and 1 indicates optimal label representation.
    .. math::
        batch \\, ASW_j = \\frac{1}{|C_j|} \\sum_{i \\in C_j} 1 - |silhouette(i)|
    :param batch_key: batch labels to be compared against
    :param group_key: group labels to be subset by e.g. cell type
    :param embed: name of column in adata.obsm
    :param metric: see sklearn silhouette score
    :param scale: if True, scale between 0 and 1
    :param return_all: if True, return all silhouette scores and label means
        default False: return average width silhouette (ASW)
    :param verbose: print silhouette score per group
    :return:
        Batch ASW  (always)
        Mean silhouette per group in pd.DataFrame (additionally, if return_all=True)
        Absolute silhouette scores per group label (additionally, if return_all=True)
    """
    if latent_key not in adata.obsm.keys():
        print(adata.obsm.keys())
        raise KeyError(f"{latent_key} not in obsm")

    sil_per_label = []
    for group in adata.obs[group_key].unique():
        adata_group = adata[adata.obs[group_key] == group]
        n_batches = adata_group.obs[batch_key].nunique()

        if (n_batches == 1) or (n_batches == adata_group.shape[0]):
            continue

        sil = silhouette_samples(
            adata_group.obsm[latent_key], adata_group.obs[batch_key], metric=metric
        )

        # take only absolute value
        sil = [abs(i) for i in sil]

        if scale:
            # scale s.t. highest number is optimal
            sil = [1 - i for i in sil]

        sil_per_label.extend([(group, score) for score in sil])

    sil_df = pd.DataFrame.from_records(
        sil_per_label, columns=["group", "silhouette_score"]
    )

    if len(sil_per_label) == 0:
        sil_means = np.nan
        asw = np.nan
    else:
        sil_means = sil_df.groupby("group").mean()
        asw = sil_means["silhouette_score"].mean()

    if verbose:
        print(f"mean silhouette per group: {sil_means}")

    if return_all:
        return asw, sil_means, sil_df

    return {"asw_batch_score":asw}


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
    asw_batch_score = silhouette_batch(adata, batch_key = batch_key, group_key= group_key, latent_key= latent_key)

    return [kbet_score, ari_score, asw_score, asw_batch_score]

def max_min_scale(dataset):
    if np.max(dataset) - np.min(dataset) != 0:
        return (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset))
    if np.max(dataset) - np.min(dataset) == 0:
        return dataset

def hyperparameter_tuning(adata, layer_key, batch_key, label_key, group_key, max_clusters, grid_search_list):
    # After malignant_cell_collection and model_preprocessing
    ari_collection = []
    asw_batch_collection = []
    kbet_collection = [] 
    asw_collection = []

    for i in grid_search_list:
        model = model_train(adata, layer_key = layer_key, batch_key = batch_key, n_layers = i, n_hidden = 512, n_latent = 10)
        latent_key = get_latent_UMAP(model, adata, batch_key, label_key, added_latent_key = 'X_VAE_{}'.format(int(i)), print_UMAP = True)
        score_collection = kbet_rni_asw(adata, latent_key = latent_key, batch_key = batch_key, label_key = label_key, group_key = group_key, max_clusters = max_clusters)
        ari_collection.append(list(score_collection[1].values())[0])
        asw_batch_collection.append(list(score_collection[3].values())[0])
        kbet_collection.append(list(score_collection[0].values())[0])
        asw_collection.append(list(score_collection[2].values())[0])

    ari_collection_mn = max_min_scale(ari_collection)
    asw_batch_collection_mn = max_min_scale(asw_batch_collection)
    kbet_collection_mn = max_min_scale(kbet_collection)
    asw_collection_mn = max_min_scale(asw_collection)


    #batch removal score includes kbet and asw_batch, bio-conservation score (cell_type_keeping) includes ari and asw_cell
    bio_score_collection = [] 
    batch_score_collection = [] 
    overall_score_collection = []
    for i in range(len(grid_search_list)):
        bio_score = np.mean((ari_collection_mn[i], asw_collection_mn[i]))
        bio_score_collection.append(bio_score)
        batch_score = np.mean((kbet_collection_mn[i], asw_batch_collection_mn[i]))
        batch_score_collection.append(batch_score)
        overall_score_collection.append(0.6 * bio_score + 0.4 * batch_score)
    return [ari_collection, asw_collection, kbet_collection, asw_batch_collection,bio_score_collection, batch_score_collection, overall_score_collection]

def convert_scorelist_into_df(scorelist, variable_name, store, csv_file_name):
    score_pd = pd.DataFrame(scorelist, index = ["ari", "asw_cell", "kbet", "asw_batch","bio_score", "batch_score", "overall_score"], columns = variable_name)
    if store:
        pd.to_csv(csv_file_name)
    return score_pd

def remove_minor_batch_for_malignant_cells(mdata, label_key, batch_key, num_minor_cell):
    # Extracts the index of the minor cell_type
    minor_cell_number_collection = []
    for ind, _ in enumerate(range(len(mdata.obs[label_key].unique()))):
        minor_cell_number_collection.append(len(mdata[mdata.obs[label_key].isin(mdata.obs[label_key].unique()[[ind]])]))
    minor_cell_indices_sort = list(np.argsort(minor_cell_number_collection))
    minor_cell_index = sorted(np.argsort(minor_cell_indices_sort)[:num_minor_cell])
    major_cell_index = sorted(np.argsort(minor_cell_indices_sort)[len(minor_cell_indices_sort)-num_minor_cell:])

    # Extracts the index of the majority batch in the minor cell_type
    removedata = mdata.copy()
    minor_cells = removedata[removedata.obs[label_key].isin(removedata.obs[label_key].unique()[[i for i in minor_cell_index]])]
    major_cells = removedata[removedata.obs[label_key].isin(removedata.obs[label_key].unique()[[i for i in major_cell_index ]])]
    minor_cell_number_collection = []
    minor_cell_index_collection = []
    for ind, _ in enumerate(range(len(minor_cells.obs[batch_key].unique()))):
        minor_cell_number_collection.append(len(minor_cells[minor_cells.obs[batch_key]==minor_cells.obs[batch_key].unique()[ind]]))
        minor_cell_index_collection.append(ind)
    minor_cell_index_collection.remove(np.argmax(minor_cell_number_collection))
    minor_cells = minor_cells[minor_cells.obs[batch_key].isin(minor_cells.obs[batch_key].unique()[[i for i in minor_cell_index_collection]])]
    removedata = ad.concat((minor_cells, major_cells))
    return removedata