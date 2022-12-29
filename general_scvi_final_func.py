def malignant_cell_collection(adata, malignant_cell_incices, label_key):
    mdata = adata[adata.obs[label_key].isin(adata.obs[label_key].unique()[[i for i in malignant_cell_incices]])]
    return mdata

def model_preprocessing(adata, batch_key):
    
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
    batch_key=batch_key
)

def model_train(adata, layer_key, batch_key, n_layers, n_hidden, n_latent, lr):
    adata = adata.copy()
    scvi.model.SCVI.setup_anndata(adata, layer=layer_key, batch_key = batch_key)
    model = scvi.model.SCVI(adata, n_layers = n_layers, n_hidden=n_hidden, n_latent=n_latent)
    model.train(max_epochs=100, validation_size=0.1, check_val_every_n_epoch=5, early_stopping=True, 
    early_stopping_monitor='elbo_validation', early_stopping_patience = 20, plan_kwargs={'lr':lr})
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

def get_latent_UMAP(model, adata, batch_key, label_key, added_latent_key, print_UMAP):
    latent = model.get_latent_representation()
    adata.obsm[added_latent_key] = latent
    sc.pp.neighbors(adata, use_rep=added_latent_key, n_neighbors=20)
    sc.tl.umap(adata, min_dist=0.3)
    if print_UMAP:
        sc.pl.umap(adata, color = [label_key, batch_key])
    return added_latent_key

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

def grid_search_list_generator(layers, num_latent, lr):
    grid_search_list = []
    for i in range(len(layers)):
        for j in range(len(num_latent)):
            for k in range(len(lr)):
                grid_search_list.append([layers[i], num_latent[j], lr[k]])

    return grid_search_list

def hyperparameter_tuning_general_scvi(adata, layer_key, batch_key, label_key, group_key, max_clusters, grid_search_list):
    # After malignant_cell_collection and model_preprocessing


    ari_collection = []
    asw_batch_collection = []
    kbet_collection = [] 
    asw_collection = []

    for i in grid_search_list:
        model = model_train(adata, layer_key = layer_key, batch_key = batch_key, n_layers = i[0], n_hidden = 512, n_latent = i[1], lr = i[2])
        print("VAE layers of %s, num latents of %s, lr of %s" % (i[0],i[1],i[2]))
        latent_key = get_latent_UMAP(model, adata, batch_key, label_key, added_latent_key = 'X_VAE_{}'.format(i), print_UMAP = True)
        score_collection = kbet_rni_asw(adata, latent_key = latent_key, batch_key = batch_key, label_key = label_key, group_key = group_key, max_clusters = max_clusters)
        for i in score_collection[1].values():
            ari_collection.append(i)
        for i in score_collection[3].values():
            asw_batch_collection.append(i)
        for i in score_collection[0].values():
            kbet_collection.append(i)
        for i in score_collection[2].values():
            asw_collection.append(i)

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
        score_pd.to_csv(csv_file_name)
    return score_pd