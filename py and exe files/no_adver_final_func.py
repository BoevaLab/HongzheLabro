def data_spliting_del_round(adata, label_key, non_malignant_cell_indices, malignant_cell_indices, delete):
    ######################################################## choosing to eliminate the cell_types
    
    # Pretending some of the cell_types are maglingant
    if "counts" not in adata.layers.keys():
        adata.layers["counts"] = adata.X.copy()

    # check whether the cell_type has float expression data, if so, delete it
    if delete:
        for ind,val in enumerate(non_malignant_cell_indices):
            checkdata = adata[adata.obs[label_key].isin(adata.obs[label_key].unique()[[val]])]

            if scipy.sparse.issparse(checkdata.layers['counts']):
                if np.any([(k%1) for k in checkdata.layers['counts'].todense().ravel()]):
                    non_malignant_cell_indices[ind] = -1
            else:
                if np.any([(k%1) for k in checkdata.layers['counts'].ravel()]):
                    non_malignant_cell_indices[ind] = -1
    
        non_malignant_cell_indices = [i for i in non_malignant_cell_indices if i != -1]

        for ind,val in enumerate(malignant_cell_indices):
            checkdata = adata[adata.obs[label_key].isin(adata.obs[label_key].unique()[[val]])]

            if scipy.sparse.issparse(checkdata.layers['counts']):
                if np.any([(k%1) for k in checkdata.layers['counts'].todense().ravel()]):
                    malignant_cell_indices[ind] = -1
            else:
                if np.any([(k%1) for k in checkdata.layers['counts'].ravel()]):
                    malignant_cell_indices[ind] = -1

        malignant_cell_indices = [i for i in malignant_cell_indices if i != -1]

    # check whether the cell_type has float expression data, if so, round it        
    else:
        if scipy.sparse.issparse(adata.layers['counts']):
            if np.any([(k%1) for k in adata.layers['counts'].todense().ravel()]):
                adata.layers['counts'] = np.round(adata.layers['counts'].todense())
        else:
            if np.any([(k%1) for k in adata.layers['counts'].ravel()]):
                adata.layers['counts'] = np.round(adata.layers['counts'])



    ndata = adata[adata.obs[label_key].isin(adata.obs[label_key].unique()[[i for i in non_malignant_cell_indices]])]
    mdata = adata[adata.obs[label_key].isin(adata.obs[label_key].unique()[[i for i in malignant_cell_indices]])]
    return ndata, mdata

def train_model_firstVAE(ndata, layer_key, cell_type_key):
    ndata = ndata.copy()
    SCVI_humanimm.setup_anndata(ndata, layer = layer_key, batch_key = cell_type_key) 
    model_imm = SCVI_humanimm(ndata)
    model_imm.train(max_epochs=100)
    return model_imm

def get_latent_UMAP(model, ndata, batch_key, label_key, added_latent_key, print_UMAP):
    latent = model.get_latent_representation()
    ndata.obsm[added_latent_key] = latent
    sc.pp.neighbors(ndata, use_rep=added_latent_key, n_neighbors=20)
    sc.tl.umap(ndata, min_dist=0.3)
    if print_UMAP:
        sc.pl.umap(ndata, color = [label_key, batch_key])
    return latent

def fetch_batch_information(ndata, mdata, batch_key, latent_info):
    batch_df_imm = pd.DataFrame(latent_info, index = ndata.obs[batch_key])
    batch_df_mean_imm = batch_df_imm.groupby(batch_df_imm.index).mean()
    batch_df_mean_loc_imm = batch_df_mean_imm.loc[mdata.obs[batch_key]]
    latent_id_imm = [f'latent{i}' for i in range(batch_df_mean_loc_imm.shape[1])]
    mdata.obs[latent_id_imm] = batch_df_mean_loc_imm.values
    return mdata, latent_id_imm

def first_VAE(adata, batch_key, label_key, layer_key, non_malignant_cell_indices, malignant_cell_indices, delete, added_latent_key, print_UMAP):
    ndata, mdata = data_spliting_del_round(adata, label_key = label_key, non_malignant_cell_indices = non_malignant_cell_indices, malignant_cell_indices = malignant_cell_indices, delete = delete)
    model_imm = train_model_firstVAE(ndata, layer_key = layer_key, cell_type_key = label_key)
    latent_imm = get_latent_UMAP(model_imm, ndata, batch_key = batch_key, label_key = label_key, added_latent_key = added_latent_key, print_UMAP = print_UMAP)
    mdata, latent_id_imm = fetch_batch_information(ndata, mdata, batch_key = batch_key, latent_info = latent_imm)
    return mdata, latent_id_imm

def second_VAE(mdata, latent_info, n_layers, n_hidden, n_latent, lr):
    scvi.model.SCVI.setup_anndata(mdata, layer='counts', continuous_covariate_keys=latent_info)
    sec_model_imm = scvi.model.SCVI(mdata, n_layers = n_layers, n_hidden = n_hidden, n_latent = n_latent)
    sec_model_imm.train(max_epochs = 100, validation_size = 0.1, check_val_every_n_epoch = 5, early_stopping=True, 
    early_stopping_monitor='elbo_validation', early_stopping_patience = 20, plan_kwargs={'lr':lr})
    return sec_model_imm

def plot_reconstruction_loss_and_elbo(model):
    train_recon_loss = model.history['reconstruction_loss_train']
    elbo_train = model.history['elbo_train']
    elbo_val = model.history['elbo_validation']
    val_recon_loss = model.history['reconstruction_loss_validation']
    ax = train_recon_loss.plot()
    elbo_train.plot(ax = ax)
    elbo_val.plot(ax = ax)
    val_recon_loss.plot(ax = ax)

def get_latent_secUMAP(model, ndata, batch_key, label_key, added_latent_key, print_UMAP):
    latent = model.get_latent_representation()
    ndata.obsm[added_latent_key] = latent
    sc.pp.neighbors(ndata, use_rep=added_latent_key, n_neighbors=20)
    sc.tl.umap(ndata, min_dist=0.3)
    if print_UMAP:
        sc.pl.umap(ndata, color = [label_key, batch_key])
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

def hyperparameter_tuning(adata, latent_info, batch_key, label_key, group_key, max_clusters, grid_search_list):
    # After data_spliting_del, train_model_firstVAE and fetch_batch_information


    ari_collection = []
    asw_batch_collection = []
    kbet_collection = [] 
    asw_collection = []
    for i in grid_search_list:
        model = second_VAE(adata, latent_info = latent_info, n_layers = i[0], n_hidden = 512, n_latent = i[1], lr = i[2])
        print("VAE layers of %s, num latents of %s, lr of %s" % (i[0],i[1],i[2]))
        latent_key = get_latent_secUMAP(model, adata, batch_key, label_key, added_latent_key = 'X_secVAE_{}'.format(i), print_UMAP = True)
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

    bio_score_collection = [] 
    batch_score_collection = [] 
    overall_score_collection = []
    for i in range(len(grid_search_list)):
        bio_score = np.mean((ari_collection_mn[i], asw_collection_mn[i]))
        bio_score_collection.append(bio_score)
        batch_score = np.mean((kbet_collection_mn[i], asw_batch_collection_mn[i]))
        batch_score_collection.append(batch_score)
        overall_score_collection.append(0.6 * bio_score + 0.4 * batch_score)
    return [ari_collection, asw_collection, kbet_collection, asw_batch_collection, bio_score_collection, batch_score_collection, overall_score_collection]

def convert_scorelist_into_df(scorelist, variable_name, store, csv_file_name):
    score_pd = pd.DataFrame(scorelist, index = ["ari", "asw_cell", "kbet", "asw_batch","bio_score", "batch_score", "overall_score"], columns = variable_name)
    if store:
        score_pd.to_csv(csv_file_name)
    return score_pd