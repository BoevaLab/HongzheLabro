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
    SCVIadver_humanimm.setup_anndata(ndata, layer = layer_key, batch_key = cell_type_key) 
    model_imm = SCVIadver_humanimm(ndata)
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
    mdata = mdata[mdata.obs[batch_key].isin(ndata.obs[batch_key].unique())]
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

#removing the major batch in the minor maglingant cells and see how it may perform on the batch effect
def remove_major_batch_for_malignant_cells(mdata, label_key, batch_key, num_minor_cell, num_batch_remove):
    # Extracts the index of the minor cell_type
    minor_cell_number_collection = []
    for ind, _ in enumerate(range(len(mdata.obs[label_key].unique()))):
        minor_cell_number_collection.append(len(mdata[mdata.obs[label_key].isin(mdata.obs[label_key].unique()[[ind]])]))

    minor_cell_indices_sort = np.argsort(minor_cell_number_collection)
    minor_cell_index = minor_cell_indices_sort[:num_minor_cell]
    major_cell_index = np.delete(minor_cell_indices_sort, np.argwhere(minor_cell_indices_sort == minor_cell_index))


    # Extracts the index of the majority batch in the minor cell_type
    removedata = mdata.copy()
    minor_cells = removedata[removedata.obs[label_key].isin(removedata.obs[label_key].unique()[[i for i in minor_cell_index]])]
    major_cells = removedata[removedata.obs[label_key].isin(removedata.obs[label_key].unique()[[i for i in major_cell_index]])]

    if len(minor_cells.obs[batch_key].unique()) <= num_batch_remove:
        raise ValueError("number of batches to remove is equivalent to or greater than the total batches that minor_cells has, please use a lower number")

    minor_cell_number_collection = []

    for ind, _ in enumerate(range(len(minor_cells.obs[batch_key].unique()))):
        minor_cell_number_collection.append(len(minor_cells[minor_cells.obs[batch_key]==minor_cells.obs[batch_key].unique()[ind]]))

    if num_batch_remove != 0:
        minor_cell_index_collection = np.argsort(minor_cell_number_collection)[:-num_batch_remove]
    else:
        minor_cell_index_collection = np.argsort(minor_cell_number_collection)

    minor_cells = minor_cells[minor_cells.obs[batch_key].isin(minor_cells.obs[batch_key].unique()[[i for i in minor_cell_index_collection]])]
    removedata = ad.concat((minor_cells, major_cells))
    return removedata

def hyperparameter_selection(dataset, method):
    import re
    
    if dataset != "all" and dataset != "lung" and dataset != "pancreas":
        raise ValueError('dataset can only be "all", "lung" or "pancreas"')
    if method != "general_scvi" and method != "adver" and method != "No_adver":
        raise ValueError('method can only be "general_scvi", "adver" or "No_adver"')
    
    if method == "adver":
        if dataset == "all":
            pd_csv = pd.read_csv("grid_adver_train_all.csv")
            pd_score_T = pd_csv.loc[pd_csv["Unnamed: 0"] == "overall_score"].T
            pd_score_T.columns = ['overall_score']
            pd_best = pd_score_T[1:].sort_values(by=['overall_score'], ascending=False)[:3]
            parameter_collection = []
            for row in pd_best.iterrows():
                parameter_list = [float(s) for s in re.findall(r'-?\d+\.?\d*', row[0])]
                for i in range(2):
                    parameter_list[i] = int(parameter_list[i])
                parameter_collection.append(parameter_list)
            for i in parameter_collection:
                if i[2] == 1.0:
                    i[2] = 1e-5
                    
            return parameter_collection

        if dataset == "lung":
            pd_csv = pd.read_csv("grid_adver_train_lung.csv")
            pd_score_T = pd_csv.loc[pd_csv["Unnamed: 0"] == "overall_score"].T
            pd_score_T.columns = ['overall_score']
            pd_best = pd_score_T[1:].sort_values(by=['overall_score'], ascending=False)[:3]
            parameter_collection = []
            for row in pd_best.iterrows():
                parameter_list = [float(s) for s in re.findall(r'-?\d+\.?\d*', row[0])]
                for i in range(2):
                    parameter_list[i] = int(parameter_list[i])
                parameter_collection.append(parameter_list)
            for i in parameter_collection:
                if i[2] == 1.0:
                    i[2] = 1e-5
                    
            return parameter_collection

        if dataset == "pancreas":
            pd_csv = pd.read_csv("grid_adver_train_pancreas.csv")
            pd_score_T = pd_csv.loc[pd_csv["Unnamed: 0"] == "overall_score"].T
            pd_score_T.columns = ['overall_score']
            pd_best = pd_score_T[1:].sort_values(by=['overall_score'], ascending=False)[:3]
            parameter_collection = []
            for row in pd_best.iterrows():
                parameter_list = [float(s) for s in re.findall(r'-?\d+\.?\d*', row[0])]
                for i in range(2):
                    parameter_list[i] = int(parameter_list[i])
                parameter_collection.append(parameter_list)
            for i in parameter_collection:
                if i[2] == 1.0:
                    i[2] = 1e-5                    
            return parameter_collection
    
    if method == "general_scvi":
        if dataset == "all":
            pd_csv = pd.read_csv("grid_general_scvi_all.csv")
            pd_score_T = pd_csv.loc[pd_csv["Unnamed: 0"] == "overall_score"].T
            pd_score_T.columns = ['overall_score']
            pd_best = pd_score_T[1:].sort_values(by=['overall_score'], ascending=False)[:3]
            parameter_collection = []
            for row in pd_best.iterrows():
                parameter_list = [float(s) for s in re.findall(r'-?\d+\.?\d*', row[0])]
                for i in range(2):
                    parameter_list[i] = int(parameter_list[i])
                parameter_collection.append(parameter_list)
            for i in parameter_collection:
                if i[2] == 1.0:
                    i[2] = 1e-5                    
            return parameter_collection

        if dataset == "lung":
            pd_csv = pd.read_csv("grid_general_scvi_lung.csv")
            pd_score_T = pd_csv.loc[pd_csv["Unnamed: 0"] == "overall_score"].T
            pd_score_T.columns = ['overall_score']
            pd_best = pd_score_T[1:].sort_values(by=['overall_score'], ascending=False)[:3]
            parameter_collection = []
            for row in pd_best.iterrows():
                parameter_list = [float(s) for s in re.findall(r'-?\d+\.?\d*', row[0])]
                for i in range(2):
                    parameter_list[i] = int(parameter_list[i])
                parameter_collection.append(parameter_list)
            for i in parameter_collection:
                if i[2] == 1.0:
                    i[2] = 1e-5                    
            return parameter_collection

        if dataset == "pancreas":
            pd_csv = pd.read_csv("grid_general_scvi_pancreas.csv")
            pd_score_T = pd_csv.loc[pd_csv["Unnamed: 0"] == "overall_score"].T
            pd_score_T.columns = ['overall_score']
            pd_best = pd_score_T[1:].sort_values(by=['overall_score'], ascending=False)[:3]
            parameter_collection = []
            for row in pd_best.iterrows():
                parameter_list = [float(s) for s in re.findall(r'-?\d+\.?\d*', row[0])]
                for i in range(2):
                    parameter_list[i] = int(parameter_list[i])
                parameter_collection.append(parameter_list)
            for i in parameter_collection:
                if i[2] == 1.0:
                    i[2] = 1e-5                    
            return parameter_collection


    if method == "No_adver":
        if dataset == "all":
            pd_csv = pd.read_csv("grid_NO_adver_train_all.csv")
            pd_score_T = pd_csv.loc[pd_csv["Unnamed: 0"] == "overall_score"].T
            pd_score_T.columns = ['overall_score']
            pd_best = pd_score_T[1:].sort_values(by=['overall_score'], ascending=False)[:3]
            parameter_collection = []
            for row in pd_best.iterrows():
                parameter_list = [float(s) for s in re.findall(r'-?\d+\.?\d*', row[0])]
                for i in range(2):
                    parameter_list[i] = int(parameter_list[i])
                parameter_collection.append(parameter_list)
            for i in parameter_collection:
                if i[2] == 1.0:
                    i[2] = 1e-5                    
            return parameter_collection

        if dataset == "lung":
            pd_csv = pd.read_csv("grid_NO_adver_train_lung.csv")
            pd_score_T = pd_csv.loc[pd_csv["Unnamed: 0"] == "overall_score"].T
            pd_score_T.columns = ['overall_score']
            pd_best = pd_score_T[1:].sort_values(by=['overall_score'], ascending=False)[:3]
            parameter_collection = []
            for row in pd_best.iterrows():
                parameter_list = [float(s) for s in re.findall(r'-?\d+\.?\d*', row[0])]
                for i in range(2):
                    parameter_list[i] = int(parameter_list[i])
                parameter_collection.append(parameter_list)
            for i in parameter_collection:
                if i[2] == 1.0:
                    i[2] = 1e-5                    
            return parameter_collection

        if dataset == "pancreas":
            pd_csv = pd.read_csv("grid_No_adver_train_pancreas.csv")
            pd_score_T = pd_csv.loc[pd_csv["Unnamed: 0"] == "overall_score"].T
            pd_score_T.columns = ['overall_score']
            pd_best = pd_score_T[1:].sort_values(by=['overall_score'], ascending=False)[:3]
            parameter_collection = []
            for row in pd_best.iterrows():
                parameter_list = [float(s) for s in re.findall(r'-?\d+\.?\d*', row[0])]
                for i in range(2):
                    parameter_list[i] = int(parameter_list[i])
                parameter_collection.append(parameter_list)
            for i in parameter_collection:
                if i[2] == 1.0:
                    i[2] = 1e-5                    
            return parameter_collection

def hyperparameter_setting(adata, latent_info, batch_key, label_key, group_key, max_clusters, hyperparameter_collection):
    ari_collection = []
    asw_batch_collection = []
    kbet_collection = [] 
    asw_collection = []

    model = second_VAE(adata, latent_info = latent_info, n_layers = hyperparameter_collection[0], n_hidden = 512, n_latent = hyperparameter_collection[1], lr = hyperparameter_collection[2])
    latent_key = get_latent_secUMAP(model, adata, batch_key, label_key, added_latent_key = 'X_secVAE_1', print_UMAP = True)
    score_collection = kbet_rni_asw(adata, latent_key = latent_key, batch_key = batch_key, label_key = label_key, group_key = group_key, max_clusters = max_clusters)
    for i in score_collection[1].values():
        ari_collection.append(i)
    for i in score_collection[3].values():
        asw_batch_collection.append(i)
    for i in score_collection[0].values():
        kbet_collection.append(i)
    for i in score_collection[2].values():
        asw_collection.append(i)

    return [ari_collection, asw_collection, kbet_collection, asw_batch_collection]

def train_model_firstVAE_noadver(ndata, layer_key, cell_type_key):
    ndata = ndata.copy()
    SCVIno_humanimm.setup_anndata(ndata, layer = layer_key, batch_key = cell_type_key) 
    model_imm = SCVIno_humanimm(ndata)
    model_imm.train(max_epochs=100)
    return model_imm

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

def hyperparameter_setting_general_scvi(adata, layer_key, batch_key, label_key, group_key, max_clusters, hyperparameter_collection):
    ari_collection = []
    asw_batch_collection = []
    kbet_collection = [] 
    asw_collection = []

    model = model_train(adata, layer_key = layer_key, batch_key = batch_key, n_layers = hyperparameter_collection[0], n_hidden = 512, n_latent = hyperparameter_collection[1], lr = hyperparameter_collection[2])
    latent_key = get_latent_secUMAP(model, adata, batch_key, label_key, added_latent_key = 'X_VAE_1', print_UMAP = True)
    score_collection = kbet_rni_asw(adata, latent_key = latent_key, batch_key = batch_key, label_key = label_key, group_key = group_key, max_clusters = max_clusters)
    for i in score_collection[1].values():
        ari_collection.append(i)
    for i in score_collection[3].values():
        asw_batch_collection.append(i)
    for i in score_collection[0].values():
        kbet_collection.append(i)
    for i in score_collection[2].values():
        asw_collection.append(i)

    return [ari_collection, asw_collection, kbet_collection, asw_batch_collection]

def convert_diffhyperlist_into_df(scorelist, variable_name, store, csv_file_name):
    score_pd = pd.DataFrame(scorelist, index = ["ari", "asw_cell", "kbet", "asw_batch"], columns = variable_name)
    if store:
        score_pd.to_csv(csv_file_name)
    return score_pd

def combined_hetero_test(dataset, method,
    num_hyperparameter_selection = 3,      
    delete = False, 
    num_batch = 4, 
    pd_columns_name = ["0_batches_removed", "1_batches_removed","2_batches_removed","3_batches_removed"]):

    if dataset != "all" and dataset != "lung" and dataset != "pancreas":
        raise ValueError('dataset can only be "all", "lung" or "pancreas"')
    if method != "general_scvi" and method != "adver" and method != "No_adver":
        raise ValueError('method can only be "general_scvi", "adver" or "No_adver"')

    if method == "adver":
        if dataset == "all":
            layers_key = 'counts'
            batch_key = 'batch'
            label_key = 'final_annotation'
            group_key = 'final_annotation'
            non_malignant_cell_indices = [0,2,3,4,6,7,8,9,10,11,12,13,14,15]
            malignant_cell_indices = [1,5]
            dataset_csv = "Immune_ALL_human.h5ad"
            idata = sc.read_h5ad(dataset_csv)

            nnewdata, mnewdata = data_spliting_del_round(idata, label_key, non_malignant_cell_indices = non_malignant_cell_indices, malignant_cell_indices = malignant_cell_indices, delete = delete)
            model_imm = train_model_firstVAE(nnewdata, layers_key, label_key)
            latent_imm = get_latent_UMAP(model_imm, nnewdata, batch_key, group_key, "X_adverVAE_1", False)
            mnewdata, latent_id_imm = fetch_batch_information(nnewdata, mnewdata, batch_key, latent_imm)

            score_collection = []
            for i in range(num_batch):
                removedata = remove_major_batch_for_malignant_cells(mnewdata, label_key = label_key, batch_key = batch_key, num_minor_cell = 1, num_batch_remove = i)
                diff_hyper = []
                for k in range(num_hyperparameter_selection):
                    hyperparameter_collection = hyperparameter_selection(dataset = dataset, method = method)
                    score_remove = hyperparameter_setting(removedata, latent_info = latent_id_imm, batch_key= batch_key, label_key= label_key, group_key= group_key, max_clusters = 8,
                    hyperparameter_collection = hyperparameter_collection[k])
                    diff_hyper.append(score_remove)
                print(hyperparameter_collection)                    
                diff_hyper = np.squeeze(np.array(diff_hyper), axis=2).T
                diff_hyper_pd = convert_diffhyperlist_into_df(diff_hyper, ['trial_1','trial_2','trial_3'], True, str(method) + "_" + str(dataset) + "_batch_" + str(i) + '.csv')

                score_collection.append(np.average(diff_hyper, axis = 1))

            score_collection = np.array(score_collection).T

            ari_collection_mn = max_min_scale(score_collection[0])
            asw_batch_collection_mn = max_min_scale(score_collection[3])
            kbet_collection_mn = max_min_scale(score_collection[2])
            asw_collection_mn = max_min_scale(score_collection[1])

            bio_score_collection = [] 
            batch_score_collection = [] 
            overall_score_collection = []
            for i in range(score_collection.shape[1]):
                bio_score = np.mean((ari_collection_mn[i], asw_collection_mn[i]))
                bio_score_collection.append(bio_score)
                batch_score = np.mean((kbet_collection_mn[i], asw_batch_collection_mn[i]))
                batch_score_collection.append(batch_score)
                overall_score_collection.append(0.6 * bio_score + 0.4 * batch_score)

            bio_score_collection = np.array(bio_score_collection).reshape(1,len(bio_score_collection))
            batch_score_collection = np.array(batch_score_collection).reshape(1,len(batch_score_collection))
            overall_score_collection = np.array(overall_score_collection).reshape(1,len(overall_score_collection))
            
            score_collection = np.concatenate((score_collection, bio_score_collection, batch_score_collection, overall_score_collection), axis = 0)

            score_collection = convert_scorelist_into_df(score_collection, 
                pd_columns_name, 
                True, 
                "removed_batches_all_adver.csv")
            return score_collection

        if dataset == "lung":
            layers_key = 'counts'
            batch_key = 'batch'
            label_key = 'cell_type'
            group_key = 'cell_type'
            non_malignant_cell_indices = [1,2,3,4,7,8,9,10,11,12,13,14,15,16]
            malignant_cell_indices = [0,2] # type 1 and type 2
            dataset_csv = "Lung_atlas_public.h5ad"
            idata = sc.read_h5ad(dataset_csv)

            nnewdata, mnewdata = data_spliting_del_round(idata, label_key, non_malignant_cell_indices = non_malignant_cell_indices, malignant_cell_indices = malignant_cell_indices, delete = delete)
            model_imm = train_model_firstVAE(nnewdata, layers_key, label_key)
            latent_imm = get_latent_UMAP(model_imm, nnewdata, batch_key, group_key, "X_adverVAE_1", False)
            mnewdata, latent_id_imm = fetch_batch_information(nnewdata, mnewdata, batch_key, latent_imm)

            score_collection = []
            for i in range(num_batch):
                removedata = remove_major_batch_for_malignant_cells(mnewdata, label_key = label_key, batch_key = batch_key, num_minor_cell = 1, num_batch_remove = i)
                diff_hyper = []
                for k in range(num_hyperparameter_selection):
                    hyperparameter_collection = hyperparameter_selection(dataset = dataset, method = method)
                    score_remove = hyperparameter_setting(removedata, latent_info = latent_id_imm, batch_key= batch_key, label_key= label_key, group_key= group_key, max_clusters = 8,
                    hyperparameter_collection = hyperparameter_collection[k])
                    diff_hyper.append(score_remove)
                print(hyperparameter_collection)                    
                diff_hyper = np.squeeze(np.array(diff_hyper), axis=2).T
                diff_hyper_pd = convert_diffhyperlist_into_df(diff_hyper, ['trial_1','trial_2','trial_3'], True, str(method) + "_" + str(dataset) + "_batch_" + str(i) + '.csv')

                score_collection.append(np.average(diff_hyper, axis = 1))

            score_collection = np.array(score_collection).T

            ari_collection_mn = max_min_scale(score_collection[0])
            asw_batch_collection_mn = max_min_scale(score_collection[3])
            kbet_collection_mn = max_min_scale(score_collection[2])
            asw_collection_mn = max_min_scale(score_collection[1])

            bio_score_collection = [] 
            batch_score_collection = [] 
            overall_score_collection = []
            for i in range(score_collection.shape[1]):
                bio_score = np.mean((ari_collection_mn[i], asw_collection_mn[i]))
                bio_score_collection.append(bio_score)
                batch_score = np.mean((kbet_collection_mn[i], asw_batch_collection_mn[i]))
                batch_score_collection.append(batch_score)
                overall_score_collection.append(0.6 * bio_score + 0.4 * batch_score)

            bio_score_collection = np.array(bio_score_collection).reshape(1,len(bio_score_collection))
            batch_score_collection = np.array(batch_score_collection).reshape(1,len(batch_score_collection))
            overall_score_collection = np.array(overall_score_collection).reshape(1,len(overall_score_collection))
            
            score_collection = np.concatenate((score_collection, bio_score_collection, batch_score_collection, overall_score_collection), axis = 0)

            score_collection = convert_scorelist_into_df(score_collection, 
                pd_columns_name, 
                True, 
                "removed_batches_lung_adver.csv")
            return score_collection

        if dataset == "pancreas":
            layers_key = 'counts'
            batch_key = 'tech'
            label_key = 'celltype'
            group_key = 'celltype'
            non_malignant_cell_indices = [1,5,6,7,8,9,10,11,12,13]
            malignant_cell_indices = [0,2] # alpha and gamma
            dataset_csv = "human_pancreas_norm_complexBatch.h5ad"
            idata = sc.read_h5ad(dataset_csv)

            nnewdata, mnewdata = data_spliting_del_round(idata, label_key, non_malignant_cell_indices = non_malignant_cell_indices, malignant_cell_indices = malignant_cell_indices, delete = delete)
            model_imm = train_model_firstVAE(nnewdata, layers_key, label_key)
            latent_imm = get_latent_UMAP(model_imm, nnewdata, batch_key, group_key, "X_adverVAE_1", False)
            mnewdata, latent_id_imm = fetch_batch_information(nnewdata, mnewdata, batch_key, latent_imm)

            score_collection = []
            for i in range(num_batch):
                removedata = remove_major_batch_for_malignant_cells(mnewdata, label_key = label_key, batch_key = batch_key, num_minor_cell = 1, num_batch_remove = i)
                diff_hyper = []
                for k in range(num_hyperparameter_selection):
                    hyperparameter_collection = hyperparameter_selection(dataset = dataset, method = method)
                    score_remove = hyperparameter_setting(removedata, latent_info = latent_id_imm, batch_key= batch_key, label_key= label_key, group_key= group_key, max_clusters = 8,
                    hyperparameter_collection = hyperparameter_collection[k])
                    diff_hyper.append(score_remove)
                print(hyperparameter_collection)                    
                diff_hyper = np.squeeze(np.array(diff_hyper), axis=2).T
                diff_hyper_pd = convert_diffhyperlist_into_df(diff_hyper, ['trial_1','trial_2','trial_3'], True, str(method) + "_" + str(dataset) + "_batch_" + str(i) + '.csv')

                score_collection.append(np.average(diff_hyper, axis = 1))

            score_collection = np.array(score_collection).T

            ari_collection_mn = max_min_scale(score_collection[0])
            asw_batch_collection_mn = max_min_scale(score_collection[3])
            kbet_collection_mn = max_min_scale(score_collection[2])
            asw_collection_mn = max_min_scale(score_collection[1])

            bio_score_collection = [] 
            batch_score_collection = [] 
            overall_score_collection = []
            for i in range(score_collection.shape[1]):
                bio_score = np.mean((ari_collection_mn[i], asw_collection_mn[i]))
                bio_score_collection.append(bio_score)
                batch_score = np.mean((kbet_collection_mn[i], asw_batch_collection_mn[i]))
                batch_score_collection.append(batch_score)
                overall_score_collection.append(0.6 * bio_score + 0.4 * batch_score)

            bio_score_collection = np.array(bio_score_collection).reshape(1,len(bio_score_collection))
            batch_score_collection = np.array(batch_score_collection).reshape(1,len(batch_score_collection))
            overall_score_collection = np.array(overall_score_collection).reshape(1,len(overall_score_collection))
            
            score_collection = np.concatenate((score_collection, bio_score_collection, batch_score_collection, overall_score_collection), axis = 0)

            score_collection = convert_scorelist_into_df(score_collection, 
                pd_columns_name, 
                True, 
                "removed_batches_pancreas_adver.csv")
            return score_collection


    if method == "No_adver":
        %run No_adver.py
        if dataset == "all":
            layers_key = 'counts'
            batch_key = 'batch'
            label_key = 'final_annotation'
            group_key = 'final_annotation'
            non_malignant_cell_indices = [0,2,3,4,6,7,8,9,10,11,12,13,14,15]
            malignant_cell_indices = [1,5]
            dataset_csv = "Immune_ALL_human.h5ad"
            idata = sc.read_h5ad(dataset_csv)

            nnewdata, mnewdata = data_spliting_del_round(idata, label_key, non_malignant_cell_indices = non_malignant_cell_indices, malignant_cell_indices = malignant_cell_indices, delete = delete)
            model_imm = train_model_firstVAE_noadver(nnewdata, layers_key, label_key)
            latent_imm = get_latent_UMAP(model_imm, nnewdata, batch_key, group_key, "X_noadverVAE_1", False)
            mnewdata, latent_id_imm = fetch_batch_information(nnewdata, mnewdata, batch_key, latent_imm)

            score_collection = []
            for i in range(num_batch):
                removedata = remove_major_batch_for_malignant_cells(mnewdata, label_key = label_key, batch_key = batch_key, num_minor_cell = 1, num_batch_remove = i)
                diff_hyper = []
                for k in range(num_hyperparameter_selection):
                    hyperparameter_collection = hyperparameter_selection(dataset = dataset, method = method)
                    score_remove = hyperparameter_setting(removedata, latent_info = latent_id_imm, batch_key= batch_key, label_key= label_key, group_key= group_key, max_clusters = 8,
                    hyperparameter_collection = hyperparameter_collection[k])
                    diff_hyper.append(score_remove)
                print(hyperparameter_collection)                    
                diff_hyper = np.squeeze(np.array(diff_hyper), axis=2).T
                diff_hyper_pd = convert_diffhyperlist_into_df(diff_hyper, ['trial_1','trial_2','trial_3'], True, str(method) + "_" + str(dataset) + "_batch_" + str(i) + '.csv')

                score_collection.append(np.average(diff_hyper, axis = 1))

            score_collection = np.array(score_collection).T

            ari_collection_mn = max_min_scale(score_collection[0])
            asw_batch_collection_mn = max_min_scale(score_collection[3])
            kbet_collection_mn = max_min_scale(score_collection[2])
            asw_collection_mn = max_min_scale(score_collection[1])

            bio_score_collection = [] 
            batch_score_collection = [] 
            overall_score_collection = []
            for i in range(score_collection.shape[1]):
                bio_score = np.mean((ari_collection_mn[i], asw_collection_mn[i]))
                bio_score_collection.append(bio_score)
                batch_score = np.mean((kbet_collection_mn[i], asw_batch_collection_mn[i]))
                batch_score_collection.append(batch_score)
                overall_score_collection.append(0.6 * bio_score + 0.4 * batch_score)

            bio_score_collection = np.array(bio_score_collection).reshape(1,len(bio_score_collection))
            batch_score_collection = np.array(batch_score_collection).reshape(1,len(batch_score_collection))
            overall_score_collection = np.array(overall_score_collection).reshape(1,len(overall_score_collection))
            
            score_collection = np.concatenate((score_collection, bio_score_collection, batch_score_collection, overall_score_collection), axis = 0)

            score_collection = convert_scorelist_into_df(score_collection, 
                pd_columns_name, 
                True, 
                "removed_batches_all_No_adver.csv")
            return score_collection

        if dataset == "lung":
            layers_key = 'counts'
            batch_key = 'batch'
            label_key = 'cell_type'
            group_key = 'cell_type'
            non_malignant_cell_indices = [1,2,3,4,7,8,9,10,11,12,13,14,15,16]
            malignant_cell_indices = [0,2]
            dataset_csv = "Lung_atlas_public.h5ad"
            idata = sc.read_h5ad(dataset_csv)

            nnewdata, mnewdata = data_spliting_del_round(idata, label_key, non_malignant_cell_indices = non_malignant_cell_indices, malignant_cell_indices = malignant_cell_indices, delete = delete)
            model_imm = train_model_firstVAE_noadver(nnewdata, layers_key, label_key)
            latent_imm = get_latent_UMAP(model_imm, nnewdata, batch_key, group_key, "X_noadverVAE_1", False)
            mnewdata, latent_id_imm = fetch_batch_information(nnewdata, mnewdata, batch_key, latent_imm)

            score_collection = []
            for i in range(num_batch):
                removedata = remove_major_batch_for_malignant_cells(mnewdata, label_key = label_key, batch_key = batch_key, num_minor_cell = 1, num_batch_remove = i)
                diff_hyper = []
                for k in range(num_hyperparameter_selection):
                    hyperparameter_collection = hyperparameter_selection(dataset = dataset, method = method)
                    score_remove = hyperparameter_setting(removedata, latent_info = latent_id_imm, batch_key= batch_key, label_key= label_key, group_key= group_key, max_clusters = 8,
                    hyperparameter_collection = hyperparameter_collection[k])
                    diff_hyper.append(score_remove)
                print(hyperparameter_collection)                    
                diff_hyper = np.squeeze(np.array(diff_hyper), axis=2).T
                diff_hyper_pd = convert_diffhyperlist_into_df(diff_hyper, ['trial_1','trial_2','trial_3'], True, str(method) + "_" + str(dataset) + "_batch_" + str(i) + '.csv')

                score_collection.append(np.average(diff_hyper, axis = 1))

            score_collection = np.array(score_collection).T

            ari_collection_mn = max_min_scale(score_collection[0])
            asw_batch_collection_mn = max_min_scale(score_collection[3])
            kbet_collection_mn = max_min_scale(score_collection[2])
            asw_collection_mn = max_min_scale(score_collection[1])

            bio_score_collection = [] 
            batch_score_collection = [] 
            overall_score_collection = []
            for i in range(score_collection.shape[1]):
                bio_score = np.mean((ari_collection_mn[i], asw_collection_mn[i]))
                bio_score_collection.append(bio_score)
                batch_score = np.mean((kbet_collection_mn[i], asw_batch_collection_mn[i]))
                batch_score_collection.append(batch_score)
                overall_score_collection.append(0.6 * bio_score + 0.4 * batch_score)

            bio_score_collection = np.array(bio_score_collection).reshape(1,len(bio_score_collection))
            batch_score_collection = np.array(batch_score_collection).reshape(1,len(batch_score_collection))
            overall_score_collection = np.array(overall_score_collection).reshape(1,len(overall_score_collection))
            
            score_collection = np.concatenate((score_collection, bio_score_collection, batch_score_collection, overall_score_collection), axis = 0)

            score_collection = convert_scorelist_into_df(score_collection, 
                pd_columns_name, 
                True, 
                "removed_batches_lung_No_adver.csv")
            return score_collection

        if dataset == "pancreas":
            layers_key = 'counts'
            batch_key = 'tech'
            label_key = 'celltype'
            group_key = 'celltype'
            non_malignant_cell_indices = [1,5,6,7,8,9,10,11,12,13]
            malignant_cell_indices = [0,2]
            dataset_csv = "human_pancreas_norm_complexBatch.h5ad"
            idata = sc.read_h5ad(dataset_csv)

            nnewdata, mnewdata = data_spliting_del_round(idata, label_key, non_malignant_cell_indices = non_malignant_cell_indices, malignant_cell_indices = malignant_cell_indices, delete = delete)
            model_imm = train_model_firstVAE_noadver(nnewdata, layers_key, label_key)
            latent_imm = get_latent_UMAP(model_imm, nnewdata, batch_key, group_key, "X_noadverVAE_1", False)
            mnewdata, latent_id_imm = fetch_batch_information(nnewdata, mnewdata, batch_key, latent_imm)

            score_collection = []
            for i in range(num_batch):
                removedata = remove_major_batch_for_malignant_cells(mnewdata, label_key = label_key, batch_key = batch_key, num_minor_cell = 1, num_batch_remove = i)
                diff_hyper = []
                for k in range(num_hyperparameter_selection):
                    hyperparameter_collection = hyperparameter_selection(dataset = dataset, method = method)
                    score_remove = hyperparameter_setting(removedata, latent_info = latent_id_imm, batch_key= batch_key, label_key= label_key, group_key= group_key, max_clusters = 8,
                    hyperparameter_collection = hyperparameter_collection[k])
                    diff_hyper.append(score_remove)
                    
                diff_hyper = np.squeeze(np.array(diff_hyper), axis=2).T
                diff_hyper_pd = convert_diffhyperlist_into_df(diff_hyper, ['trial_1','trial_2','trial_3'], True, str(method) + "_" + str(dataset) + "_batch_" + str(i) + '.csv')
                print(hyperparameter_collection)
                score_collection.append(np.average(diff_hyper, axis = 1))
                
            score_collection = np.array(score_collection).T

            ari_collection_mn = max_min_scale(score_collection[0])
            asw_batch_collection_mn = max_min_scale(score_collection[3])
            kbet_collection_mn = max_min_scale(score_collection[2])
            asw_collection_mn = max_min_scale(score_collection[1])

            bio_score_collection = [] 
            batch_score_collection = [] 
            overall_score_collection = []
            for i in range(score_collection.shape[1]):
                bio_score = np.mean((ari_collection_mn[i], asw_collection_mn[i]))
                bio_score_collection.append(bio_score)
                batch_score = np.mean((kbet_collection_mn[i], asw_batch_collection_mn[i]))
                batch_score_collection.append(batch_score)
                overall_score_collection.append(0.6 * bio_score + 0.4 * batch_score)

            bio_score_collection = np.array(bio_score_collection).reshape(1,len(bio_score_collection))
            batch_score_collection = np.array(batch_score_collection).reshape(1,len(batch_score_collection))
            overall_score_collection = np.array(overall_score_collection).reshape(1,len(overall_score_collection))
            
            score_collection = np.concatenate((score_collection, bio_score_collection, batch_score_collection, overall_score_collection), axis = 0)

            score_collection = convert_scorelist_into_df(score_collection, 
                pd_columns_name, 
                True, 
                "removed_batches_pancreas_No_adver.csv")
            return score_collection

    if method == "general_scvi":
        if dataset == "all":
            layers_key = 'counts'
            batch_key = 'batch'
            label_key = 'final_annotation'
            group_key = 'final_annotation'
            non_malignant_cell_indices = [0,2,3,4,6,7,8,9,10,11,12,13,14,15]
            malignant_cell_indices = [1,5]
            dataset_csv = "Immune_ALL_human.h5ad"
            idata = sc.read_h5ad(dataset_csv)

            mdata = malignant_cell_collection(idata, malignant_cell_indices, label_key)
            model_preprocessing(mdata, batch_key)

            score_collection = []
            for i in range(num_batch):
                removedata = remove_major_batch_for_malignant_cells(mdata, label_key = label_key, batch_key = batch_key, num_minor_cell = 1, num_batch_remove = i)
                diff_hyper = []
                for k in range(num_hyperparameter_selection):
                    hyperparameter_collection = hyperparameter_selection(dataset = dataset, method = method)
                    score_remove = hyperparameter_setting_general_scvi(removedata, layer_key = layers_key, batch_key= batch_key, label_key= label_key, group_key= group_key, max_clusters = 8,
                    hyperparameter_collection = hyperparameter_collection[k])
                    diff_hyper.append(score_remove)
                print(hyperparameter_collection)                    
                diff_hyper = np.squeeze(np.array(diff_hyper), axis=2).T
                diff_hyper_pd = convert_diffhyperlist_into_df(diff_hyper, ['trial_1','trial_2','trial_3'], True, str(method) + "_" + str(dataset) + "_batch_" + str(i) + '.csv')

                score_collection.append(np.average(diff_hyper, axis = 1))

            score_collection = np.array(score_collection).T

            ari_collection_mn = max_min_scale(score_collection[0])
            asw_batch_collection_mn = max_min_scale(score_collection[3])
            kbet_collection_mn = max_min_scale(score_collection[2])
            asw_collection_mn = max_min_scale(score_collection[1])

            bio_score_collection = [] 
            batch_score_collection = [] 
            overall_score_collection = []
            for i in range(score_collection.shape[1]):
                bio_score = np.mean((ari_collection_mn[i], asw_collection_mn[i]))
                bio_score_collection.append(bio_score)
                batch_score = np.mean((kbet_collection_mn[i], asw_batch_collection_mn[i]))
                batch_score_collection.append(batch_score)
                overall_score_collection.append(0.6 * bio_score + 0.4 * batch_score)

            bio_score_collection = np.array(bio_score_collection).reshape(1,len(bio_score_collection))
            batch_score_collection = np.array(batch_score_collection).reshape(1,len(batch_score_collection))
            overall_score_collection = np.array(overall_score_collection).reshape(1,len(overall_score_collection))
            
            score_collection = np.concatenate((score_collection, bio_score_collection, batch_score_collection, overall_score_collection), axis = 0)

            score_collection = convert_scorelist_into_df(score_collection, 
                pd_columns_name, 
                True, 
                "removed_batches_all_general_scvi.csv")
            return score_collection

        if dataset == "lung":
            layers_key = 'counts'
            batch_key = 'batch'
            label_key = 'cell_type'
            group_key = 'cell_type'
            non_malignant_cell_indices = [1,2,3,4,7,8,9,10,11,12,13,14,15,16]
            malignant_cell_indices = [0,2]
            dataset_csv = "Lung_atlas_public.h5ad"
            idata = sc.read_h5ad(dataset_csv)

            mdata = malignant_cell_collection(idata, malignant_cell_indices, label_key)
            model_preprocessing(mdata, batch_key)

            score_collection = []
            for i in range(num_batch):
                removedata = remove_major_batch_for_malignant_cells(mdata, label_key = label_key, batch_key = batch_key, num_minor_cell = 1, num_batch_remove = i)
                diff_hyper = []
                for k in range(num_hyperparameter_selection):
                    hyperparameter_collection = hyperparameter_selection(dataset = dataset, method = method)
                    score_remove = hyperparameter_setting_general_scvi(removedata, layer_key = layers_key, batch_key= batch_key, label_key= label_key, group_key= group_key, max_clusters = 8,
                    hyperparameter_collection = hyperparameter_collection[k])
                    diff_hyper.append(score_remove)
                print(hyperparameter_collection)
                diff_hyper = np.squeeze(np.array(diff_hyper), axis=2).T
                diff_hyper_pd = convert_diffhyperlist_into_df(diff_hyper, ['trial_1','trial_2','trial_3'], True, str(method) + "_" + str(dataset) + "_batch_" + str(i) + '.csv')

                score_collection.append(np.average(diff_hyper, axis = 1))

            score_collection = np.array(score_collection).T

            ari_collection_mn = max_min_scale(score_collection[0])
            asw_batch_collection_mn = max_min_scale(score_collection[3])
            kbet_collection_mn = max_min_scale(score_collection[2])
            asw_collection_mn = max_min_scale(score_collection[1])

            bio_score_collection = [] 
            batch_score_collection = [] 
            overall_score_collection = []
            for i in range(score_collection.shape[1]):
                bio_score = np.mean((ari_collection_mn[i], asw_collection_mn[i]))
                bio_score_collection.append(bio_score)
                batch_score = np.mean((kbet_collection_mn[i], asw_batch_collection_mn[i]))
                batch_score_collection.append(batch_score)
                overall_score_collection.append(0.6 * bio_score + 0.4 * batch_score)

            bio_score_collection = np.array(bio_score_collection).reshape(1,len(bio_score_collection))
            batch_score_collection = np.array(batch_score_collection).reshape(1,len(batch_score_collection))
            overall_score_collection = np.array(overall_score_collection).reshape(1,len(overall_score_collection))
            
            score_collection = np.concatenate((score_collection, bio_score_collection, batch_score_collection, overall_score_collection), axis = 0)

            score_collection = convert_scorelist_into_df(score_collection, 
                pd_columns_name, 
                True, 
                "removed_batches_lung_general_scvi.csv")
            return score_collection

        if dataset == "pancreas":
            layers_key = 'counts'
            batch_key = 'tech'
            label_key = 'celltype'
            group_key = 'celltype'
            non_malignant_cell_indices = [1,5,6,7,8,9,10,11,12,13]
            malignant_cell_indices = [0,2]
            dataset_csv = "human_pancreas_norm_complexBatch.h5ad"
            idata = sc.read_h5ad(dataset_csv)

            mdata = malignant_cell_collection(idata, malignant_cell_indices, label_key)
            model_preprocessing(mdata, batch_key)

            score_collection = []
            for i in range(num_batch):
                removedata = remove_major_batch_for_malignant_cells(mdata, label_key = label_key, batch_key = batch_key, num_minor_cell = 1, num_batch_remove = i)
                diff_hyper = []
                for k in range(num_hyperparameter_selection):
                    hyperparameter_collection = hyperparameter_selection(dataset = dataset, method = method)
                    score_remove = hyperparameter_setting_general_scvi(removedata, layer_key = layers_key, batch_key= batch_key, label_key= label_key, group_key= group_key, max_clusters = 8,
                    hyperparameter_collection = hyperparameter_collection[k])
                    diff_hyper.append(score_remove)
                print(hyperparameter_collection)                    
                diff_hyper = np.squeeze(np.array(diff_hyper), axis=2).T
                diff_hyper_pd = convert_diffhyperlist_into_df(diff_hyper, ['trial_1','trial_2','trial_3'], True, str(method) + "_" + str(dataset) + "_batch_" + str(i) + '.csv')

                score_collection.append(np.average(diff_hyper, axis = 1))

            score_collection = np.array(score_collection).T

            ari_collection_mn = max_min_scale(score_collection[0])
            asw_batch_collection_mn = max_min_scale(score_collection[3])
            kbet_collection_mn = max_min_scale(score_collection[2])
            asw_collection_mn = max_min_scale(score_collection[1])

            bio_score_collection = [] 
            batch_score_collection = [] 
            overall_score_collection = []
            for i in range(score_collection.shape[1]):
                bio_score = np.mean((ari_collection_mn[i], asw_collection_mn[i]))
                bio_score_collection.append(bio_score)
                batch_score = np.mean((kbet_collection_mn[i], asw_batch_collection_mn[i]))
                batch_score_collection.append(batch_score)
                overall_score_collection.append(0.6 * bio_score + 0.4 * batch_score)

            bio_score_collection = np.array(bio_score_collection).reshape(1,len(bio_score_collection))
            batch_score_collection = np.array(batch_score_collection).reshape(1,len(batch_score_collection))
            overall_score_collection = np.array(overall_score_collection).reshape(1,len(overall_score_collection))
            
            score_collection = np.concatenate((score_collection, bio_score_collection, batch_score_collection, overall_score_collection), axis = 0)

            score_collection = convert_scorelist_into_df(score_collection, 
                pd_columns_name, 
                True, 
                "removed_batches_pancreas_general_scvi.csv")
            return score_collection