import torch
import scvi
import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np


import warnings
from math import ceil
from typing import Any, Dict, Optional, Union

import numpy as np

from scvi.dataloaders import DataSplitter
from scvi.train import TrainingPlan, AdversarialTrainingPlan, TrainRunner


def _check_warmup(
    plan_kwargs: Dict[str, Any],
    max_epochs: int,
    n_cells: int,
    batch_size: int,
    train_size: float = 1.0,
) -> None:
    """
    Raises a warning if the max_kl_weight is not reached by the end of training.
    Parameters
    ----------
    plan_kwargs
        Keyword args for :class:`~scvi.train.TrainingPlan`.
    max_epochs
        Number of passes through the dataset.
    n_cells
        Number of cells in the whole datasets.
    batch_size
        Minibatch size to use during training.
    train_size
        Fraction of cells used for training.
    """
    _WARNING_MESSAGE = (
        "max_{mode}={max} is less than n_{mode}_kl_warmup={warm_up}. "
        "The max_kl_weight will not be reached during training."
    )

    n_steps_kl_warmup = plan_kwargs.get("n_steps_kl_warmup", None)
    n_epochs_kl_warmup = plan_kwargs.get("n_epochs_kl_warmup", None)

    # The only time n_steps_kl_warmup is used is when n_epochs_kl_warmup is explicitly
    # set to None. This also catches the case when both n_epochs_kl_warmup and
    # n_steps_kl_warmup are set to None and max_kl_weight will always be reached.
    if (
        "n_epochs_kl_warmup" in plan_kwargs
        and plan_kwargs["n_epochs_kl_warmup"] is None
    ):
        n_cell_train = ceil(train_size * n_cells)
        steps_per_epoch = n_cell_train // batch_size + (n_cell_train % batch_size >= 3)
        max_steps = max_epochs * steps_per_epoch
        if n_steps_kl_warmup and max_steps < n_steps_kl_warmup:
            warnings.warn(
                _WARNING_MESSAGE.format(
                    mode="steps", max=max_steps, warm_up=n_steps_kl_warmup
                )
            )
    elif n_epochs_kl_warmup:
        if max_epochs < n_epochs_kl_warmup:
            warnings.warn(
                _WARNING_MESSAGE.format(
                    mode="epochs", max=max_epochs, warm_up=n_epochs_kl_warmup
                )
            )
    else:
        if max_epochs < 400:
            warnings.warn(
                _WARNING_MESSAGE.format(mode="epochs", max=max_epochs, warm_up=400)
            )

class UnsupervisedadverTrainingMixin_imm:
    def train(
        self,
        max_epochs: Optional[int] = None,
        use_gpu: Optional[Union[str, int, bool]] = None,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        batch_size: int = 128,
        early_stopping: bool = False,
        plan_kwargs: Optional[dict] = None,
        **trainer_kwargs,
    ):
        """
        Train the model.
        Parameters
        ----------
        max_epochs
            Number of passes through the dataset. If `None`, defaults to
            `np.min([round((20000 / n_cells) * 400), 400])`
        use_gpu
            Use default GPU if available (if None or True), or index of GPU to use (if int),
            or name of GPU (if str, e.g., `'cuda:0'`), or use CPU (if False).
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        batch_size
            Minibatch size to use during training.
        early_stopping
            Perform early stopping. Additional arguments can be passed in `**kwargs`.
            See :class:`~scvi.train.Trainer` for further options.
        plan_kwargs
            Keyword args for :class:`~scvi.train.TrainingPlan`. Keyword arguments passed to
            `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        **trainer_kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """
        n_cells = self.adata.n_obs
        if max_epochs is None:
            max_epochs = int(np.min([round((20000 / n_cells) * 400), 400]))

        plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else dict()

        _check_warmup(plan_kwargs, max_epochs, n_cells, batch_size)

        data_splitter = DataSplitter(
            self.adata_manager,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
            use_gpu=use_gpu,
        )
        training_plan = AdversarialTrainingPlan(self.module, lr=0.01, lr_patience = 30, scale_adversarial_loss = 1, **plan_kwargs)

        es = "early_stopping"
        trainer_kwargs[es] = (
            early_stopping if es not in trainer_kwargs.keys() else trainer_kwargs[es]
        )
        runner = TrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            use_gpu=use_gpu,
            **trainer_kwargs,
        )
        return runner()


from typing import Optional

from anndata import AnnData
from scvi.module import VAE
from scvi.model.base import VAEMixin, BaseModelClass, UnsupervisedTrainingMixin
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import (
    LayerField,
    CategoricalObsField,
    NumericalObsField,
    CategoricalJointObsField,
    NumericalJointObsField,
)
from scvi.nn import DecoderSCVI

class SCVIadver_humanimm(VAEMixin, UnsupervisedadverTrainingMixin_imm, BaseModelClass):
    """
    single-cell Variational Inference [Lopez18]_.
    """

    def __init__(
        self,
        adata: AnnData,
        n_latent: int = 10,
        #n_hidden: int = 256,
        #n_layers: int = 5,
        **model_kwargs,
    ):
        super().__init__(adata)

        self.module = VAE(
            n_input=self.summary_stats["n_vars"],
            n_batch=self.summary_stats["n_batch"],
            n_latent=n_latent,
            #n_hidden=n_hidden,
            #n_layers=n_layers,
            **model_kwargs,
        )

        self._model_summary_string = (
            "SCVI Model with the following params: \nn_latent: {}"
        ).format(
            n_latent,
        )
        self.init_params_ = self._get_init_params(locals())

    @classmethod
    def setup_anndata(
        cls,
        adata: AnnData,
        batch_key: Optional[str] = None,
        labels_key: Optional[str] = None,
        layer: Optional[str] = None,
        **kwargs,
    ) -> Optional[AnnData]:
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            # Dummy fields required for VAE class.
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            NumericalObsField(
                REGISTRY_KEYS.SIZE_FACTOR_KEY, None, required=False
            ),
            CategoricalJointObsField(
                REGISTRY_KEYS.CAT_COVS_KEY, None
            ),
            NumericalJointObsField(
                REGISTRY_KEYS.CONT_COVS_KEY, None
            ),
        ]
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)


def data_spliting(adata, label_key, non_malignant_cell_indices, malignant_cell_indices):
    ######################################################## choosing to eliminate the cell_types
    
    # Pretending some of the cell_types are maglingant
    #adata.layers["counts"] = adata.X.copy()
    
    for ind,val in enumerate(non_malignant_cell_indices):
        checkdata = adata[adata.obs[label_key].isin(adata.obs[label_key].unique()[[val]])]

        # check whether the cell_type has float expression data
        if any([(k%1) for k in checkdata.layers['counts'].ravel()]):
            non_malignant_cell_indices[ind] = -1
    
    non_malignant_cell_indices = [i for i in non_malignant_cell_indices if i != -1]

    for ind,val in enumerate(malignant_cell_indices):
        checkdata = adata[adata.obs[label_key].isin(adata.obs[label_key].unique()[[val]])]

        # check whether the cell_type has float expression data
        if any([(k%1) for k in checkdata.layers['counts'].ravel()]):
            malignant_cell_indices[ind] = -1
    
    malignant_cell_indices = [i for i in malignant_cell_indices if i != -1]

    ######################################################## choosing to round the cell_types
    #np.round(adata[adata.obs['final_annotation'].isin(adata.obs['final_annotation'].unique()[[i for i in non_malignant_cell_indices]])].layers['counts'])
    #np.round(adata[adata.obs['final_annotation'].isin(adata.obs['final_annotation'].unique()[[i for i in malignant_cell_indices]])].layers['counts'])
    #adata[adata.obs['final_annotation'].isin(adata.obs['final_annotation'].unique()[[i for i in non_malignant_cell_indices]])].layers['counts'][adata[adata.obs['final_annotation'].isin(adata.obs['final_annotation'].unique()[[i for i in non_malignant_cell_indices]])].layers['counts']<0] = 0
    #adata[adata.obs['final_annotation'].isin(adata.obs['final_annotation'].unique()[[i for i in malignant_cell_indices]])].layers['counts'][adata[adata.obs['final_annotation'].isin(adata.obs['final_annotation'].unique()[[i for i in malignant_cell_indices]])].layers['counts']<0] = 0
    #################################################################################################################################
    ndata = adata[adata.obs[label_key].isin(adata.obs[label_key].unique()[[i for i in non_malignant_cell_indices]])]
    mdata = adata[adata.obs[label_key].isin(adata.obs[label_key].unique()[[i for i in malignant_cell_indices]])]
    return ndata, mdata


def train_model_firstVAE(ndata, layer_key, batch_key):
    ndata = ndata.copy()
    SCVIadver_humanimm.setup_anndata(ndata, layer=layer_key, batch_key = batch_key) 
    model_imm = SCVIadver_humanimm(ndata)
    model_imm.train(max_epochs=100)
    return model_imm

def get_latent_UMAP(model, ndata, batch_key, label_key, added_latent_key):
    latent = model.get_latent_representation()
    ndata.obsm[added_latent_key] = latent
    sc.pp.neighbors(ndata, use_rep=added_latent_key, n_neighbors=20)
    sc.tl.umap(ndata, min_dist=0.3)
    sc.pl.umap(ndata, color = [label_key, batch_key])


def fetch_batch_information(ndata, mdata, batch_key):
    batch_df_imm = pd.DataFrame(latent_imm, index = ndata.obs[batch_key])
    batch_df_mean_imm = batch_df_imm.groupby(batch_df_imm.index).mean()
    batch_df_mean_loc_imm = batch_df_mean_imm.loc[mdata.obs[batch_key]]
    latent_id_imm = [f'latent{i}' for i in range(batch_df_mean_loc_imm.shape[1])]
    mdata.obs[latent_id_imm] = batch_df_mean_loc_imm.values
    return mdata, latent_id_imm

def second_VAE(mdata):
    scvi.model.SCVI.setup_anndata(mdata, layer='counts', continuous_covariate_keys=latent_id_imm)
    sec_model_imm = scvi.model.SCVI(mdata, n_layers = 5, n_hidden = 512, n_latent = 10)
    sec_model_imm.train(max_epochs = 100, validation_size = 0.1, check_val_every_n_epoch = 5, early_stopping=True, 
    early_stopping_monitor='elbo_validation', early_stopping_patience = 5)

def plot_reconstruction_loss_and_elbo(model):
    train_recon_loss = model.history['reconstruction_loss_train']
    elbo_train = model.history['elbo_train']
    elbo_val = model.history['elbo_validation']
    val_recon_loss = model.history['reconstruction_loss_validation']
    ax = train_recon_loss.plot()
    elbo_train.plot(ax = ax)
    elbo_val.plot(ax = ax)
    val_recon_loss.plot(ax = ax)

def get_latent_UMAP(model, batch_key, label_key):
    latent = model.get_latent_representation()
    mdata.obsm["X_secVAE_1"] = latent
    sc.pp.neighbors(mdata, use_rep="X_secVAE_1", n_neighbors=20)
    sc.tl.umap(mdata, min_dist=0.3)
    sc.pl.umap(mdata, color = [label_key, batch_key])

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
        bdata = _binary_search_leiden_resolution(bdata, k = int(i), start = 0.1, end = 0.9, key_added ='final_annotation', random_state = 0, directed = False, 
        use_weights = False, _epsilon = 1e-3)

        adata.obs['cluster_{}'.format(int(i))] = bdata.obs['final_annotation']
        ari_score_collection.append(compute_ari(adata, group_key = group_key, cluster_key = 'cluster_{}'.format(int(i))))


    # Note, all keys should come from the columns in adata.obs
    ari_score = {f"maximum ARI_score with {int(k[np.argmax(ari_score_collection)])} clusters": np.max(ari_score_collection)}
    sc.pl.umap(adata, color = ['cluster_{}'.format(int(k[np.argmax(ari_score_collection)]))])
    kbet_score = kbet(adata, latent_key=latent_key, batch_key=batch_key, label_key=label_key)
    asw_score = compute_asw(adata, group_key = group_key, latent_key = latent_key)
    return (kbet_score, ari_score, asw_score)
