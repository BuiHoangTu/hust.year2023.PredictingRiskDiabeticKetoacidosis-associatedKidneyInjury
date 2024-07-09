from functools import partial
from pathlib import Path
import tabpfn.encoders as encoders

from tabpfn.transformer import TransformerModel
from tabpfn.utils import get_uniform_single_eval_pos_sampler
import torch
from torch import nn, tensor
import math
import subprocess as sp
import os
from torch.utils.checkpoint import checkpoint

from constants import TEMP_PATH


class CausalAttention(nn.Module):
    def __init__(self, dropout=0):
        super(CausalAttention, self).__init__()
        self.attention = nn.MultiheadAttention(3, 1, dropout=dropout)

    def forward(self, x):
        attn_output, _ = self.attention(
            x, x, x, attn_mask=self.generate_square_subsequent_mask(x.size(0))
        )
        return attn_output

    def generate_square_subsequent_mask(self, sz):
        mask = torch.tril(torch.ones(sz, sz)).transpose(0, 1)
        mask = mask.masked_fill(mask == 0, float("-inf")).masked_fill(
            mask == 1, float(0.0)
        )
        return mask


class LstmTabPFN(TransformerModel):
    @staticmethod
    def from_parent(transformerModel, time_series_feature_count):
        return LstmTabPFN(
            transformerModel.encoder,
            transformerModel.n_out,
            transformerModel.ninp,
            transformerModel.nhead,
            transformerModel.nhid,
            transformerModel.nlayers,
            transformerModel.dropout,
            transformerModel.style_encoder,
            transformerModel.y_encoder,
            transformerModel.pos_encoder,
            transformerModel.decoder,
            transformerModel.input_normalization,
            transformerModel.init_method,
            transformerModel.pre_norm,
            transformerModel.activation,
            transformerModel.recompute_attn,
            transformerModel.num_global_att_tokens,
            transformerModel.full_attention,
            transformerModel.all_layers_same_init,
            transformerModel.efficient_eval_masking,
            time_series_feature_count,
        )
    
    def __init__(
        self,
        encoder,
        n_out,
        ninp,
        nhead,
        nhid,
        nlayers,
        dropout=0,
        style_encoder=None,
        y_encoder=None,
        pos_encoder=None,
        decoder=None,
        input_normalization=False,
        init_method=None,
        pre_norm=False,
        activation="gelu",
        recompute_attn=False,
        num_global_att_tokens=0,
        full_attention=False,
        all_layers_same_init=False,
        efficient_eval_masking=True,
        time_series_feature_count=0,
    ):
        super().__init__(
            encoder,
            n_out,
            ninp,
            nhead,
            nhid,
            nlayers,
            dropout,
            style_encoder,
            y_encoder,
            pos_encoder,
            decoder,
            input_normalization,
            init_method,
            pre_norm,
            activation,
            recompute_attn,
            num_global_att_tokens,
            full_attention,
            all_layers_same_init,
            efficient_eval_masking,
        )

        assert time_series_feature_count > 0, "time_series_feature_count should be > 0"

        self.time_series_feature_count = time_series_feature_count
        self.lstm = nn.LSTM(time_series_feature_count, time_series_feature_count, dropout=dropout, batch_first=True)

        self.attention = CausalAttention(dropout=dropout)

    def forward(self, src, src_mask=None, single_eval_pos=None):
        assert isinstance(
            src, tuple
        ), "inputs (src) have to be given as (x,y) or (style,x,y) tuple"
        if len(src) == 2:  # (x,y) and no style
            src = (None,) + src

        assert isinstance(
            src[1], tuple
        ), "inputs (src.x) have to be given as (x_dynamic, x_static)"

        X_dynamic, X_static = src[1]
        X_dynamic, _ = self.lstm(X_dynamic)

        X_dynamic = X_dynamic.transpose(0, 1)  # (batch, seq, feature)
        X_dynamic = self.attention(X_dynamic)
        X_dynamic = X_dynamic.transpose(0, 1)  # (seq, batch, feature) -> (batch, seq, feature)

        X_dynamic = X_dynamic[:, -1, :] # get the last hidden state

        combined = torch.cat((X_dynamic, X_static), dim=1)
        
        src = (src[0], combined, src[2])

        return super().forward(src, src_mask=src_mask, single_eval_pos=single_eval_pos)


###### replace original utils ######


def get_gpu_memory():
    command = "nvidia-smi"
    memory_free_info = sp.check_output(command.split()).decode("ascii")
    return memory_free_info


def load_model_only_inference(path, filename, device, time_series_feature_count):
    """
    Loads a saved model from the specified position. This function only restores inference capabilities and
    cannot be used for further training.
    """

    model_state, optimizer_state, config_sample = torch.load(
        os.path.join(path, filename), map_location="cpu"
    )

    if (
        (
            "nan_prob_no_reason" in config_sample
            and config_sample["nan_prob_no_reason"] > 0.0
        )
        or (
            "nan_prob_a_reason" in config_sample
            and config_sample["nan_prob_a_reason"] > 0.0
        )
        or (
            "nan_prob_unknown_reason" in config_sample
            and config_sample["nan_prob_unknown_reason"] > 0.0
        )
    ):
        encoder = encoders.NanHandlingEncoder
    else:
        encoder = partial(encoders.Linear, replace_nan_by_zero=True)

    n_out = config_sample["max_num_classes"]

    device = device if torch.cuda.is_available() else "cpu:0"
    encoder = encoder(config_sample["num_features"], config_sample["emsize"])

    nhid = config_sample["emsize"] * config_sample["nhid_factor"]
    y_encoder_generator = (
        encoders.get_Canonical(config_sample["max_num_classes"])
        if config_sample.get("canonical_y_encoder", False)
        else encoders.Linear
    )

    assert config_sample["max_num_classes"] > 2
    loss = torch.nn.CrossEntropyLoss(
        reduction="none", weight=torch.ones(int(config_sample["max_num_classes"]))
    )

    model = LstmTabPFN(
        encoder,
        n_out,
        config_sample["emsize"],
        config_sample["nhead"],
        nhid,
        config_sample["nlayers"],
        y_encoder=y_encoder_generator(1, config_sample["emsize"]),
        dropout=config_sample["dropout"],
        efficient_eval_masking=config_sample["efficient_eval_masking"],
        time_series_feature_count=time_series_feature_count,
    )

    # print(f"Using a Transformer with {sum(p.numel() for p in model.parameters()) / 1000 / 1000:.{2}f} M parameters")

    model.criterion = loss
    module_prefix = "module."
    model_state = {k.replace(module_prefix, ""): v for k, v in model_state.items()}
    
    # init state for lstm if not exist
    if "lstm.weight_ih_l0" not in model_state:
        for name, param in model.lstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)
        model_state["lstm.weight_ih_l0"] = model.lstm.weight_ih_l0
        model_state["lstm.weight_hh_l0"] = model.lstm.weight_hh_l0
        model_state["lstm.bias_ih_l0"] = model.lstm.bias_ih_l0
        model_state["lstm.bias_hh_l0"] = model.lstm.bias_hh_l0
    
    # init state for attention if not exist
    if "attention.attention.in_proj_weight" not in model_state:
        for name, param in model.attention.attention.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)
        model_state["attention.attention.in_proj_weight"] = model.attention.attention.in_proj_weight
        model_state["attention.attention.in_proj_bias"] = model.attention.attention.in_proj_bias
        model_state["attention.attention.out_proj.weight"] = model.attention.attention.out_proj.weight
        model_state["attention.attention.out_proj.bias"] = model.attention.attention.out_proj.bias    
    
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    return (float("inf"), float("inf"), model), config_sample  # no loss measured


def load_model(path, filename, device, eval_positions, verbose):
    # TODO: This function only restores evaluation functionality but training canät be continued. It is also not flexible.
    # print('Loading....')
    print("!! Warning: GPyTorch must be installed !!")
    model_state, optimizer_state, config_sample = torch.load(
        os.path.join(path, filename), map_location="cpu"
    )
    if (
        "differentiable_hyperparameters" in config_sample
        and "prior_mlp_activations" in config_sample["differentiable_hyperparameters"]
    ):
        config_sample["differentiable_hyperparameters"]["prior_mlp_activations"][
            "choice_values_used"
        ] = config_sample["differentiable_hyperparameters"]["prior_mlp_activations"][
            "choice_values"
        ]
        config_sample["differentiable_hyperparameters"]["prior_mlp_activations"][
            "choice_values"
        ] = [
            torch.nn.Tanh
            for k in config_sample["differentiable_hyperparameters"][
                "prior_mlp_activations"
            ]["choice_values"]
        ]

    config_sample["categorical_features_sampler"] = lambda: lambda x: ([], [], [])
    config_sample["num_features_used_in_training"] = config_sample["num_features_used"]
    config_sample["num_features_used"] = lambda: config_sample["num_features"]
    config_sample["num_classes_in_training"] = config_sample["num_classes"]
    config_sample["num_classes"] = 2
    config_sample["batch_size_in_training"] = config_sample["batch_size"]
    config_sample["batch_size"] = 1
    config_sample["bptt_in_training"] = config_sample["bptt"]
    config_sample["bptt"] = 10
    config_sample["bptt_extra_samples_in_training"] = config_sample[
        "bptt_extra_samples"
    ]
    config_sample["bptt_extra_samples"] = None

    # print('Memory', str(get_gpu_memory()))

    model = get_model(config_sample, device=device, should_train=False, verbose=verbose)
    module_prefix = "module."
    model_state = {k.replace(module_prefix, ""): v for k, v in model_state.items()}
    model[2].load_state_dict(model_state)
    model[2].to(device)
    model[2].eval()

    return model, config_sample


def fix_loaded_config_sample(loaded_config_sample, config):
    def copy_to_sample(*k):
        t, s = loaded_config_sample, config
        for k_ in k[:-1]:
            t = t[k_]
            s = s[k_]
        t[k[-1]] = s[k[-1]]

    copy_to_sample("num_features_used")
    copy_to_sample("num_classes")
    copy_to_sample(
        "differentiable_hyperparameters", "prior_mlp_activations", "choice_values"
    )


def load_config_sample(path, template_config):
    model_state, optimizer_state, loaded_config_sample = torch.load(
        path, map_location="cpu"
    )
    fix_loaded_config_sample(loaded_config_sample, template_config)
    return loaded_config_sample


def get_default_spec(test_datasets, valid_datasets):
    bptt = 10000
    eval_positions = [
        1000,
        2000,
        3000,
        4000,
        5000,
    ]  # list(2 ** np.array([4, 5, 6, 7, 8, 9, 10, 11, 12]))
    max_features = max(
        [X.shape[1] for (_, X, _, _, _, _) in test_datasets]
        + [X.shape[1] for (_, X, _, _, _, _) in valid_datasets]
    )
    max_splits = 5

    return bptt, eval_positions, max_features, max_splits


def get_mlp_prior_hyperparameters(config):
    from tabpfn.priors.utils import gamma_sampler_f

    config = {
        hp: (list(config[hp].values())[0]) if type(config[hp]) is dict else config[hp]
        for hp in config
    }

    if "random_feature_rotation" not in config:
        config["random_feature_rotation"] = True

    if "prior_sigma_gamma_k" in config:
        sigma_sampler = gamma_sampler_f(
            config["prior_sigma_gamma_k"], config["prior_sigma_gamma_theta"]
        )
        config["init_std"] = sigma_sampler
    if "prior_noise_std_gamma_k" in config:
        noise_std_sampler = gamma_sampler_f(
            config["prior_noise_std_gamma_k"], config["prior_noise_std_gamma_theta"]
        )
        config["noise_std"] = noise_std_sampler

    return config


def get_gp_mix_prior_hyperparameters(config):
    return {
        "lengthscale_concentration": config["prior_lengthscale_concentration"],
        "nu": config["prior_nu"],
        "outputscale_concentration": config["prior_outputscale_concentration"],
        "categorical_data": config["prior_y_minmax_norm"],
        "y_minmax_norm": config["prior_lengthscale_concentration"],
        "noise_concentration": config["prior_noise_concentration"],
        "noise_rate": config["prior_noise_rate"],
    }


def get_gp_prior_hyperparameters(config):
    return {
        hp: (list(config[hp].values())[0]) if type(config[hp]) is dict else config[hp]
        for hp in config
    }


def get_meta_gp_prior_hyperparameters(config):
    from tabpfn.priors.utils import trunc_norm_sampler_f

    config = {
        hp: (list(config[hp].values())[0]) if type(config[hp]) is dict else config[hp]
        for hp in config
    }

    if "outputscale_mean" in config:
        outputscale_sampler = trunc_norm_sampler_f(
            config["outputscale_mean"],
            config["outputscale_mean"] * config["outputscale_std_f"],
        )
        config["outputscale"] = outputscale_sampler
    if "lengthscale_mean" in config:
        lengthscale_sampler = trunc_norm_sampler_f(
            config["lengthscale_mean"],
            config["lengthscale_mean"] * config["lengthscale_std_f"],
        )
        config["lengthscale"] = lengthscale_sampler

    return config


def get_model(
    config,
    device,
    should_train=True,
    verbose=False,
    state_dict=None,
    epoch_callback=None,
):
    import tabpfn.priors as priors
    from tabpfn.train import train, Losses

    extra_kwargs = {}
    verbose_train, verbose_prior = verbose >= 1, verbose >= 2
    config["verbose"] = verbose_prior

    if "aggregate_k_gradients" not in config or config["aggregate_k_gradients"] is None:
        config["aggregate_k_gradients"] = math.ceil(
            config["batch_size"]
            * (
                (config["nlayers"] * config["emsize"] * config["bptt"] * config["bptt"])
                / 10824640000
            )
        )

    config["num_steps"] = math.ceil(
        config["num_steps"] * config["aggregate_k_gradients"]
    )
    config["batch_size"] = math.ceil(
        config["batch_size"] / config["aggregate_k_gradients"]
    )
    config["recompute_attn"] = (
        config["recompute_attn"] if "recompute_attn" in config else False
    )

    def make_get_batch(model_proto, **extra_kwargs):
        def new_get_batch(
            batch_size,
            seq_len,
            num_features,
            hyperparameters,
            device,
            model_proto=model_proto,
            **kwargs,
        ):
            kwargs = {**extra_kwargs, **kwargs}  # new args overwrite pre-specified args
            return model_proto.get_batch(
                batch_size=batch_size,
                seq_len=seq_len,
                device=device,
                hyperparameters=hyperparameters,
                num_features=num_features,
                **kwargs,
            )

        return new_get_batch

    if config["prior_type"] == "prior_bag":
        # Prior bag combines priors
        get_batch_gp = make_get_batch(priors.fast_gp)
        get_batch_mlp = make_get_batch(priors.mlp)
        if "flexible" in config and config["flexible"]:
            get_batch_gp = make_get_batch(
                priors.flexible_categorical, **{"get_batch": get_batch_gp}
            )
            get_batch_mlp = make_get_batch(
                priors.flexible_categorical, **{"get_batch": get_batch_mlp}
            )
        prior_bag_hyperparameters = {
            "prior_bag_get_batch": (get_batch_gp, get_batch_mlp),
            "prior_bag_exp_weights_1": 2.0,
        }
        prior_hyperparameters = {
            **get_mlp_prior_hyperparameters(config),
            **get_gp_prior_hyperparameters(config),
            **prior_bag_hyperparameters,
        }
        model_proto = priors.prior_bag
    else:
        if config["prior_type"] == "mlp":
            prior_hyperparameters = get_mlp_prior_hyperparameters(config)
            model_proto = priors.mlp
        elif config["prior_type"] == "gp":
            prior_hyperparameters = get_gp_prior_hyperparameters(config)
            model_proto = priors.fast_gp
        elif config["prior_type"] == "gp_mix":
            prior_hyperparameters = get_gp_mix_prior_hyperparameters(config)
            model_proto = priors.fast_gp_mix # type: ignore
        else:
            raise Exception()

        if "flexible" in config and config["flexible"]:
            get_batch_base = make_get_batch(model_proto)
            extra_kwargs["get_batch"] = get_batch_base
            model_proto = priors.flexible_categorical

    if config.get("flexible"):
        prior_hyperparameters["normalize_labels"] = True
        prior_hyperparameters["check_is_compatible"] = True
    prior_hyperparameters["prior_mlp_scale_weights_sqrt"] = (
        config["prior_mlp_scale_weights_sqrt"]
        if "prior_mlp_scale_weights_sqrt" in prior_hyperparameters
        else None
    )
    prior_hyperparameters["rotate_normalized_labels"] = (
        config["rotate_normalized_labels"]
        if "rotate_normalized_labels" in prior_hyperparameters
        else True
    )

    use_style = False

    if "differentiable" in config and config["differentiable"]:
        get_batch_base = make_get_batch(model_proto, **extra_kwargs)
        extra_kwargs = {
            "get_batch": get_batch_base,
            "differentiable_hyperparameters": config["differentiable_hyperparameters"],
        }
        model_proto = priors.differentiable_prior
        use_style = True
    print(f"Using style prior: {use_style}")

    if (
        ("nan_prob_no_reason" in config and config["nan_prob_no_reason"] > 0.0)
        or ("nan_prob_a_reason" in config and config["nan_prob_a_reason"] > 0.0)
        or (
            "nan_prob_unknown_reason" in config
            and config["nan_prob_unknown_reason"] > 0.0
        )
    ):
        encoder = encoders.NanHandlingEncoder
    else:
        encoder = partial(encoders.Linear, replace_nan_by_zero=True)

    if config["max_num_classes"] == 2:
        loss = Losses.bce
    elif config["max_num_classes"] > 2:
        loss = Losses.ce(config["max_num_classes"])

    check_is_compatible = (  # noqa: F841
        False
        if "multiclass_loss_type" not in config
        else (config["multiclass_loss_type"] == "compatible")
    ) 
    config["multiclass_type"] = (
        config["multiclass_type"] if "multiclass_type" in config else "rank"
    )
    config["mix_activations"] = (
        config["mix_activations"] if "mix_activations" in config else False
    )

    config["bptt_extra_samples"] = (
        config["bptt_extra_samples"] if "bptt_extra_samples" in config else None
    )
    config["eval_positions"] = (
        [int(config["bptt"] * 0.95)]
        if config["bptt_extra_samples"] is None
        else [int(config["bptt"])]
    )

    epochs = 0 if not should_train else config["epochs"]
    # print('MODEL BUILDER', model_proto, extra_kwargs['get_batch'])
    model = train(
        model_proto.DataLoader,
        loss,
        encoder,
        style_encoder_generator=encoders.StyleEncoder if use_style else None,
        emsize=config["emsize"],
        nhead=config["nhead"]
        # For unsupervised learning change to NanHandlingEncoder
        ,
        y_encoder_generator=(
            encoders.get_Canonical(config["max_num_classes"])
            if config.get("canonical_y_encoder", False)
            else encoders.Linear
        ),
        pos_encoder_generator=None,
        batch_size=config["batch_size"],
        nlayers=config["nlayers"],
        nhid=config["emsize"] * config["nhid_factor"],
        epochs=epochs,
        warmup_epochs=20,
        bptt=config["bptt"],
        gpu_device=device,
        dropout=config["dropout"],
        steps_per_epoch=config["num_steps"],
        single_eval_pos_gen=get_uniform_single_eval_pos_sampler(
            config.get("max_eval_pos", config["bptt"]),
            min_len=config.get("min_eval_pos", 0),
        ),
        load_weights_from_this_state_dict=state_dict,
        aggregate_k_gradients=config["aggregate_k_gradients"],
        recompute_attn=config["recompute_attn"],
        epoch_callback=epoch_callback,
        bptt_extra_samples=config["bptt_extra_samples"],
        train_mixed_precision=config["train_mixed_precision"],
        extra_prior_kwargs_dict={
            "num_features": config["num_features"],
            "hyperparameters": prior_hyperparameters
            # , 'dynamic_batch_size': 1 if ('num_global_att_tokens' in config and config['num_global_att_tokens']) else 2
            ,
            "batch_size_per_gp_sample": config.get("batch_size_per_gp_sample", None),
            **extra_kwargs,
        },
        lr=config["lr"],
        verbose=verbose_train,
        weight_decay=config.get("weight_decay", 0.0),
    )

    return model

import pickle
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "Manager":
            from settings import Manager

            return Manager
        try:
            return self.find_class_cpu(module, name)
        except:
            return None

    def find_class_cpu(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


def load_model_workflow(
    i, e, add_name, base_path, device="cpu", eval_addition="", only_inference=True, time_series_feature_count = 0
):
    """
    Workflow for loading a model and setting appropriate parameters for diffable hparam tuning.

    :param i:
    :param e:
    :param eval_positions_valid:
    :param add_name:
    :param base_path:
    :param device:
    :param eval_addition:
    :return:
    """

    def get_file(e):
        """
        Returns the different paths of model_file, model_path and results_file
        """
        model_file = (
            f"models_diff/prior_diff_real_checkpoint{add_name}_n_{i}_epoch_{e}.cpkt"
        )
        model_path = os.path.join(base_path, model_file)
        # print('Evaluate ', model_path)
        results_file = os.path.join(
            base_path,
            f"models_diff/prior_diff_real_results{add_name}_n_{i}_epoch_{e}_{eval_addition}.pkl",
        )
        return model_file, model_path, results_file

    def check_file(e):
        model_file, model_path, results_file = get_file(e)
        if not Path(model_path).is_file():  # or Path(results_file).is_file():
            print(
                "We have to download the TabPFN, as there is no checkpoint at ",
                model_path,
            )
            print("It has about 100MB, so this might take a moment.")
            import requests

            url = "https://github.com/automl/TabPFN/raw/main/tabpfn/models_diff/prior_diff_real_checkpoint_n_0_epoch_42.cpkt"
            r = requests.get(url, allow_redirects=True)
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            open(model_path, "wb").write(r.content)
        return model_file, model_path, results_file

    model_file = None
    if e == -1:
        for e_ in list(range(100, -1, -1)):
            model_file_, model_path_, results_file_ = check_file(e_)
            if model_file_ is not None:
                e = e_
                model_file, model_path, results_file = (
                    model_file_,
                    model_path_,
                    results_file_,
                )
                break
    else:
        model_file, model_path, results_file = check_file(e)

    if model_file is None:
        model_file, model_path, results_file = get_file(e)
        raise Exception("No checkpoint found at " + str(model_path))

    # print(f'Loading {model_file}')
    if only_inference:
        # print('Loading model that can be used for inference only')
        model, c = load_model_only_inference(base_path, model_file, device, time_series_feature_count)
    else:
        # until now also only capable of inference
        model, c = load_model(
            base_path, model_file, device, eval_positions=[], verbose=False
        )
    # model, c = load_model(base_path, model_file, device, eval_positions=[], verbose=False)

    return model, c, results_file

from sklearn.preprocessing import PowerTransformer, QuantileTransformer, RobustScaler

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils import column_or_1d
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from tabpfn.scripts.model_builder import load_model
import os
import pickle
import io
from tabpfn.utils import NOP, normalize_by_used_features_f
from tabpfn.utils import normalize_data, to_ranking_low_mem, remove_outliers
import random


class LstmTabPFNClassifier(BaseEstimator, ClassifierMixin):

    models_in_memory = {}

    def __init__(
        self,
        time_series_feature_count,
        device="cpu",
        base_path=TEMP_PATH,
        model_string="",
        N_ensemble_configurations=3,
        no_preprocess_mode=False,
        multiclass_decoder="permutation",
        feature_shift_decoder=True,
        only_inference=True,
        seed=0,
        no_grad=True,
        batch_size_inference=32,
        subsample_features=False,
    ):
        """
        Initializes the classifier and loads the model.
        Depending on the arguments, the model is either loaded from memory, from a file, or downloaded from the
        repository if no model is found.

        Can also be used to compute gradients with respect to the inputs X_train and X_test. Therefore no_grad has to be
        set to False and no_preprocessing_mode must be True. Furthermore, X_train and X_test need to be given as
        torch.Tensors and their requires_grad parameter must be set to True.


        :param device: If the model should run on cuda or cpu.
        :param base_path: Base path of the directory, from which the folders like models_diff can be accessed.
        :param model_string: Name of the model. Used first to check if the model is already in memory, and if not,
               tries to load a model with that name from the models_diff directory. It looks for files named as
               follows: "prior_diff_real_checkpoint" + model_string + "_n_0_epoch_e.cpkt", where e can be a number
               between 100 and 0, and is checked in a descending order.
        :param N_ensemble_configurations: The number of ensemble configurations used for the prediction. Thereby the
               accuracy, but also the running time, increases with this number.
        :param no_preprocess_mode: Specifies whether preprocessing is to be performed.
        :param multiclass_decoder: If set to permutation, randomly shifts the classes for each ensemble configuration.
        :param feature_shift_decoder: If set to true shifts the features for each ensemble configuration according to a
               random permutation.
        :param only_inference: Indicates if the model should be loaded to only restore inference capabilities or also
               training capabilities. Note that the training capabilities are currently not being fully restored.
        :param seed: Seed that is used for the prediction. Allows for a deterministic behavior of the predictions.
        :param batch_size_inference: This parameter is a trade-off between performance and memory consumption.
               The computation done with different values for batch_size_inference is the same,
               but it is split into smaller/larger batches.
        :param no_grad: If set to false, allows for the computation of gradients with respect to X_train and X_test.
               For this to correctly function no_preprocessing_mode must be set to true.
        :param subsample_features: If set to true and the number of features in the dataset exceeds self.max_features (100),
                the features are subsampled to self.max_features.
        """

        # Model file specification (Model name, Epoch)
        i = 0
        model_key = model_string + "|" + str(device)
        if model_key in self.models_in_memory:
            model, c, results_file = self.models_in_memory[model_key]
        else:
            model, c, results_file = load_model_workflow(
                i,
                -1,
                add_name=model_string,
                base_path=base_path,
                device=device,
                eval_addition="",
                only_inference=only_inference,
                time_series_feature_count=time_series_feature_count,
            )
            self.models_in_memory[model_key] = (model, c, results_file)
            if len(self.models_in_memory) == 2:
                print(
                    "Multiple models in memory. This might lead to memory issues. Consider calling remove_models_from_memory()"
                )
        # style, temperature = self.load_result_minimal(style_file, i, e)

        self.device = device
        self.model = model
        self.c = c
        self.style = None
        self.temperature = None
        self.N_ensemble_configurations = N_ensemble_configurations
        self.base__path = base_path
        self.base_path = base_path
        self.i = i
        self.model_string = model_string

        self.max_num_features = self.c["num_features"]
        self.max_num_classes = self.c["max_num_classes"]
        self.differentiable_hps_as_style = self.c["differentiable_hps_as_style"]

        self.no_preprocess_mode = no_preprocess_mode
        self.feature_shift_decoder = feature_shift_decoder
        self.multiclass_decoder = multiclass_decoder
        self.only_inference = only_inference
        self.seed = seed
        self.no_grad = no_grad
        self.subsample_features = subsample_features

        assert (
            self.no_preprocess_mode if not self.no_grad else True
        ), "If no_grad is false, no_preprocess_mode must be true, because otherwise no gradient can be computed."

        self.batch_size_inference = batch_size_inference

    def remove_models_from_memory(self):
        self.models_in_memory = {}

    def load_result_minimal(self, path, i, e):
        with open(path, "rb") as output:
            _, _, _, style, temperature, optimization_route = CustomUnpickler(
                output
            ).load()

            return style, temperature

    def _validate_targets(self, y):
        y_ = column_or_1d(y, warn=True)
        check_classification_targets(y)
        cls, y = np.unique(y_, return_inverse=True)
        if len(cls) < 2:
            raise ValueError(
                "The number of classes has to be greater than one; got %d class"
                % len(cls)
            )

        self.classes_ = cls

        return np.asarray(y, dtype=np.float64, order="C")

    def fit(self, X_static, X_dynamic, y, overwrite_warning=False):
        """
        Validates the training set and stores it.

        If clf.no_grad (default is True):
        X, y should be of type np.array
        else:
        X should be of type torch.Tensors (y can be np.array or torch.Tensor)
        """
        # if self.no_grad:
        #     # Check that X and y have correct shape
        #     X, y = check_X_y(X, y, force_all_finite=False)
        # Store the classes seen during fit
        y = self._validate_targets(y)
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(y)
        
        X_static = torch.tensor(X_static).to(self.device)
        X_dynamic = torch.tensor(X_dynamic).to(self.device)

        self.X_static_ = X_static
        self.X_dynamic_ = X_dynamic
        self.y_ = y

        if X_static.shape[1] + X_dynamic.shape[2]  > self.max_num_features:
            if self.subsample_features:
                print(
                    "WARNING: The number of features for this classifier is restricted to ",
                    self.max_num_features,
                    " and will be subsampled.",
                )
            else:
                raise ValueError(
                    "The number of features for this classifier is restricted to ",
                    self.max_num_features,
                )
        if len(np.unique(y)) > self.max_num_classes:
            raise ValueError(
                "The number of classes for this classifier is restricted to ",
                self.max_num_classes,
            )
        if X_static.shape[0] > 1024 and not overwrite_warning:
            raise ValueError(
                "⚠️ WARNING: TabPFN is not made for datasets with a trainingsize > 1024. Prediction might take a while, be less reliable. We advise not to run datasets > 10k samples, which might lead to your machine crashing (due to quadratic memory scaling of TabPFN). Please confirm you want to run by passing overwrite_warning=True to the fit function."
            )

        # Return the classifier
        return self

    def predict_proba(self, X_dynamic, X_static, normalize_with_test=False, return_logits=False):
        """
        Predict the probabilities for the input X depending on the training set previously passed in the method fit.

        If no_grad is true in the classifier the function takes X as a numpy.ndarray. If no_grad is false X must be a
        torch tensor and is not fully checked.
        """
        # Check is fit had been called
        check_is_fitted(self)
        
        # Input validation
        if self.no_grad:
            X_static = check_array(X_static, force_all_finite=False)
            X_static_full = np.concatenate([self.X_static_, X_static], axis=0)
            X_static_full = torch.tensor(X_static_full, device=self.device).float().unsqueeze(1)

            X_dynamic_full = np.concatenate([self.X_dynamic_, X_dynamic], axis=0)
            X_dynamic_full = torch.tensor(X_dynamic_full, device=self.device).float().unsqueeze(1)
        else:
            assert torch.is_tensor(self.X_static_) & torch.is_tensor(X_static), (
                "If no_grad is false, this function expects X as "
                "a tensor to calculate a gradient"
            )
            X_static_full = torch.cat((self.X_static_, X_static), dim=0).float().unsqueeze(1).to(self.device)
            X_dynamic_full = torch.cat((self.X_dynamic_, X_dynamic), dim=0).float().unsqueeze(1).to(self.device)

            if int(torch.isnan(X_static_full).sum()) or int(torch.isnan(X_dynamic_full).sum()):
                print(
                    "X contains nans and the gradient implementation is not designed to handel nans."
                )
                
        y_full = np.concatenate([self.y_, np.zeros(shape=X_static.shape[0])], axis=0)
        y_full = torch.tensor(y_full, device=self.device).float().unsqueeze(1)

        eval_pos = self.X_static_.shape[0]

        prediction = transformer_predict(
            self.model[2],
            (X_dynamic_full, X_static_full),
            y_full,
            eval_pos,
            device=self.device,
            style=self.style,
            inference_mode=True,
            preprocess_transform="none" if self.no_preprocess_mode else "mix",
            normalize_with_test=normalize_with_test,
            N_ensemble_configurations=self.N_ensemble_configurations,
            softmax_temperature=self.temperature, # type: ignore
            multiclass_decoder=self.multiclass_decoder,
            feature_shift_decoder=self.feature_shift_decoder,
            differentiable_hps_as_style=self.differentiable_hps_as_style,
            seed=self.seed,
            return_logits=return_logits,
            no_grad=self.no_grad,
            batch_size_inference=self.batch_size_inference,
            **get_params_from_config(self.c),
        )
        prediction_, y_ = prediction.squeeze(0), y_full.squeeze(1).long()[eval_pos:]

        return prediction_.detach().cpu().numpy() if self.no_grad else prediction_

    def predict(self, X_dynamic, X_static, return_winning_probability=False, normalize_with_test=False):
        p = self.predict_proba(X_dynamic, X_static, normalize_with_test=normalize_with_test)
        y = np.argmax(p, axis=-1)
        y = self.classes_.take(np.asarray(y, dtype=np.intp))
        if return_winning_probability:
            return y, p.max(axis=-1)
        return y


import time


def transformer_predict(
    model,
    eval_xs,
    eval_ys,
    eval_position,
    device="cpu",
    max_features=100,
    style=None,
    inference_mode=False,
    num_classes=2,
    extend_features=True,
    normalize_with_test=False,
    normalize_to_ranking=False,
    softmax_temperature=0.0,
    multiclass_decoder="permutation",
    preprocess_transform="mix",
    categorical_feats=[],
    feature_shift_decoder=False,
    N_ensemble_configurations=10,
    batch_size_inference=16,
    differentiable_hps_as_style=False,
    average_logits=True,
    fp16_inference=False,
    normalize_with_sqrt=False,
    seed=0,
    no_grad=True,
    return_logits=False,
    **kwargs,
):
    """

    :param model:
    :param eval_xs:
    :param eval_ys:
    :param eval_position:
    :param rescale_features:
    :param device:
    :param max_features:
    :param style:
    :param inference_mode:
    :param num_classes:
    :param extend_features:
    :param normalize_to_ranking:
    :param softmax_temperature:
    :param multiclass_decoder:
    :param preprocess_transform:
    :param categorical_feats:
    :param feature_shift_decoder:
    :param N_ensemble_configurations:
    :param average_logits:
    :param normalize_with_sqrt:
    :param metric_used:
    :return:
    """
    num_classes = len(torch.unique(eval_ys))

    def predict(eval_xs, eval_ys, used_style, softmax_temperature, return_logits):
        eval_xs_dynamic, eval_xs_static = eval_xs

        # print("In function predict, line", 1047, "eval_xs_static.shape", eval_xs_static.shape, "eval_xs_dynamic.shape", eval_xs_dynamic.shape)

        # Initialize results array size S, B, Classes

        # no_grad disables inference_mode, because otherwise the gradients are lost
        inference_mode_call = (
            torch.inference_mode() if inference_mode and no_grad else NOP()
        )
        with inference_mode_call:
            start = time.time()

            # TODO: need more monitoring here eval_xs_dynamic.shape 1 or 2
            eval_xs_len = eval_xs_static.shape[1] + eval_xs_dynamic.shape[1]
            style = (
                used_style.repeat(eval_xs_len, 1)
                if used_style is not None
                else None
            )

            output = model(
                (
                    style,
                    eval_xs,
                    eval_ys.float(),
                ),
                single_eval_pos=eval_position,
            )[:, :, 0:num_classes]

            output = output[:, :, 0:num_classes] / torch.exp(softmax_temperature)
            if not return_logits:
                output = torch.nn.functional.softmax(output, dim=-1)
            # else:
            #    output[:, :, 1] = model((style.repeat(eval_xs.shape[1], 1) if style is not None else None, eval_xs, eval_ys.float()),
            #               single_eval_pos=eval_position)

            #    output[:, :, 1] = torch.sigmoid(output[:, :, 1]).squeeze(-1)
            #    output[:, :, 0] = 1 - output[:, :, 1]

        # print('RESULTS', eval_ys.shape, torch.unique(eval_ys, return_counts=True), output.mean(axis=0))

        return output

    def preprocess_input(eval_xs, preprocess_transform):
        import warnings

        eval_xs_dynamic, eval_xs_static = eval_xs

        if eval_xs_static.shape[1] > 1:
            raise Exception("Transforms only allow one batch dim - TODO")

        # TODO: also check for dynamic features (+ eval_xs_dynamic.shape[3])
        if eval_xs_static.shape[2] > max_features:
            eval_xs = eval_xs[
                :,
                :,
                sorted(np.random.choice(eval_xs.shape[2], max_features, replace=False)),
            ]

        if preprocess_transform != "none":
            if preprocess_transform == "power" or preprocess_transform == "power_all":
                pt = PowerTransformer(standardize=True)
            elif (
                preprocess_transform == "quantile"
                or preprocess_transform == "quantile_all"
            ):
                pt = QuantileTransformer(output_distribution="normal")
            elif (
                preprocess_transform == "robust" or preprocess_transform == "robust_all"
            ):
                pt = RobustScaler(unit_variance=True)

        # eval_xs, eval_ys = normalize_data(eval_xs), normalize_data(eval_ys)
        eval_xs_static = normalize_data(
            eval_xs_static,
            normalize_positions=-1 if normalize_with_test else eval_position,
        )

        # Removing empty features
        eval_xs_static = eval_xs_static[:, 0, :]
        sel = [
            len(torch.unique(eval_xs_static[0 : eval_ys.shape[0], col])) > 1
            for col in range(eval_xs_static.shape[1])
        ]
        eval_xs_static = eval_xs_static[:, sel]

        warnings.simplefilter("error")
        if preprocess_transform != "none":
            eval_xs_static = eval_xs_static.cpu().numpy()
            feats = (
                set(range(eval_xs_static.shape[1]))
                if "all" in preprocess_transform
                else set(range(eval_xs_static.shape[1])) - set(categorical_feats)
            )
            for col in feats:
                try:
                    pt.fit(eval_xs_static[0:eval_position, col : col + 1])
                    trans = pt.transform(eval_xs_static[:, col : col + 1])
                    # print(scipy.stats.spearmanr(trans[~np.isnan(eval_xs[:, col:col+1])], eval_xs[:, col:col+1][~np.isnan(eval_xs[:, col:col+1])]))
                    eval_xs_static[:, col : col + 1] = trans
                except:
                    pass
            eval_xs_static = torch.tensor(eval_xs_static).float()
        warnings.simplefilter("default")

        eval_xs_static = eval_xs_static.unsqueeze(1)

        # TODO: Caution there is information leakage when to_ranking is used, we should not use it
        eval_xs_static = (
            remove_outliers(
                eval_xs_static,
                normalize_positions=-1 if normalize_with_test else eval_position,
            )
            if not normalize_to_ranking
            else normalize_data(to_ranking_low_mem(eval_xs_static))
        )
        # Rescale X
        eval_xs_static = normalize_by_used_features_f(
            eval_xs_static,
            eval_xs_static.shape[-1],
            max_features,
            normalize_with_sqrt=normalize_with_sqrt,
        )

        return (eval_xs_dynamic.to(device), eval_xs_static.to(device))

    eval_xs_dynamic, eval_xs_static = eval_xs
    eval_xs_dynamic, eval_xs_static, eval_ys = eval_xs_dynamic.to(device), eval_xs_static.to(device), eval_ys.to(device)
    eval_ys = eval_ys[:eval_position]

    model.to(device)

    model.eval()

    import itertools

    if not differentiable_hps_as_style:
        style = None

    if style is not None:
        style = style.to(device)
        style = style.unsqueeze(0) if len(style.shape) == 1 else style
        num_styles = style.shape[0]
        softmax_temperature = (
            softmax_temperature
            if softmax_temperature.shape # type: ignore
            else softmax_temperature.unsqueeze(0).repeat(num_styles) # type: ignore
        )
    else:
        num_styles = 1
        style = None
        softmax_temperature = torch.log(torch.tensor([0.8]))

    styles_configurations = range(0, num_styles)

    def get_preprocess(i):
        if i == 0:
            return "power_all"
        #            if i == 1:
        #                return 'robust_all'
        if i == 1:
            return "none"

    preprocess_transform_configurations = (
        ["none", "power_all"]
        if preprocess_transform == "mix"
        else [preprocess_transform]
    )

    if seed is not None:
        torch.manual_seed(seed)

    feature_shift_configurations = (
        torch.randperm(eval_xs_static.shape[2]) if feature_shift_decoder else [0]
    )
    class_shift_configurations = (
        torch.randperm(len(torch.unique(eval_ys)))
        if multiclass_decoder == "permutation"
        else [0]
    )

    ensemble_configurations = list(
        itertools.product(class_shift_configurations, feature_shift_configurations)
    )
    # default_ensemble_config = ensemble_configurations[0]

    rng = random.Random(seed)
    rng.shuffle(ensemble_configurations)
    ensemble_configurations = list(
        itertools.product(
            ensemble_configurations,
            preprocess_transform_configurations,
            styles_configurations,
        )
    )
    ensemble_configurations = ensemble_configurations[0:N_ensemble_configurations]
    # if N_ensemble_configurations == 1:
    #    ensemble_configurations = [default_ensemble_config]

    output = None

    eval_xs_transformed = {}
    inputs, labels = [], []
    start = time.time()
    for ensemble_configuration in ensemble_configurations:
        (
            (class_shift_configuration, feature_shift_configuration),
            preprocess_transform_configuration,
            styles_configuration,
        ) = ensemble_configuration

        style_ = (
            style[styles_configuration : styles_configuration + 1, :]
            if style is not None
            else style
        )
        softmax_temperature_ = softmax_temperature[styles_configuration] # type: ignore

        eval_xs_dynamic_, eval_xs_static_, eval_ys_ = (
            eval_xs_dynamic.clone(),
            eval_xs_static.clone(),
            eval_ys.clone(),
        )

        if preprocess_transform_configuration in eval_xs_transformed:
            eval_xs_dynamic_, eval_xs_static_ = eval_xs_transformed[preprocess_transform_configuration]
            eval_xs_dynamic_ = eval_xs_dynamic_.clone()
            eval_xs_static_ = eval_xs_static_.clone()
        else:
            eval_xs_dynamic_, eval_xs_static_ = preprocess_input(
                (eval_xs_dynamic_, eval_xs_static_),
                preprocess_transform=preprocess_transform_configuration,
            )
            if no_grad:
                eval_xs_dynamic_ = eval_xs_dynamic_.detach()
                eval_xs_static_ = eval_xs_static_.detach()
            eval_xs_transformed[preprocess_transform_configuration] = (eval_xs_dynamic_, eval_xs_static_)

        eval_ys_ = ((eval_ys_ + class_shift_configuration) % num_classes).float()

        eval_xs_dynamic_ = torch.cat(
            [
                eval_xs_dynamic_[..., feature_shift_configuration:],
                eval_xs_dynamic_[..., :feature_shift_configuration],
            ],
            dim=-1,
        )

        eval_xs_static_ = torch.cat(
            [
                eval_xs_static_[..., feature_shift_configuration:],
                eval_xs_static_[..., :feature_shift_configuration],
            ],
            dim=-1,
        )

        # Extend X
        if extend_features:
            # only extend for static features, dynamic features consider to be a part of statics
            eval_xs_static_ = torch.cat(
                [
                    eval_xs_static_,
                    torch.zeros(
                        (
                            eval_xs_static_.shape[0],
                            eval_xs_static_.shape[1],
                            max_features - eval_xs_static_.shape[2] - eval_xs_dynamic_.shape[3],
                        )
                    ).to(device),
                ],
                -1,
            )

        inputs += [(eval_xs_dynamic_, eval_xs_static_)]
        labels += [eval_ys_]

    inputs_dynamic = torch.cat([input_[0] for input_ in inputs], 1)
    inputs_dynamic = torch.split(inputs_dynamic, batch_size_inference, dim=1)

    inputs_static = torch.cat([input_[1] for input_ in inputs], 1)
    inputs_static = torch.split(inputs_static, batch_size_inference, dim=1)

    labels = torch.cat(labels, 1)
    labels = torch.split(labels, batch_size_inference, dim=1)
    # print('PREPROCESSING TIME', str(time.time() - start))
    outputs = []
    start = time.time()
    for batch_input_dynamic, batch_input_static, batch_label in zip(inputs_dynamic, inputs_static, labels):
        # preprocess_transform_ = preprocess_transform if styles_configuration % 2 == 0 else 'none'
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="None of the inputs have requires_grad=True. Gradients will be None",
            )
            warnings.filterwarnings(
                "ignore",
                message="torch.cuda.amp.autocast only affects CUDA ops, but CUDA is not available.  Disabling.",
            )
            if device == "cpu":
                output_batch = checkpoint(
                    predict,
                    (batch_input_dynamic, batch_input_static),
                    batch_label,
                    style_,
                    softmax_temperature_,
                    True,
                )
            else:
                with torch.cuda.amp.autocast(enabled=fp16_inference):
                    output_batch = checkpoint(
                        predict,
                        (batch_input_dynamic, batch_input_static),
                        batch_label,
                        style_,
                        softmax_temperature_,
                        True,
                    )
        outputs += [output_batch]
    # print('MODEL INFERENCE TIME ('+str(batch_input.device)+' vs '+device+', '+str(fp16_inference)+')', str(time.time()-start))

    outputs = torch.cat(outputs, 1)
    for i, ensemble_configuration in enumerate(ensemble_configurations):
        (
            (class_shift_configuration, feature_shift_configuration),
            preprocess_transform_configuration,
            styles_configuration,
        ) = ensemble_configuration
        output_ = outputs[:, i : i + 1, :]
        output_ = torch.cat(
            [
                output_[..., class_shift_configuration:],
                output_[..., :class_shift_configuration],
            ],
            dim=-1,
        )

        # output_ = predict(eval_xs, eval_ys, style_, preprocess_transform_)
        if not average_logits and not return_logits:
            # transforms every ensemble_configuration into a probability -> equal contribution of every configuration
            output_ = torch.nn.functional.softmax(output_, dim=-1)
        output = output_ if output is None else output + output_

    output = output / len(ensemble_configurations) # type: ignore
    if average_logits and not return_logits:
        if fp16_inference:
            output = output.float()
        output = torch.nn.functional.softmax(output, dim=-1)

    output = torch.transpose(output, 0, 1)

    return output


def get_params_from_config(c):
    return {
        "max_features": c["num_features"],
        "rescale_features": c["normalize_by_used_features"],
        "normalize_to_ranking": c["normalize_to_ranking"],
        "normalize_with_sqrt": c.get("normalize_with_sqrt", False),
    }
