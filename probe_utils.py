import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
import os
from omegaconf import OmegaConf
from data import DataArgs, Dataset, iterate_batches, make_dataset
from ihead_full_model import ModelArgs, Transformer, forward_hook
import pickle
import numpy as np
from data import *

def plot_attns(cfg, ax, seq_idx, head_idx, layer_idx, seq_len, outputs_list, text):
    attns = outputs_list[layer_idx]['attn_weights'][seq_idx, head_idx, :seq_len, :seq_len].detach().numpy()
    # attns = scores[0, i, :20, :20]
    mask = 1 - np.tril(np.ones_like(attns)) # manually impose causal mask for better-looking plots
    if head_idx == 0:
        yticklabels = text
    else:
        yticklabels = False
    if layer_idx == cfg.model_args['n_layers'] - 1:
        xticklabels = text
    else:
        xticklabels = False
    sns.heatmap(
            attns, mask=mask,
            cmap="coolwarm", xticklabels=xticklabels, yticklabels=yticklabels,
            ax=ax,
        )
    ax.set_title(f"Layer{layer_idx} Head{head_idx}")
    return ax

def get_model_name(n_layers=1, n_heads=1, bos_num=1, train_steps=4999, delim=0, mix_p=None,):
    d_name = float_to_str(delim)
    if mix_p is not None:
        mix_p_name = float_to_str(mix_p, digits=4)
        model_name = f"model_L{n_layers}_H{n_heads}_bos{bos_num}_delim" + d_name + "_mix_p" + mix_p_name
    else:
        model_name = f"model_L{n_layers}_H{n_heads}_bos{bos_num}_delim" + d_name
    return model_name



def load_model(run_path_local="/Users/guotianyu/GitHub/birth/gens/special/dormant_copy", run_path_server="/data/tianyu_guo/birth/gens/special/dormant_copy_2", n_layers=1, n_heads=1, bos_num=1, train_steps=4999, delim=0, mix_p=None, with_data=True, model_name=None, data_path_local="/Users/guotianyu/GitHub/birth/data", data_path_server="/data/tianyu_guo/birth/data", seeds=[42, 27]):
    if model_name is None:
        model_name = get_model_name(n_layers, n_heads, bos_num, train_steps, delim, mix_p,)
    path_local = os.path.join(run_path_local, model_name, "params.yaml")
    path_server = os.path.join(run_path_server, model_name, "params.yaml")

    try:
        cfg = OmegaConf.load(path_local)
    except:
        cfg = OmegaConf.load(path_server)
    model = Transformer(cfg.model_args)
    model.eval()

    state_path_local = os.path.join(run_path_local, model_name, f"state_{train_steps}.pt")
    state_path_server = os.path.join(run_path_server, model_name, f"state_{train_steps}.pt")
    try:
        state = torch.load(state_path_local, map_location="cpu")
    except:
        state = torch.load(state_path_server, map_location="cpu")
    model.load_state_dict(state["model_state_dict"], strict=False, )
    if not with_data:
        return model, cfg
    else:
        d_name = float_to_str(delim)
        data_name = f"bos{bos_num}_d" + d_name
        try:
            if cfg.data_args.delim_num > 1:
                data_name = data_name + "_delim2"
        except:
            data_name = data_name
        data_path_local = os.path.join(data_path_local, data_name, "meta.pickle")
        data_path_server = os.path.join(data_path_server, data_name, "meta.pickle")
        try:
            with open(data_path_local, "rb") as f:
                meta_info = pickle.load(f)
        except:
            with open(data_path_server, "rb") as f:
                meta_info = pickle.load(f)

        # data_cfg = OmegaConf.structured(meta_info)


        ds = make_dataset(cfg, meta_info)
        x = ds.gen_batch(rng=np.random.default_rng(seeds), batch_size=cfg.optim_args.batch_size)
        x = torch.from_numpy(x)
        # y = torch.from_numpy(y)
        y = x[:, 1:]
        x = x[:, :-1]
        return model, cfg, x, y, ds

def move_device(outputs_list):
    for i, outputs in enumerate(outputs_list):
        for key, value in outputs.items():
            outputs[key] = value.cpu()

    return outputs_list



def get_triggers(ds, model, hook, cutoff=0.89):
    markov_tok = [i for i in ds.tok_range if i not in ds.idxs and i not in ds.bos and i != ds.delimiter]
    # and i not in (-ds.marginal).argsort()[-15:]
    x_dormant = torch.LongTensor([ds.bos + markov_tok + [i] for i in ds.tok_range]).cuda()
    _, outputs_list_dormant = model.modified_forward_with_hook(x_dormant, hook)
    outputs_list_dormant = move_device(outputs_list_dormant)
    trigger_toks = [i for i in ds.tok_range if outputs_list_dormant[0]['attn_weights'][i, 0, -1, 0]<cutoff]
    attns_to_0 = outputs_list_dormant[0]['attn_weights'][:, 0, -1, 0].detach().cpu().numpy()
    return trigger_toks, attns_to_0, markov_tok

def get_oracle_predicts(x, ds):
    B, N, V = x.shape[0], x.shape[1], ds.num_tokens
    predicts_oracle = np.zeros((B, N, V))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i, j] in ds.idxs:
                predicts_oracle[i, j, x[i, j-1]] = 1
            else:
                predicts_oracle[i, j, :] = ds.cond[x[i, j]]
    return torch.from_numpy(predicts_oracle).float()

def get_risk(probs, predicts, predict_in_logits, triggers_pos):
    if predict_in_logits:
        predicts = torch.nn.functional.softmax(predicts, dim=-1)
    loss = - torch.log(predicts)
    loss[torch.where(probs == 0)] = 0
    risk = torch.einsum("ikj,ikj->ik", probs, loss)
    risk_icl = risk[triggers_pos].mean()
    risk_markov = risk[~triggers_pos].mean()
    return risk, risk_icl, risk_markov
