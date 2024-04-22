from dataclasses import dataclass
import itertools
import logging
import random
import math
import numpy as np
import json
import pickle
import time
import torch
import sys
import yaml
import os
import pdb

from omegaconf import OmegaConf
from pathlib import Path
from torch import nn, Tensor
from torch.nn import functional as F
from typing import List, Optional, Tuple

from ihead_data import DataArgs, Dataset, iterate_batches
from ihead_full_model import ModelArgs, Transformer
from tqdm import tqdm

logging.getLogger().setLevel(logging.INFO)


@dataclass
class OptimArgs:
    learning_rate: float = 0.01
    weight_decay: float = 0.01
    momentum: float = 0.9  # for SGD
    batch_size: int = 64
    use_sgd: bool = False  # otherwise use AdamW


@dataclass
class TrainerArgs:
    optim_args: OptimArgs
    data_args: DataArgs
    model_args: ModelArgs
    max_iters: Optional[int] = None
    eval_delta: int = 5
    log_norms: bool = False
    log_probes: bool = False
    num_data_workers: int = 60
    save_dir: Optional[str] = None
    root_dir: str = ''


if __name__ == '__main__':
    args = TrainerArgs(
           optim_args=OptimArgs(),
           data_args=DataArgs(),
           model_args=ModelArgs()
        )
    cfg = OmegaConf.merge(OmegaConf.structured(args), OmegaConf.from_cli())
    cfg.model_args.bos_num = cfg.data_args.bos_num
    ds = Dataset(cfg.data_args, train_test=None)
    ds_test = Dataset(cfg.data_args, train_test=None)
    ds_test.idxs = ds.idxs

    cfg.model_args.vocab_size = ds.num_tokens

    if cfg.save_dir is not None:
        outdir = Path(cfg.root_dir) / Path(cfg.save_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        # save params
        dict_cfg = dict(cfg)
        print(dict_cfg)
        with open(outdir / 'params.yaml', 'w') as f:
            OmegaConf.save(dict_cfg, f,)
        outfile = open(outdir / 'res.jsonl', 'w')

    model = Transformer(cfg.model_args)
    model.cuda()

    # memory probes
    range_toks = torch.from_numpy(np.arange(ds.n_train_toks)).cuda()
    def test_wo1():
        toks = model.tok_embeddings(range_toks)
        toks = model.layers[1].attention.wv(toks)
        toks = model.layers[1].attention.wo(toks)
        toks = model.output(toks)
        return (toks.argmax(-1) == range_toks).float().mean().item()

    range_pos_toks = torch.from_numpy(np.arange(cfg.model_args.max_length)).cuda()
    def test_wk0(cutoff=None):
        if cfg.model_args.sin_cos:
            pe = model.pe[:cutoff,:]
        else:
            pe = model.pos_embeddings.weight[:cutoff,:]
        k = model.layers[0].attention.wk(pe[:-1])
        q = model.layers[0].attention.wq(pe[1:])
        return ((q @ k.t()).argmax(-1) == range_pos_toks[:pe.shape[0]-1]).float().mean().item()

    full_range_toks = torch.from_numpy(np.arange(ds.num_tokens)).cuda()
    wk1_range_toks = full_range_toks.clone()
    if cfg.data_args.fixed_special_toks:
        wk1_range_toks = wk1_range_toks[ds.idxs]
    def test_wk1():
        toksk = model.tok_embeddings(wk1_range_toks)
        toksk = model.layers[0].attention.wv(toksk)
        toksk = model.layers[0].attention.wo(toksk)
        toksk = model.layers[1].attention.wk(toksk)

        toksq = model.tok_embeddings(wk1_range_toks)
        toksq = model.layers[1].attention.wq(toksq)
        return ((toksq @ toksk.t()).argmax(-1) == range_toks[:wk1_range_toks.shape[0]]).float().mean().item()

    full_range_toks = torch.from_numpy(np.arange(ds.num_tokens)).cuda()
    conds = torch.from_numpy(np.array(ds.cond)).cuda()
    used_idxs = np.arange(ds.num_tokens)
    if cfg.data_args.fixed_special_toks:
        used_idxs = np.setdiff1d(used_idxs, ds.idxs)
    def test_ff1():
        toks = model.tok_embeddings(full_range_toks[used_idxs])
        toks = model.layers[1].ff(toks)
        toks = model.output(toks)
        return F.kl_div(F.log_softmax(toks, dim=1), conds[used_idxs], reduction='batchmean').item()


    # optim
    if cfg.optim_args.use_sgd:
        optimizer = torch.optim.SGD(model.parameters(),
                lr=cfg.optim_args.learning_rate,
                weight_decay=cfg.optim_args.weight_decay,
                momentum=cfg.optim_args.momentum)
    else:
        optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=cfg.optim_args.learning_rate,
                weight_decay=cfg.optim_args.weight_decay,
                betas=(0.9, 0.99),
                eps=1e-8)

    def predict(text):
        toks = text.split()
        x = torch.from_numpy(np.array(tokenizer.encode(toks))).cuda().unsqueeze(0)
        return tokenizer.decode([model(x)[0,-1].argmax(-1)])[0]

    # pdb.set_trace()
    # a test batch for experimentation
    x_exp, out_exp = ds.gen_batch(np.random.default_rng(0), 128)
    x_exp = x_exp[:,:ds.seq_length]

    x_test, out_test = ds_test.gen_batch(np.random.default_rng(0), 512)
    x_t = torch.from_numpy(x_test[:,:ds.seq_length]).cuda()
    y_t = torch.from_numpy(x_test[:,1:ds.seq_length + 1]).cuda()
    outs_t = torch.from_numpy(out_test[:,:ds.seq_length]).cuda()

    t = time.time()
    t0 = t
    res = []
    for i, (x, y, outs) in enumerate(iterate_batches(ds, batch_size=cfg.optim_args.batch_size,
                                                     num_workers=cfg.num_data_workers)):
        dt_data = time.time() - t
        if cfg.max_iters is not None and i >= cfg.max_iters:
            if cfg.save_dir is not None:
                outfile.close()
                xt = torch.from_numpy(x_exp).cuda()
                outt = torch.from_numpy(out_exp).cuda()
                scores = []
                for layer in range(model.n_layers):
                    scores.append(model.get_layer_scores(xt, n=layer).detach().cpu().numpy())
                preds = model(xt).detach().cpu().numpy()
                pickle.dump({'x': x_exp, 'out': out_exp, 'scores': scores, 'preds': preds}, open(outdir / 'exp.pkl', 'wb'))
            sys.exit(0)
        x = torch.from_numpy(x).cuda()
        y = torch.from_numpy(y).cuda()
        outs = torch.from_numpy(outs).cuda()

        optimizer.zero_grad()
        pred = model(x)

        loss = F.cross_entropy(pred.flatten(0, 1), y.flatten(0, 1))
        loss.backward()

        optimizer.step()
        dt = time.time() - t
        t = time.time()

        if i % cfg.eval_delta == 0:
            print(f"round_{i}_loss_{loss:.2}")
        if i in [0, 50, 100, 500, 1000, 2000, 5000-1]:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i,
            }
            torch.save(training_state, os.path.join(cfg.save_dir, f"state_{i}.pt"))
