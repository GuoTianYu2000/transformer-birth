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

from data import DataArgs, Dataset, iterate_batches, make_dataset
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
    data_name: str = ''


if __name__ == '__main__':
    args = TrainerArgs(
           optim_args=OptimArgs(),
           data_args=DataArgs(),
           model_args=ModelArgs()
        )
    cfg = OmegaConf.merge(OmegaConf.structured(args), OmegaConf.from_cli())
    cfg.model_args.bos_num = cfg.data_args.bos_num
    ds = make_dataset(cfg)
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
    # pdb.set_trace()
    # a test batch for experimentation
    x_exp = ds.gen_batch(np.random.default_rng(0), 128)
    x_exp = x_exp[:,:ds.seq_length]


    t = time.time()
    t0 = t
    res = []
    for i, (x, y) in enumerate(iterate_batches(ds, batch_size=cfg.optim_args.batch_size, num_workers=cfg.num_data_workers)):
        dt_data = time.time() - t
        if cfg.max_iters is not None and i >= cfg.max_iters:
            sys.exit(0)
        x = torch.from_numpy(x).cuda()
        y = torch.from_numpy(y).cuda()

        optimizer.zero_grad()
        pred = model(x)

        loss = F.cross_entropy(pred.flatten(0, 1), y.flatten(0, 1))
        loss.backward()

        optimizer.step()
        dt = time.time() - t
        t = time.time()

        if i % cfg.eval_delta == 0:
            print(f"round_{i}_loss_{loss:.2}")
        if (i <= 200 and i % 10 == 0) or i in [500, 700, 1000, 2000, 4999]:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i,
            }
            torch.save(training_state, os.path.join(cfg.save_dir, f"state_{i}.pt"))
