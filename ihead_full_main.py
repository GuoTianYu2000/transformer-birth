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
import wandb

from omegaconf import OmegaConf
from pathlib import Path
from torch import nn, Tensor
from torch.nn import functional as F
from typing import List, Optional, Tuple

from data import DataArgs, Dataset, iterate_batches, make_dataset
from ihead_full_model import ModelArgs, Transformer
from tqdm import tqdm

logging.getLogger().setLevel(logging.INFO)

def set_random_seed(seed):
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed) # RZ: Sets the seed for generating random numbers for all devices (both CPU and CUDA).
    torch.manual_seed(seed) # RZ: Specifically sets the seed for all CUDA GPUs for generating random numbers. This is necessary for ensuring that all GPUs have the same initial random state.
    random.seed(seed)
    torch.backends.cudnn.deterministic = True # RZ: Some operations in CUDA are non-deterministic by default. To ensure determinism, you might need to set additional flags in PyTorch. However, this could potentially impact performance.
    torch.backends.cudnn.benchmark = False


@dataclass
class OptimArgs:
    learning_rate: float = 0.01
    weight_decay: float = 0.01
    momentum: float = 0.9  # for SGD
    batch_size: int = 64
    use_sgd: bool = False  # otherwise use AdamW

@dataclass
class WandbArgs:
    project: str = 'birth'
    entity: str = 'tianyu_guo'
    name: str = 'dormant_test'

@dataclass
class TrainerArgs:
    optim_args: OptimArgs
    data_args: DataArgs
    model_args: ModelArgs
    wandb_args: WandbArgs
    max_iters: Optional[int] = None
    eval_delta: int = 5
    log_norms: bool = False
    log_probes: bool = False
    num_data_workers: int = 60
    save_dir: Optional[str] = None
    fine_grid_log: int = 0
    root_dir: str = ''
    task_name: str = ''
    seperate_loss: bool = False
    seed: int = 42


if __name__ == '__main__':
    args = TrainerArgs(
           optim_args=OptimArgs(),
           data_args=DataArgs(),
           model_args=ModelArgs(),
           wandb_args=WandbArgs(),
        )
    cfg = OmegaConf.merge(OmegaConf.structured(args), OmegaConf.from_cli())
    cfg.model_args.bos_num = cfg.data_args.bos_num
    set_random_seed(cfg.seed)
    if cfg.data_args.delimiter_p > 0:
        meta_path = "/data/tianyu_guo/birth/data/bos1_d005/meta.pickle"
    else:
        meta_path = "/data/tianyu_guo/birth/data/bos1_d0/meta.pickle"
    with open(meta_path, "rb") as f:
        meta_info = pickle.load(f)
    # pdb.set_trace()
    ds = make_dataset(cfg, meta_info)
    cfg.model_args.vocab_size = ds.num_tokens

    if cfg.save_dir is not None:
        outdir = Path(cfg.root_dir) / Path(cfg.save_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        # save params
        dict_cfg = dict(cfg)
        print(dict_cfg)
        with open(outdir / 'params.yaml', 'w') as f:
            OmegaConf.save(dict_cfg, f,)
    # wandb.init()
    wandb.init(
            dir=str(outdir),
            project=cfg.wandb_args.project,
            entity=cfg.wandb_args.entity,
            name=cfg.wandb_args.name,
        )
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
    if cfg.fine_grid_log == 0:
        log_steps = np.linspace(cfg.fine_grid_log, cfg.max_iters, 5).tolist()
    elif cfg.fine_grid_log < cfg.max_iters:
        log_steps = np.arange(0, cfg.fine_grid_log, 5).tolist() + np.linspace(cfg.fine_grid_log, cfg.max_iters, 5).tolist()
    else:
        log_steps = np.arange(0, cfg.max_iters, 20).tolist()
    for i, (x, y) in enumerate(iterate_batches(ds, batch_size=cfg.optim_args.batch_size, num_workers=cfg.num_data_workers)):
        if i in log_steps:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i,
            }
            torch.save(training_state, os.path.join(cfg.save_dir, f"state_{i}.pt"))
        dt_data = time.time() - t
        if cfg.max_iters is not None and i >= cfg.max_iters:
            sys.exit(0)
        triggers_pos = ds.get_triggers_pos(x)
        x = torch.from_numpy(x).cuda()
        y = torch.from_numpy(y).cuda()
        triggers_pos = torch.from_numpy(triggers_pos).cuda()
        exist_icl = torch.sum(triggers_pos) > 0

        optimizer.zero_grad()
        pred = model(x)

        loss = F.cross_entropy(pred.flatten(0, 1), y.flatten(0, 1))
        loss.backward()

        optimizer.step()
        dt = time.time() - t
        t = time.time()
        if cfg.seperate_loss:
            # pdb.set_trace()
            icl_loss = F.cross_entropy(pred[triggers_pos, :], y[triggers_pos]) if exist_icl else 0
            markov_loss = F.cross_entropy(pred[~triggers_pos, :], y[~triggers_pos])
            wandb.log({
                            f"{cfg.task_name}/overall_loss": loss,
                            f"{cfg.task_name}/markov_loss": markov_loss,
                            f"{cfg.task_name}/icl_loss": icl_loss,
                        },
                        step=i,
                    )
        else:
            wandb.log({f"{cfg.task_name}/overall_loss": loss}, step=i)

    # save the last state
    training_state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_step": i+1,
    }
    torch.save(training_state, os.path.join(cfg.save_dir, f"state_{i+1}.pt"))


