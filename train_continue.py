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

from data import *
from ihead_full_model import *
from tqdm import tqdm
from probe_utils import get_model_name, ModelLoader
from ihead_full_main import *

if __name__ == '__main__':
    torch.cuda.set_device(5)
    device = 'cuda:5'
    run_path_server = "/data/tianyu/birth/gens/pre-iclr/dynamics/dormant_copy_long_train_redo"
    init_step = 100000

    model_loader = ModelLoader(run_path_local=None, run_path_server=run_path_server, bos_num=1, train_steps=init_step, delim=0, n_layers=1, n_heads=1, no_attn_norm=(), no_ffn_norm=(), no_attn=( ), no_ffn=(), linear_ffn=(), lr=0.0003, use_simple_model=False, use_vo=True, use_read_out=False, seed=42, with_data=True, with_optim=True, data_path_local=None, data_path_server="/data/tianyu/birth/data", device=device)
    model_loader.change_steps(init_step)
    model, cfg, x, y, ds, optimizer = model_loader(with_data=True)
    idxs_in_torch = torch.Tensor(ds.idxs).to(device)
    x_exp = ds.gen_batch(np.random.default_rng(0), 128)
    x_exp = x_exp[:,:ds.seq_length]
    cfg.max_iters = 500000


    log_steps = np.arange(init_step, init_step+1000, 200).tolist()
    log_steps = log_steps + np.arange(init_step+1000, cfg.max_iters+1, 200).tolist()
    
    for i, (x, y) in enumerate(iterate_batches(ds, batch_size=cfg.optim_args.batch_size, num_workers=cfg.num_data_workers)):
        if init_step+i in log_steps:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": init_step+i,
            }
            torch.save(training_state, os.path.join(cfg.save_dir, f"attn_state_{init_step+i}.pt"))
        if cfg.max_iters is not None and i >= cfg.max_iters:
            sys.exit(0)
        x = torch.from_numpy(x).cuda()
        y = torch.from_numpy(y).cuda()
        triggers_pos = torch.isin(x, idxs_in_torch)

        optimizer.zero_grad()
        pred = model(x)

        loss = F.cross_entropy(pred.flatten(0, 1), y.flatten(0, 1))
        loss.backward()

        optimizer.step()

    # save the last state
    training_state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_step": init_step+i+1,
    }
    torch.save(training_state, os.path.join(cfg.save_dir, f"attn_state_{init_step+i+1}.pt"))
