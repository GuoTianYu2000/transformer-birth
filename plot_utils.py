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
import seaborn as sns
import matplotlib.pyplot as plt
from probe_utils import *

from omegaconf import OmegaConf
from pathlib import Path
from torch import nn, Tensor
from torch.nn import functional as F
from typing import List, Optional, Tuple
import os
# os.chdir("/data/tianyu_guo/birth")
from data import DataArgs, Dataset, iterate_batches, make_dataset
from ihead_full_model import ModelArgs, Transformer, forward_hook, test_value, test_sink


def plot_attn_weights(outputs_list, seqs, ds, seq_indices, seq_len, layer_idx, head_idx, seq_start=0, keep_label=None, ticks_size=14, titles=[], save_files_fn=[], fn=None):
    attns = outputs_list[layer_idx]['attn_weights'].detach().cpu().numpy()
    keep_label = list(range(seq_start, seq_len)) if keep_label is None else keep_label
    for idx, seq_idx in enumerate(seq_indices):
        fig, ax = plt.subplots()
        sub_seq = seqs[seq_idx, seq_start:seq_len].clone().detach().cpu().tolist()
        sub_seq = [idx if idx in ds.idxs or num in keep_label else -1 for num, idx in enumerate(sub_seq)]
        ds.itos[-1] = ''
        ds.itos[0] = '\n'
        for i in ds.idxs:
            ds.itos[i] = 't'
        text = ds.decode(sub_seq)
        # text[0] = r'<s>'
        # if seq_idx == 0:
        #     text[-3] = r"\n"
        label_text_x = text
        label_text_y = text
        mask = 1 - np.tril(np.ones_like(attns[seq_idx, head_idx, seq_start:seq_len, seq_start:seq_len]))
        # label_text = text_test
        ax = sns.heatmap(
            attns[seq_idx, head_idx, seq_start:seq_len, seq_start:seq_len], mask=mask,
            cmap="Blues", xticklabels=label_text_x, yticklabels=label_text_y,
            ax=ax, vmin=0, vmax=1, cbar=True, cbar_kws={"shrink": 1.0, "pad": 0.01, "aspect":50, "ticks": [0, 1]}
        )
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=ticks_size)
        # ax.set_title(titles[seq_idx])
        ax.tick_params(axis='y', labelsize=ticks_size, length=0)
        ax.tick_params(axis='x', labelsize=ticks_size, length=0)
        if len(save_files_fn) > 0:
            plt.savefig(os.path.join(fn, save_files_fn[idx]), bbox_inches='tight', dpi=150)
        plt.show()
