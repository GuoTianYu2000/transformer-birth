from dataclasses import dataclass
import itertools
import logging
import random
import math
import numpy as np
import pickle
import time
import torch
import sys
import pdb

from torch import nn, Tensor
from torch.nn import functional as F
from typing import List, Optional, Tuple


class forward_hook():
    def __init__(self, target_layers, target_name, ) -> None:
        self.target_layers = target_layers
        self.target_name = target_name

    def __call__(self, layer_idx, name, func, *args):
        if self.target_layers is not None and layer_idx in self.target_layers and name in self.target_name:
            # no rope
            return self.intervention(func, layer_idx, name, *args)
        else:
            # not zero-out layers
            if func is None:
                return args[0]
            else:
                return func(*args)
        
    def intervention(self, func, *args):
        return args[0]

class test_value(forward_hook):
    def __init__(self, target_layers) -> None:
        super().__init__(target_layers, target_name = ["attn_weights"], )
    def intervention(self, func, layer_idx, name, *args):
        attns = func(*args)
        B, H, N, N = attns.shape
        assert B <= N
        for i in range(B):
            attns[i, 0, -1, :] = 0
            attns[i, 0, -1, i] = 1
        return attns

class test_sink(forward_hook):
    def __init__(self, target_layers) -> None:
        super().__init__(target_layers, target_name = ["attn_weights"], )
    def intervention(self, func, layer_idx, name, *args):
        attns = func(*args)
        B, H, N, N = attns.shape
        for i in range(B):
            attns[i, 0, -1, :] = 0
            attns[i, 0, -1, 0] = 1
        return attns

class clean_attention(forward_hook):
    def __init__(self, target_layers, dormant_pos) -> None:
        super().__init__(target_layers, target_name = ["attn_weights"], )
        self.dormant_pos = dormant_pos
    def intervention(self, func, layer_idx, name, *args):
        attns = func(*args)
        B, H, N, N = attns.shape
        attns[self.dormant_pos[0], 0, self.dormant_pos[1], :] = 0
        attns[self.dormant_pos[0], 0, self.dormant_pos[1], 0] = 1
        return attns

class zero_out_attention(forward_hook):
    def __init__(self, target_layers, target_heads) -> None:
        super().__init__(target_layers, target_name = ["attn_weights"], )
        self.target_heads = target_heads
    def intervention(self, func, layer_idx, name, *args):
        attns = func(*args)
        zero_out_indices = []
        for (l, h) in self.target_heads:
            if l == layer_idx:
                zero_out_indices.append(h)
        attns[:, zero_out_indices, :, :] = 0
        return attns

@dataclass
class ModelArgs:
    vocab_size: int = -1  # defined later
    n_layers: int = 2
    dim: int = 128
    n_heads: int = 4
    max_length: int = 256
    pre_norm: bool = True
    no_ffn: bool = False
    no_norm: bool = False
    linear_ffn: bool = False
    no_first_layer_ffn: bool = False
    freeze_embeddings: bool = False
    freeze_output: bool = False
    tie_output: bool = False
    freeze_wv: bool = False
    freeze_wo: bool = False
    no_wo: bool = False
    no_wv: bool = False
    sin_cos: bool = False
    bos_num: int = 0


class Attention(nn.Module):
    def __init__(self,
                 dim: int,
                 head_dim: int,
                 n_heads: int,
                 freeze_wv: bool = False,
                 freeze_wo: bool = False,
                 no_wv: bool = False,
                 no_wo: bool = False):
        super().__init__()
        self.head_dim = head_dim
        self.n_heads = n_heads

        self.wq = nn.Linear(dim, n_heads*head_dim, bias=False)
        self.wk = nn.Linear(dim, n_heads*head_dim, bias=False)
        if no_wv:
            self.wv = nn.Identity()
        else:
            self.wv = nn.Linear(dim, n_heads*head_dim, bias=False)
            if freeze_wv:
                self.wv.weight.requires_grad_(False)

        if no_wo:
            self.wo = nn.Identity()
        else:
            self.wo = nn.Linear(n_heads*head_dim, dim, bias=False)
            if freeze_wo:
                self.wo.weight.requires_grad_(False)

    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor):
        bs, slen, _ = x.shape
        assert mask is not None

        xq = self.wq(x).view(bs, slen, self.n_heads, self.head_dim)
        xk = self.wk(x).view(bs, slen, self.n_heads, self.head_dim)
        xv = self.wv(x).view(bs, slen, self.n_heads, self.head_dim)

        # change to (bs, n_heads, slen, head_dim)
        xq, xk, xv = xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2)
        # pdb.set_trace()
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        scores = scores + mask  # (bs, n_heads, slen, slen)
        scores = F.softmax(scores.float(), dim=-1).type_as(x)
        output = torch.matmul(scores, xv)  # (bs, n_heads, slen, head_dim)
        output = output.transpose(1, 2)  # (bs, slen, n_heads, head_dim)

        output = output.reshape(bs, slen, -1)
        return self.wo(output), scores
    
    def modified_forward(self,
                layer_idx: int,
                hook: forward_hook,
                x: torch.Tensor,
                mask: torch.Tensor):
        attn_outputs = {}
        bs, slen, _ = x.shape
        assert mask is not None

        x = hook(layer_idx, "attn_input", None, x)
        attn_outputs["attn_input"] = x
        func_wq = lambda x: self.wq(x).view(bs, slen, self.n_heads, self.head_dim).transpose(1, 2)
        xq = hook(layer_idx, "query_states", func_wq, x)
        attn_outputs["query_states"] = xq
        func_wk = lambda x: self.wk(x).view(bs, slen, self.n_heads, self.head_dim).transpose(1, 2)
        xk = hook(layer_idx, "key_states", func_wk, x)
        attn_outputs["key_states"] = xk
        func_wv = lambda x: self.wv(x).view(bs, slen, self.n_heads, self.head_dim).transpose(1, 2)
        xv = hook(layer_idx, "value_states", func_wv, x)
        attn_outputs["value_states"] = xv

        func_scores = lambda xq, xk: torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim) + mask
        scores = hook(layer_idx, "attn_logits", func_scores, xq, xk)
        attn_outputs["attn_logits"] = scores


        func_weights = lambda scores: F.softmax(scores.float(), dim=-1).type_as(x)
        scores = hook(layer_idx, "attn_weights", func_weights, scores)
        attn_outputs["attn_weights"] = scores

        func_attns = lambda scores, xv: torch.matmul(scores, xv).transpose(1, 2) # (bs, slen, n_heads, head_dim)
        output = hook(layer_idx, "attn_output_per_head", func_attns, scores, xv)
        attn_outputs["attn_output_per_head"] = output

        func_output = lambda output: self.wo(output.reshape(bs, slen, -1))
        output = hook(layer_idx, "attn_output_proj", func_output, output)
        attn_outputs["attn_output_proj"] = output

        output = hook(layer_idx, "attn_output", None, output)
        attn_outputs["attn_output"] = output
        return output, scores, attn_outputs


class FeedForward(nn.Module):
    def __init__(self,
                 dim: int,
                 hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        h = self.w1(x)
        h = F.relu(h.float()).type_as(x)
        return self.w2(h)


class TransformerBlock(nn.Module):
    def __init__(self,
                 dim: int,
                 hidden_dim: int,
                 n_heads: int,
                 pre_norm: bool,
                 no_norm: bool = False,
                 no_ffn: bool = False,
                 linear_ffn: bool = False,
                 freeze_wv: bool = False,
                 freeze_wo: bool = False,
                 no_wv: bool = False,
                 no_wo: bool = False,
                ):
        super().__init__()
        assert dim % n_heads == 0
        head_dim = dim // n_heads
        self.attention = Attention(
                dim=dim,
                head_dim=head_dim,
                n_heads=n_heads,
                freeze_wv=freeze_wv,
                freeze_wo=freeze_wo,
                no_wv=no_wv,
                no_wo=no_wo)
        if not no_ffn:
            if linear_ffn:
                self.ff = nn.Linear(dim, dim, bias=False)
            else:
                self.ff = FeedForward(dim=dim, hidden_dim=hidden_dim)
        if no_norm:
            self.attention_norm = nn.Identity()
            self.ff_norm = nn.Identity()
        else:
            self.attention_norm = nn.LayerNorm(dim, eps=1e-5)
            self.ff_norm = nn.LayerNorm(dim, eps=1e-5)
        self.pre_norm = pre_norm
        self.no_ffn = no_ffn

    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor,
                return_scores: bool = False,
                no_ffn: bool = False):
        no_ffn = no_ffn or self.no_ffn
        if self.pre_norm:
            h, scores = self.attention(self.attention_norm(x), mask)
            if return_scores:
                return scores
            h = x + h
            res = h
            if no_ffn:
                return h
            else:
                return res + self.ff(self.ff_norm(h))
        else:
            h, scores = self.attention(x, mask)
            if return_scores:
                return scores
            h = self.attention_norm(x + h)
            if no_ffn:
                return h
            else:
                return self.ff_norm(h + self.ff(h))
            
    def modified_forward(self,
                layer_idx: int,
                hook: forward_hook,
                x: torch.Tensor,
                mask: torch.Tensor,
                no_ffn: bool = False):
        outputs_layer = {}
        no_ffn = no_ffn or self.no_ffn
        h = hook(layer_idx, "input", None, x)
        outputs_layer["input"] = h
        if self.pre_norm:
            h = hook(layer_idx, "input_norm", self.attention_norm, h)
            outputs_layer["input_norm"] = h
            h, scores, attn_outputs = self.attention.modified_forward(layer_idx, hook, h, mask)
            outputs_layer.update(attn_outputs)
            func_add_res = lambda x, h: x + h
            h = hook(layer_idx, "attn_output_add_res", func_add_res, x, h)
            outputs_layer["attn_output_add_res"] = h
            if no_ffn:
                h = hook(layer_idx, "no_ffn", None, h)
                outputs_layer["no_ffn"] = h
            else:
                h = hook(layer_idx, "mlp_input_pre_norm", None, h)
                outputs_layer["mlp_input_pre_norm"] = h
                res = h
                h = hook(layer_idx, "mlp_input_norm", self.ff_norm, h)
                outputs_layer["mlp_input_norm"] = h
                h = hook(layer_idx, "mlp_input", None, h)
                outputs_layer["mlp_input"] = h
                h = hook(layer_idx, "mlp_output", self.ff, h)
                outputs_layer["mlp_output"] = h
                h = hook(layer_idx, "mlp_output_add_res", func_add_res, res, h)
                outputs_layer["mlp_output_add_res"] = h
            h = hook(layer_idx, "output", None, h)
            outputs_layer["output"] = h
            return h, outputs_layer
        else:
            h, scores, attn_outputs = self.attention.modified_forward(layer_idx, hook, h, mask)
            outputs_layer.update(attn_outputs)
            func_add_res = lambda x, h: x + h
            h = hook(layer_idx, "attn_output_add_res", func_add_res, x, h)
            outputs_layer["attn_output_add_res"] = h
            if no_ffn:
                h = hook(layer_idx, "no_ffn", None, h)
                outputs_layer["no_ffn"] = h
            else:
                h = hook(layer_idx, "mlp_input_pre_norm", None, h)
                outputs_layer["mlp_input_pre_norm"] = h
                res = h
                h = hook(layer_idx, "mlp_input_norm", self.attention_norm, h)
                outputs_layer["mlp_input_norm"] = h
                h = hook(layer_idx, "mlp_input", None, h)
                outputs_layer["mlp_input"] = h
                h = hook(layer_idx, "mlp_output", self.ff, h)
                outputs_layer["mlp_output"] = h
                h = hook(layer_idx, "mlp_output_add_res", func_add_res, res, h)
                outputs_layer["mlp_output_add_res"] = h
            h = hook(layer_idx, "output", None, h)
            outputs_layer["output"] = h
            h = hook(layer_idx, "output_norm", self.ff_norm, h)
            outputs_layer["output_norm"] = h
            return h, outputs_layer

class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tie_output = args.tie_output
        self.dim = args.dim
        self.sin_cos = args.sin_cos

        # embeddings
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.pos_embeddings = nn.Embedding(args.max_length, args.dim)
        if args.freeze_embeddings:
            self.tok_embeddings.weight.requires_grad_(False)
            self.pos_embeddings.weight.requires_grad_(False)

        # sin/cos position embeddings
        if self.sin_cos:
            position = torch.arange(args.max_length).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, args.dim, 2) * (-math.log(10000.0) / args.dim))
            pe = torch.zeros(args.max_length, args.dim)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe)

        self.layers = nn.ModuleList([TransformerBlock(
            dim=args.dim,
            hidden_dim=4*args.dim,
            n_heads=args.n_heads,
            pre_norm=args.pre_norm,
            no_norm=args.no_norm,
            no_ffn=args.no_ffn or (i == 0 and args.no_first_layer_ffn),
            linear_ffn=args.linear_ffn,
            freeze_wv=args.freeze_wv,
            freeze_wo=args.freeze_wo,
            no_wv=args.no_wv,
            no_wo=args.no_wo,
            ) for i in range(args.n_layers)])

        # final normalization layer (only needed for pre-norm)
        self.norm: Optional[nn.Module] = None
        if args.pre_norm:
            if args.no_norm:
                self.norm = nn.Identity()
            else:
                self.norm = nn.LayerNorm(args.dim, eps=1e-5)

        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        if args.freeze_output:
            self.output.weight.requires_grad_(False)
        elif args.tie_output:
            # self.tok_embeddings.weight.data /= math.sqrt(args.dim)
            self.output.weight = self.tok_embeddings.weight

    def apply_pe(self, h, N, ):
        if self.sin_cos:
            h = h + self.pe.unsqueeze(0)
        else:
            h = h + self.pos_embeddings(torch.arange(N, device=h.device).view(1, N))
        return h


    def forward(self, tokens: torch.Tensor, return_layer: Optional[int] = None, before_ffn: bool = False):
        B, N = tokens.shape

        # embedding layer
        h = self.tok_embeddings(tokens)

        if self.sin_cos:
            h = h + self.pe.unsqueeze(0)
        else:
            h = h + self.pos_embeddings(torch.arange(N, device=tokens.device).view(1, N))

        if return_layer == 0:
            return h

        # causal mask
        mask = torch.full((1, 1, N, N), float('-inf'), device=tokens.device)
        mask = torch.triu(mask, diagonal=1).type_as(h)

        # transformer blocks
        for i, layer in enumerate(self.layers):
            if return_layer == i + 1:
                return layer(h, mask, no_ffn=before_ffn)
            h = layer(h, mask)
            # pdb.set_trace()

        # output layer
        if self.norm is not None:
            h = self.norm(h)
        output = self.output(h)
        return output.float()

    def modified_forward_with_hook(self, tokens: torch.Tensor, hook: forward_hook):
        outputs_list = []
        B, N = tokens.shape

        # embedding layer
        outputs_layer = {}
        h = hook(0, "embed", self.tok_embeddings, tokens)
        outputs_layer["embed"] = h
        # pdb.set_trace()
        # TODO: this may fail
        h = hook(0, "embed_post_pos", self.apply_pe, h, N)
        outputs_layer["embed_post_pos"] = h
        outputs_list.append(outputs_layer)

        # causal mask
        mask = torch.full((1, 1, N, N), float('-inf'), device=tokens.device)
        mask = torch.triu(mask, diagonal=1).type_as(h)

        # transformer blocks
        for i, layer in enumerate(self.layers):
            h, outputs_layer = layer.modified_forward(i, hook, h, mask)
            outputs_list[-1].update(outputs_layer)
            outputs_list.append({})
            # pdb.set_trace()

        # output layer
        h = hook(i, "output_last_layer_norm", self.norm, h)
        outputs_list[-1]["output_last_layer_norm"] = h
        output = hook(i, "output_last_layer_pred", self.output, h)
        outputs_list[-1]["output_last_layer_pred"] = output
        return output, outputs_list

    def forward_ff_only(self, tokens: torch.Tensor):
        B, N = tokens.shape

        # embedding layer
        h = self.tok_embeddings(tokens)
        if self.sin_cos:
            h = h + self.pe.unsqueeze(0)
        else:
            h = h + self.pos_embeddings(torch.arange(N, device=tokens.device).view(1, N))

        # transformer blocks
        for i, layer in enumerate(self.layers):
            h = h + layer.ff(h)

        # output layer
        if self.norm is not None:
            h = self.norm(h)
        output = self.output(h)
        if self.tie_output:
            output /= math.sqrt(self.dim)
        return output.float()

    def get_layer_scores(self, tokens: torch.Tensor, n: int = 0):
        assert n < len(self.layers)
        B, N = tokens.shape

        # embedding layer
        h = self.tok_embeddings(tokens)

        if self.sin_cos:
            h = h + self.pe.unsqueeze(0)
        else:
            h = h + self.pos_embeddings(torch.arange(N, device=tokens.device).view(1, N))
        

        # causal mask
        mask = torch.full((1, 1, N, N), float('-inf'), device=tokens.device)
        mask = torch.triu(mask, diagonal=1).type_as(h)

        # transformer blocks
        for i, layer in enumerate(self.layers):
            if i == n:
                return layer(h, mask, return_scores=True)
            else:
                h = layer(h, mask)

    def get_top_preds(self, tokens: torch.Tensor, n: int = 4):
        squeeze = False
        if len(tokens.shape) == 1:
            squeeze = True
            tokens = tokens.unsqueeze(0)
        with torch.no_grad():
            preds = self(tokens).detach()
        vals, idxs = preds.sort(-1, descending=True)
        vals = vals[:,:,:n]
        idxs = idxs[:,:,:n]
        if squeeze:
            return vals.squeeze(0), idxs.squeeze(0)
        return vals, idxs
