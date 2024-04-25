from dataclasses import dataclass
import itertools
import logging
import random
import math
import numpy as np
import pickle
import time
import sys
import pdb

from typing import List, Optional, Tuple

logging.getLogger().setLevel(logging.INFO)


@dataclass
class DataArgs:
    k: int = 0
    seq_length: int = 256
    show_latents: bool = False
    fixed_special_toks: bool = False
    special_toks_offset: int = 0
    output_counter: bool = True
    no_repeat: bool = False
    bos_num: int = 1
    delimiter_p: float = 0.05


class Dataset:
    def __init__(self, args: DataArgs,
                 train_test: Optional[str] = None,
                 bigram_outs: Optional[bool] = False):
        self.k = args.k
        self.seq_length = args.seq_length
        self.bos_num = args.bos_num
        self.show_latents = args.show_latents
        self.train_test = train_test
        self.output_counter = args.output_counter
        self.no_repeat = args.no_repeat
        self.bigram_outs = bigram_outs
        self.delimiter_p = args.delimiter_p

        # init distributions
        self.meta = pickle.load(open('data/meta.pkl', 'rb'))
        self.meta = self.add_bos(self.meta)
        self.meta = self.add_delimiter(self.meta)
        self.itos = self.meta['itos']
        self.stoi = self.meta['stoi']
        self.num_tokens = self.meta['vocab_size']
        self.tok_range = list(np.arange(self.num_tokens))

        # OOD
        if self.train_test is not None:
            assert not self.bigram_outs  # this requires distributions over all tokens
            self.n_train_toks = int(0.75 * self.num_tokens)
        else:
            self.n_train_toks = self.num_tokens

        # marginal probabilities over characters
        self.marginal = np.zeros(self.num_tokens)
        for k, cnt in self.meta['unigrams'].items():
            self.marginal[self.stoi[k]] = cnt
        self.marginal /= self.marginal.sum()

        # conditionals
        self.cond = [np.zeros(self.num_tokens) for _ in range(self.num_tokens)]
        for (w1, w2), cnt in self.meta['bigrams'].items():
            self.cond[self.stoi[w1]][self.stoi[w2]] += cnt
        for i in range(self.num_tokens):
            self.cond[i] /= self.cond[i].sum()

        # special tokens
        self.idxs = None
        if args.fixed_special_toks:
            # use unigram marginals
            self.idxs = list(self.marginal.argsort()[self.num_tokens-args.special_toks_offset-self.k:self.num_tokens-args.special_toks_offset])

    def decode(self, idxs: List[int]) -> str:
        text = [self.itos[idx] for idx in idxs]
        return text

    def icl_transition(self, x, rng, contexts):
        if x in contexts.keys():
            return contexts[x]
        else:
            return None
    
    def markov_transition(self, x, rng):
        probs = self.cond[x]
        return rng.choice(self.tok_range, p=probs)

    # start and end are indices of the start and the end of the copy
    def copy(self, x, rng, seq, start, end):
        return seq[start:(end+1)]

    def iid_transition(self, x, rng, ):
        return rng.choice(self.tok_range, p=self.marginal, size=self.k, replace=False)
    
    def make_icl_context(self, triggers, rng):
        if self.no_repeat:  # prevent next token to be same as idx
            pools = [self.tok_range.copy() for idx in triggers]
            for i, idx in enumerate(triggers):
                pools[i].remove(idx)
        else:
            pools = [self.tok_range for idx in triggers]
        # outs = [rng.choice(self.tok_range) for idx in idxs]
        if self.bigram_outs:
            outs = [rng.choice(pool, p=(self.cond[idx][pool] / self.cond[idx][pool].sum())) for pool, idx in zip(pools, triggers)]
        else:
            outs = [rng.choice(pool) for pool in pools]
        context = [(t, o) for t, o in zip(triggers, outs)]
        context = dict(context)
        return context
    
    def bos_init(self, ):
        seq = []
        for idx in range(self.num_tokens):
            if self.itos[idx] == '<s>':
                seq.append(idx)
        return seq
    
    def gen_seq(self, rng: np.random.Generator):
        contexts = self.make_icl_context(self.idxs, rng)
        seq = self.bos_init()
        while len(seq) < self.seq_length:
            x = seq[-1]
            x_markov, x_icl = self.markov_transition(x, rng), self.icl_transition(x, rng, contexts)
            if x_icl is None:
                seq.append(x_markov)
            else:
                seq.append(x_icl)


    def gen_seqs(self, rng: np.random.Generator):
        while True:
            seq, outputs_seq = self.gen_seq(rng)
            yield (seq, outputs_seq)

    def gen_batch(self, rng: np.random.Generator, batch_size: int):
        seqs = []
        outs = []
        for _ in range(batch_size):
            seq, out = self.gen_seq(rng)
            seqs += seq
            outs += out
        x = np.array(seqs).reshape(batch_size, self.seq_length + 1)
        outs = np.array(outs).reshape(batch_size, self.seq_length + 1)
        return x, outs
    
    # here to make sure that <s> generates a new token following unigrams, and nothing generates <s>, nor <s> gets included in unigrams
    def add_bos(self, meta):
        for i in range(self.bos_num):
            idx = meta['vocab_size']
            tok = f'<s>'
            ref_pre = [(tok, 0) for tok in meta['unigrams'].keys()]
            ref_pre = dict(ref_pre)
            ref_post = meta['unigrams']
            ref_post[tok], ref_pre[tok] = 0, 0
            meta = self.update_meta(meta, idx, tok, ref_pre=ref_pre, ref_post=ref_post)
        return meta
    
    # here to make sure that 1. <d> is not in unigrams; 2. <d> generates a new token following unigrams; 3. all other tokens (except for <s>) generate <d> with probability p
    def add_delimiter(self, meta):
        idx = meta['vocab_size']
        tok = '<d>'
        ref_pre = [(tok, 0) for tok in meta['unigrams'].keys()]
        ref_pre = dict(ref_pre)
        for (w1, w2), cnt in self.meta['bigrams'].items():
            if w1 == '<s>':
                continue
            ref_pre[w1] += cnt
        for (w1, cnt) in ref_pre.items():
            ref_pre[w1] = cnt * self.delimiter_p / (1- self.delimiter_p)
        ref_post = meta['unigrams']
        ref_post[tok], ref_pre[tok] = 0, 0
        meta = self.update_meta(meta, idx, tok, ref_pre=ref_pre, ref_post=ref_post)
        return meta
    
    def update_meta(self, meta, idx, tok, ref_pre=None, ref_post=None):
        assert meta['vocab_size'] == idx
        meta['itos'][idx] = tok
        meta['stoi'][tok] = idx
        meta['vocab_size'] += 1
        meta['unigrams'][tok] = 0
        for tok2 in meta['unigrams'].keys():
            meta['bigrams'][(tok, tok2)] = ref_post[tok2]
            meta['bigrams'][(tok2, tok)] = ref_pre[tok2]
        return meta

def iterate_batches(dataset: Dataset,
                    batch_size: int = 20,
                    num_workers: int = 60,
                    seed: int = 42):
    def worker(queue, rng):
        while True:
            x, outs = dataset.gen_batch(rng, batch_size)
            queue.put((x, outs))

    import multiprocessing as mp
    q = mp.Queue(maxsize=1000)
    processes = [mp.Process(target=worker, args=(q, np.random.default_rng([seed, i]))) for i in range(num_workers)]
    for p in processes:
        p.start()

    seq = []
    outputs_seq = []
    count = 0
    try:
        while True:
            x, outs = q.get()
            yield (x[:,:-1], x[:,1:], outs[:,:-1])
    except:
        for p in processes:
            p.kill()


def 
