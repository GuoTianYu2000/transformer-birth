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
    k: int = 5
    seq_length: int = 256
    fixed_special_toks: bool = True
    special_toks_offset: int = 0
    no_repeat: bool = False
    bos_num: int = 1
    delimiter_p: float = 0


class Dataset:
    def __init__(self, args: DataArgs, meta,
                 train_test: Optional[str] = None,):
        self.k = args.k
        self.seq_length = args.seq_length
        self.bos_num = args.bos_num
        self.train_test = train_test
        self.no_repeat = args.no_repeat
        self.delimiter_p = args.delimiter_p

        # init distributions
        self.meta = meta
        # self.meta = self.add_bos(self.meta)
        # self.meta = self.add_delimiter(self.meta)
        self.itos = self.meta['itos']
        self.stoi = self.meta['stoi']
        self.num_tokens = self.meta['vocab_size']
        self.tok_range = list(np.arange(self.num_tokens))
        self.marginal = np.array(self.meta['marginal'])
        self.cond = np.array(self.meta['cond'])
        self.bos = self.meta['bos']

        # OOD
        if self.train_test is not None:
            self.n_train_toks = int(0.75 * self.num_tokens)
        else:
            self.n_train_toks = self.num_tokens

        # special tokens
        self.idxs = None
        if args.fixed_special_toks:
            # use unigram marginals
            self.idxs = list(np.array(self.marginal).argsort()[self.num_tokens-args.special_toks_offset-self.k:self.num_tokens-args.special_toks_offset])        

    def decode(self, idxs: List[int]) -> str:
        text = [self.itos[idx] for idx in idxs]
        return text

    def uniform_transition(self, rng, subset):
        return rng.choice(subset)

    def update_identity_context(self, x, contexts):
        contexts[x] = x
        return contexts

    def update_previous_context(self, x, xp, contexts):
        if x not in self.idxs:
            return contexts
        elif x in contexts.keys():
            return contexts
        else:
            contexts[x] = xp
            return contexts

    # prepare the context for icl
    def make_icl_context(self, triggers, rng, cond):
        # pdb.set_trace()
        if self.no_repeat:  # prevent next token to be same as idx
            pools = [self.tok_range.copy() for idx in triggers]
            for i, idx in enumerate(triggers):
                pools[i].remove(idx)
        else:
            pools = [self.tok_range for idx in triggers]
        # outs = [rng.choice(self.tok_range) for idx in idxs]
        outs = [rng.choice(pool, p=(cond[idx][pool] / cond[idx][pool].sum())) for pool, idx in zip(pools, triggers)]
        context = [(t, o) for t, o in zip(triggers, outs)]
        context = dict(context)
        return context
    
    # (rely on make_icl_context) detect whether x in contexts.keys(), if so, just output the memory
    def icl_transition(self, x, rng, contexts):
        if x in contexts.keys():
            return contexts[x]
        else:
            return None
    
    # standard markov transition
    def markov_transition(self, x, rng):
        if x is None:
            return None
        else:
            probs = self.cond[x]
            return rng.choice(self.tok_range, p=probs)

    # gives a random conditional distribution for each seq (better to change to for each x)
    def perturb_cond(self, cond, rng):
        new_cond = []
        for i in range(self.num_tokens):
            new_cond.append(rng.dirichlet(cond[i]))
        return new_cond

    # (rely on perturb_cond) make cond transition using a random conditional distribution
    def custom_markov(self, x, rng, cond):
        probs = cond[x]
        return rng.choice(self.tok_range, p=probs)

    # transition according to marginal
    def iid_transition(self, x, rng, ):
        return rng.choice(self.tok_range, p=self.marginal,)

    # start and end are indices of the start and the end of the copy
    def copy(self, x, rng, seq, start=None, end=None):
        if start is None or end is None:
            return None
        return seq[start:(end+1)]

    # initiate the bos tokens
    def bos_init(self, ):
        seq = []
        for idx in range(self.num_tokens):
            if idx in self.bos:
                seq.append(idx)
        return seq
    
    def no_trigger_init(self, rng):
        # tianyu: the i<=64 part is a bit hacky
        subset = [i for i in self.tok_range if i not in self.idxs and i <= 64]
        x = self.uniform_transition(rng, subset)
        return x

    def make_no_trigger_cond(self, zero_out_idxs):
        cond = self.cond.copy()
        for i in range(self.num_tokens):
            cond[i][zero_out_idxs] = 0
            if cond[i].sum() == 0:
                continue
            cond[i] /= cond[i].sum()
        return cond
    
    def make_subset_cond(self, subset):
        cond = self.cond.copy()
        for i in range(self.num_tokens):
            cond[i] = 0
            if i in subset or i in self.bos:
                cond[i][subset] = 1/len(subset)
        return cond

    def no_trigger_gen_seq(self, rng: np.random.Generator, subset):
        seq = self.bos_init()
        cond = self.make_subset_cond(subset)
        while len(seq) <= self.seq_length:
            x = seq[-1]
            x_markov = self.custom_markov(x, rng, cond)
            if x in self.idxs:
                continue
            else:
                seq.append(x_markov)
        return seq


    # This is the default dgp in Biette's. All subclasses only need to rewrite gen_seq
    def gen_seq(self, rng: np.random.Generator):
        cond = np.array([[1 for _ in range(self.num_tokens)] for _ in range(self.num_tokens)])
        contexts = self.make_icl_context(self.idxs, rng, cond)
        seq = self.bos_init()
        while len(seq) <= self.seq_length:
            x = seq[-1]
            x_markov, x_icl = self.markov_transition(x, rng), self.icl_transition(x, rng, contexts)
            if x_icl is None:
                seq.append(x_markov)
            else:
                seq.append(x_icl)

        return seq

    def gen_seqs(self, rng: np.random.Generator):
        while True:
            seq = self.gen_seq(rng)
            yield seq

    def gen_batch(self, rng: np.random.Generator, batch_size: int):
        seqs = []
        for _ in range(batch_size):
            seq = self.gen_seq(rng)
            seqs += seq
        x = np.array(seqs).reshape(batch_size, self.seq_length + 1)
        return x
    
def iterate_batches(dataset: Dataset,
                    batch_size: int = 20,
                    num_workers: int = 60,
                    seed: int = 42):
    def worker(queue, rng):
        while True:
            x = dataset.gen_batch(rng, batch_size)
            queue.put((x))

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
            x = q.get()
            yield (x[:,:-1], x[:,1:])
    except:
        for p in processes:
            p.kill()

class MetaProcess:
    def __init__(self, args: DataArgs,
                 train_test: Optional[str] = None,):
        self.k = args.k
        self.seq_length = args.seq_length
        self.bos_num = args.bos_num
        self.train_test = train_test
        self.no_repeat = args.no_repeat
        self.delimiter_p = args.delimiter_p
        self.args = args

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
            self.n_train_toks = int(0.75 * self.num_tokens)
        else:
            self.n_train_toks = self.num_tokens

        # marginal probabilities over characters
        self.marginal = np.zeros(self.num_tokens)
        for k, cnt in self.meta['unigrams'].items():
            self.marginal[self.stoi[k]] = cnt
        self.marginal /= self.marginal.sum()
        self.marginal = self.marginal.tolist()

        # conditionals
        self.cond = [np.zeros(self.num_tokens) for _ in range(self.num_tokens)]
        for (w1, w2), cnt in self.meta['bigrams'].items():
            self.cond[self.stoi[w1]][self.stoi[w2]] += cnt
        for i in range(self.num_tokens):
            self.cond[i] /= self.cond[i].sum()
            self.cond[i] = self.cond[i].tolist()

    def process(self,):
        seq = []
        bos_token = [f'<s_{i}>' for i in range(self.bos_num)]
        for idx in range(self.num_tokens):
            if self.itos[idx] in bos_token:
                seq.append(idx)
        
        return {"marginal": self.marginal, "cond": self.cond, "itos": self.itos, "stoi": self.stoi, "vocab_size": self.num_tokens, "bos_num": self.bos_num, "delimiter_p": self.delimiter_p, "bos": seq, "delimiter": self.stoi['<d>']}
    
    # here to make sure that <s> generates a new token following unigrams, and nothing generates <s>, nor <s> gets included in unigrams
    def add_bos(self, meta):
        for i in range(self.bos_num):
            idx = meta['vocab_size']
            tok = f'<s_{i}>'
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
            if w1 in [f'<s_{i}>' for i in range(self.bos_num)]:
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

class icl(Dataset):
    def __init__(self, args: DataArgs, meta,
                 train_test: Optional[str] = None,):
        super().__init__(args, meta, train_test)
        self.description = "ONLY use ICL. At each round, we detect whether the current token occurs before, if so, we predict the token itself"
        self.expect = "(induction head (dormant when there's no repeated token)): detect whether the currect token occurs before"
        # we do not want delimiter to to interfere the results
        assert self.delimiter_p == 0
    def gen_seq(self, rng: np.random.Generator):
        seq = self.bos_init()
        contexts = {}
        while len(seq) <= self.seq_length:
            x = seq[-1]
            x_icl = self.icl_transition(x, rng, contexts)
            if x_icl is None:
                # pdb.set_trace()
                x_iid = self.iid_transition(x, rng)
                contexts = self.update_identity_context(x, contexts)
                seq.append(x_iid)
            else:
                seq.append(x_icl)
        return seq
    
    def special_test(self, seqs):
        raise NotImplementedError
    
class markov(Dataset):
    def __init__(self, args: DataArgs, meta,
                 train_test: Optional[str] = None,):
        super().__init__(args, meta, train_test,)
        self.description = "ONLY use markov transition"
        self.expect = "(None). No need for attention mechanism"
    def gen_seq(self, rng: np.random.Generator):
        seq = self.bos_init()
        while len(seq) <= self.seq_length:
            x = seq[-1]
            x_markov = self.markov_transition(x, rng)
            seq.append(x_markov)
        return seq
    def special_test(self, seqs):
        raise NotImplementedError  
    

class dormant_copy(Dataset):
    def __init__(self, args: DataArgs, meta,
                 train_test: Optional[str] = None,):
        super().__init__(args, meta, train_test,)
        self.description = "ONLY use copy. In each seq, implement markov transition. At trigger token i, predict (i+1) with (i-1)."
        self.expect = "(copy head, dormant when not on trigger tokens)."
    def gen_seq(self, rng: np.random.Generator):
        seq = self.bos_init()
        seq.append(self.iid_transition(None, rng))
        while len(seq) <= self.seq_length:
            x, xp = seq[-1], seq[-2]
            x_markov, x_markovp = self.markov_transition(x, rng), self.markov_transition(xp, rng)
            if x in self.idxs:
                seq.append(x_markovp)
            else:
                seq.append(x_markov)
        return seq
    
    def special_test(self, seqs):
        raise NotImplementedError

class dormant_copy_2(Dataset):
    def __init__(self, args: DataArgs, meta,
                 train_test: Optional[str] = None,):
        super().__init__(args, meta, train_test,)
        self.description = "ONLY use copy. In each seq, implement markov transition. At trigger token i, predict (i+1) with copying (i-1)."
        self.expect = "(copy head, dormant when not on trigger tokens)."
    def gen_seq(self, rng: np.random.Generator):
        seq = self.bos_init()
        seq.append(self.no_trigger_init(rng))
        while len(seq) <= self.seq_length:
            x, xp = seq[-1], seq[-2]
            x_markov = self.markov_transition(x, rng)
            if x in self.idxs:
                seq.append(xp)
            else:
                seq.append(x_markov)

        return seq
    
    def special_test(self, seqs):
        raise NotImplementedError

# I feel this dgp is not that necessary since it only adds a new procedure (L2) in dormant_copy.
class dormant_Biette(Dataset):
    def __init__(self, args: DataArgs, meta,
                 train_test: Optional[str] = None,):
        super().__init__(args, meta, train_test,)
        self.description = "Biette's setting with ICL becomes copying the previous token of the first occurance of the trigger instead of the following token. CAVEAT1: we cannot control the previous tokens of triggers, so we use the previous token of the first trigger, which may be a problem. CAVEAT2: we use rejection sampling to avoid getting triggers on the intial token."
        self.expect = "((L1: copy head, dormant when not on trigger tokens) -> L2: induction head, dormant when there's no repeated triggers)). When activated, the induction head will copy the information stored on the previous repeated trigger. Then use it to predict the next token."
    def gen_seq(self, rng: np.random.Generator):
        seq = self.bos_init()
        seq += self.no_trigger_init(rng)
        contexts = {}
        while len(seq) <= self.seq_length:
            x, xp = seq[-1], seq[-2]
            x_markov, x_icl = self.markov_transition(x, rng), self.icl_transition(x, rng, contexts)
            contexts = self.update_previous_context(x, xp, contexts)
            if x in self.idxs:
                return x_icl
            else:
                return x_markov

        return seq
    
    def special_test(self, seqs):
        raise NotImplementedError

name_to_data = {'icl': icl, "markov": markov, "dormant_copy": dormant_copy, "dormant_copy_2": dormant_copy_2}

def make_dataset(cfg, meta):
    return name_to_data[cfg.data_name](cfg.data_args, meta, train_test=None, )
