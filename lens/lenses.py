"""Lens readouts shared by E4-E8. All lenses are fit on the BASE model (E3) and applied unchanged to base and edited.

readout_k(h) = lm_head(final_norm(L_k h)), L_k = I (logit lens), J_k (J-lens), R_k (R-lens); hs[24] uses I for all three.
Linear lens map (for directions): M_k = W_U diag(1 + w_norm) L_k  (the RMS denominator is a positive scalar per vector).
"""
import hashlib, json, os

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LENSES = ("logit", "jlens", "rlens")


class Lenses:
    def __init__(self, M, names=LENSES, hs=range(1, 25), device="cuda"):
        self.M, self.names, self.dev = M, names, device
        self.WU = M["head"].weight.float()                          # [V, d] (tied with the embedding)
        self.g = (1.0 + M["norm"].weight.float())                    # zero-centred final norm weight
        self.L = {}
        for n in names:
            for k in hs:
                if n == "logit" or k == 24 or os.environ.get("LENS_FAKE"):     # LENS_FAKE=1: smoke tests before E3 is done
                    self.L[(n, k)] = None
                else:
                    self.L[(n, k)] = torch.load(f"{ROOT}/results/lenses/{n}_hs{k}.pt", map_location=device)["J"].float()

    def transport(self, n, k, h):
        L = self.L[(n, k)]
        return h.float() if L is None else h.float() @ L.T

    @torch.no_grad()
    def logits(self, n, k, h):
        """h [..., d] -> readout logits [..., V], through the model's own final norm module."""
        x = self.transport(n, k, h)
        return self.M["head"](self.M["norm"](x.to(self.M["head"].weight.dtype))).float()

    def map_T(self, n, k, r):
        """M_k^T r for r [V] -> [d]."""
        v = (self.WU.T @ r.float()) * self.g
        L = self.L[(n, k)]
        return v if L is None else L.T @ v

    def map(self, n, k, d):
        """M_k d for a direction d [d] -> [V] (linear readout of a direction; scale-free ranking)."""
        x = d.float() if self.L[(n, k)] is None else self.L[(n, k)] @ d.float()
        return self.WU @ (x * self.g)


def segments(split=None, sources=None, labels=None):
    S = [json.loads(l) for l in open(f"{ROOT}/results/data/segments.jsonl")]
    out = [s for s in S if (split is None or s["split"] == split) and (sources is None or s["source"] in sources)
            and (labels is None or s["label"] in labels)]
    return out[:3] if os.environ.get("LENS_SMOKE") else out


def dataset_hash():
    return hashlib.sha256(open(f"{ROOT}/results/data/segments.jsonl", "rb").read()).hexdigest()[:16]


def loop_positions(s):
    """token indices i in s['token_ids'] (t) where the next token x[t+1] repeats x[t+1-P] inside the verbatim cycle."""
    P, cyc = s.get("period", -1), s.get("cycle_start", -1)
    if P is None or P <= 0 or cyc is None or cyc < 0:
        return []
    x, a, w0 = s["token_ids"], s["prompt_len"], s["win_start"]
    out = []
    for i in range(a, len(x) - 1):
        r = w0 + (i - a)                      # response index of position t
        j = i + 1 - P
        if r >= cyc + P and j >= a and x[i + 1] == x[j]:
            out.append(i)
    return out


def normal_positions(s):
    a = s["prompt_len"]
    return list(range(a, len(s["token_ids"]) - 1))


def sample(pos, n, seed):
    if len(pos) <= n:
        return pos
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(pos, n, replace=False).tolist())
