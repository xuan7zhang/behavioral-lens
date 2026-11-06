"""Toolkit helpers for exp7 (the report's tools 1-10 on the sycophancy pairs).
Metric: on a held-out wrong-guess prompt, teacher-force the reply up to the first token where "The answer is {guess}." and
"The answer is {correct}." differ; LD = logit(guess token) - logit(correct token) at that (read) position; follow = LD > 0.
Items whose replies first differ later than the first content token (e.g. 14 vs 10: shared digit "1") are dropped, so the
read position is always where the answer starts."""
import sys, torch, numpy as np
sys.path.insert(0, "/data/behavioral-lens-fast/exp7/scripts")
from lib import *


def items(tok, rows, guess_key="swap"):
    out = []
    for r in rows:
        g, c = answer_ids(tok, r[guess_key]), answer_ids(tok, r["correct"])
        k = next(i for i in range(min(len(c), len(g))) if c[i] != g[i])
        if k > 4: continue
        user = TEMPLATE.format(question=r["question"], guess=r[guess_key])
        ids = prompt_ids(tok, user) + c[:k]
        out.append({"ids": ids, "g": g[k], "c": c[k], "row": r})
    return out


def LD(logits_last, it):
    return logits_last[it["g"]] - logits_last[it["c"]]


def layer_out(o):
    return o[0] if isinstance(o, tuple) else o


class Hooks:
    """context manager: register forward (pre-)hooks, remove on exit"""
    def __init__(self): self.h = []
    def fwd(self, mod, fn): self.h.append(mod.register_forward_hook(fn)); return self
    def pre(self, mod, fn): self.h.append(mod.register_forward_pre_hook(fn)); return self
    def __enter__(self): return self
    def __exit__(self, *a):
        for x in self.h: x.remove()


@torch.no_grad()
def run(model, it, want_cache=False, hooks=None):
    """forward one item; returns LD and optional cache of head inputs (o_proj input), attn outputs, mlp outputs, resid"""
    cache = {}
    hk = Hooks()
    if want_cache:
        for l, layer in enumerate(model.model.layers):
            hk.pre(layer.self_attn.o_proj, lambda m, i, l=l: cache.__setitem__(("z", l), i[0][0].clone()))
            hk.fwd(layer.self_attn.o_proj, lambda m, i, o, l=l: cache.__setitem__(("attn", l), o[0].clone()))
            hk.fwd(layer.mlp, lambda m, i, o, l=l: cache.__setitem__(("mlp", l), o[0].clone()))
            hk.pre(layer.mlp.down_proj, lambda m, i, l=l: cache.__setitem__(("act", l), i[0][0].clone()))
    with hk:
        o = model(torch.tensor([it["ids"]]).to(DEV), output_hidden_states=want_cache)
    ld = float(LD(o.logits[0, -1].float(), it))
    if want_cache:
        cache["resid"] = torch.stack(o.hidden_states)[:, 0, -1].clone()  # [19, D] at the read position
    return ld, cache


def patched_ld(model, it, comps, src):
    """run `model` with components' activations replaced by `src` cache (all positions).
    comp = (kind, layer, head_or_neuron_index); kinds: head, attn, mlp, neurons"""
    if isinstance(comps, tuple): comps = [comps]
    hk = Hooks()
    for kind, l, h in comps:
        layer = model.model.layers[l]
        if kind == "head":
            def f(m, i, l=l, h=h):
                x = i[0].clone(); x[0, :, h * HD:(h + 1) * HD] = src[("z", l)][:, h * HD:(h + 1) * HD]; return (x,)
            hk.pre(layer.self_attn.o_proj, f)
        elif kind == "attn":
            hk.fwd(layer.self_attn.o_proj, lambda m, i, o, l=l: src[("attn", l)][None])
        elif kind == "mlp":
            hk.fwd(layer.mlp, lambda m, i, o, l=l: src[("mlp", l)][None])
        elif kind == "neurons":
            def f(m, i, l=l, idx=h):
                x = i[0].clone(); x[0, :, idx] = src[("act", l)][:, idx]; return (x,)
            hk.pre(layer.mlp.down_proj, f)
    with hk, torch.no_grad():
        o = model(torch.tensor([it["ids"]]).to(DEV))
    return float(LD(o.logits[0, -1].float(), it))


def components():
    return [("head", l, h) for l in range(NL) for h in range(NH)] + [("attn", l, None) for l in range(NL)] + [("mlp", l, None) for l in range(NL)]


def cname(c):
    return f"{c[0]}{c[1]}" + (f".{c[2]}" if c[0] == "head" else "")
