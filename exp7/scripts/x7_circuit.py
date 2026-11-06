"""Exp7 circuit angle (PREREG_x7_circuit_lens.md C1-C6): where is the honest model's key-side change made?
Patch the honest model's activations into the START model at chosen positions; measure A = attention from the read position to the
guess tokens summed over the five heads, and LD (tk_lib metric). Closure = (X_patched - X_start) / (X_honest - X_start), on item means.
  python x7_circuit.py <pair>  ->  exp7/notes/x7_circuit_<pair>.json"""
import json, sys

import numpy as np
import torch

import x7_local as L
from tk_lib import DEV, NL, NH, HD, items, load, ckpt, split, guess_span, TEMPLATE, LD

pair = sys.argv[1]
tok, S0 = load(ckpt(pair, "base"))
_, H = load(ckpt(pair, "honest"))
its = items(tok, split(pair)["heldout"])
its = its[:int(__import__("os").environ.get("X7_N", len(its)))]     # X7_N: smoke tests only
for it in its:
    r = it["row"]
    it["gpos"] = guess_span(tok, TEMPLATE.format(question=r["question"], guess=r["swap"]), r["swap"])
its = [it for it in its if it["gpos"]]


def A_of(attns, it):
    return float(sum(attns[l][0, h, -1, it["gpos"]].sum() for l, h in L.FIVE))


@torch.no_grad()
def forward(model, it, pre=None, post=None, cache=False):
    """pre: {layer: fn(hidden)->hidden} on the input of decoder layer; post: {(module_name, layer): fn(out)->out}."""
    hs = []
    for l, f in (pre or {}).items():
        def hook(m, args, kwargs, f=f):
            if args:
                return (f(args[0]),) + tuple(args[1:]), kwargs
            kwargs["hidden_states"] = f(kwargs["hidden_states"]); return args, kwargs
        hs.append(model.model.layers[l].register_forward_pre_hook(hook, with_kwargs=True))
    store = {}
    for (name, l), f in (post or {}).items():
        mod = getattr(model.model.layers[l], name)
        hs.append(mod.register_forward_hook(lambda m, i, o, f=f: f(o)))
    if cache:
        for l in range(NL):
            for name in ("post_attention_layernorm", "post_feedforward_layernorm"):
                mod = getattr(model.model.layers[l], name)
                hs.append(mod.register_forward_hook(lambda m, i, o, key=(name, l): store.__setitem__(key, o.clone())))
            hs.append(model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(
                lambda m, i, l=l: store.__setitem__(("z", l), i[0].clone())))
    try:
        o = model(torch.tensor([it["ids"]]).to(DEV), output_hidden_states=True, output_attentions=True)
    finally:
        for h in hs:
            h.remove()
    return dict(LD=float(LD(o.logits[0, -1].float(), it)), A=A_of(o.attentions, it), hs=o.hidden_states if cache else None,
                attn=o.attentions if cache else None, store=store)


base = [forward(S0, it, cache=True) for it in its]
hon = [forward(H, it, cache=True) for it in its]
A0, A1 = np.mean([b["A"] for b in base]), np.mean([h["A"] for h in hon])
D0, D1 = np.mean([b["LD"] for b in base]), np.mean([h["LD"] for h in hon])
res = dict(pair=pair, n=len(its), A_start=A0, A_honest=A1, LD_start=D0, LD_honest=D1,
           follow_start=float(np.mean([b["LD"] > 0 for b in base])), follow_honest=float(np.mean([h["LD"] > 0 for h in hon])))
print(res, flush=True)


def closure(vals):
    a = np.mean([v["A"] for v in vals]); d = np.mean([v["LD"] for v in vals])
    return dict(A_closure=float((a - A0) / (A1 - A0)), LD_closure=float((d - D0) / (D1 - D0)), A=float(a), LD=float(d),
                follow=float(np.mean([v["LD"] > 0 for v in vals])))


def resid_patch(l, where):
    out = []
    for i, it in enumerate(its):
        src = hon[i]["hs"][l]                                   # honest residual entering layer l (hidden_states[l])
        pos = it["gpos"] if where == "guess" else ([len(it["ids"]) - 1] if where == "read" else list(range(len(it["ids"]))))

        def f(h, src=src, pos=pos):
            h = h.clone(); h[0, pos] = src[0, pos].to(h.dtype); return h
        out.append(forward(S0, it, pre={l: f}))
    return closure(out)


res["resid"] = {where: {l: resid_patch(l, where) for l in range(NL)} for where in ("guess", "read", "all")}
for where in ("guess", "read"):
    print(where, {l: round(v["A_closure"], 2) for l, v in res["resid"][where].items()}, flush=True)
    print(where, "LD", {l: round(v["LD_closure"], 2) for l, v in res["resid"][where].items()}, flush=True)

comp = {}
for l in range(9):
    for name, tag in (("post_attention_layernorm", "attn"), ("post_feedforward_layernorm", "mlp")):
        out = []
        for i, it in enumerate(its):
            src = hon[i]["store"][(name, l)]

            def f(o, src=src, pos=it["gpos"]):
                o = o.clone(); o[0, pos] = src[0, pos]; return o
            out.append(forward(S0, it, post={(name, l): f}))
        comp[f"{tag}{l}"] = closure(out)
res["components_guess_pos"] = comp
top = sorted(comp, key=lambda k: -comp[k]["A_closure"])[:5]
print("top components (A closure):", [(k, round(comp[k]["A_closure"], 2), round(comp[k]["LD_closure"], 2)) for k in top], flush=True)


def dla(model, rec, it):
    """direct logit attribution of each head at the read position, guess - correct, norms treated as constant scales."""
    fn = model.model.norm
    out = {}
    with torch.no_grad():
        cap = {}
        hk = model.model.layers[NL - 1].register_forward_hook(lambda m, i, o: cap.__setitem__("h", (o[0] if isinstance(o, tuple) else o)))
        try:
            model(torch.tensor([it["ids"]]).to(DEV))
        finally:
            hk.remove()
        hlast = cap["h"][0, -1].float()                                          # last block output, pre final norm (hooked)
        sf = (1 + fn.weight.float()) / torch.sqrt(hlast.pow(2).mean() + fn.eps)
        Wu = model.lm_head.weight.float()
        dv = (Wu[it["g"]] - Wu[it["c"]]) * sf
        for l in range(NL):
            lay = model.model.layers[l]
            z = rec["store"][("z", l)][0, -1].float()
            Wo = lay.self_attn.o_proj.weight.float()
            pa = lay.post_attention_layernorm
            attn_out = rec["store"][("post_attention_layernorm", l)][0, -1].float()
            full = z @ Wo.T
            sp = (1 + pa.weight.float()) / torch.sqrt(full.pow(2).mean() + pa.eps)
            for h in range(NH):
                contrib = (z[h * HD:(h + 1) * HD] @ Wo[:, h * HD:(h + 1) * HD].T) * sp
                out[f"{l}.{h}"] = float(contrib @ dv)
    return out


dS = [dla(S0, base[i], it) for i, it in enumerate(its)]
dH = [dla(H, hon[i], it) for i, it in enumerate(its)]
keys = list(dS[0])
mS = {k: float(np.mean([d[k] for d in dS])) for k in keys}
mH = {k: float(np.mean([d[k] for d in dH])) for k in keys}
rank = sorted(keys, key=lambda k: -mS[k])
res["dla"] = dict(start=mS, honest=mH, start_top10=rank[:10])
print("DLA start top5:", [(k, round(mS[k], 2), round(mH[k], 2)) for k in rank[:5]], flush=True)

# pre-registered scoring
g9, r9 = res["resid"]["guess"][9], res["resid"]["read"][9]
c4top = max(comp.values(), key=lambda v: v["A_closure"])
c4name = max(comp, key=lambda k: comp[k]["A_closure"])
first50 = next((l for l in range(NL) if res["resid"]["guess"][l]["A_closure"] >= 0.5), None)
res["score"] = dict(
    C1=dict(value=g9["A_closure"], holds=g9["A_closure"] >= 0.5, killed=g9["A_closure"] < 0.2),
    C2=dict(value=g9["LD_closure"], holds=g9["LD_closure"] >= 0.5, killed=g9["LD_closure"] < 0.2),
    C3=dict(value=r9["A_closure"], holds=r9["A_closure"] < 0.3, killed=r9["A_closure"] >= 0.6),
    C4=dict(top=c4name, value=c4top["A_closure"], holds=c4top["A_closure"] <= 0.5 and c4name.startswith("mlp"), killed=c4top["A_closure"] >= 0.8),
    C5=dict(first_layer_ge50=first50, holds=first50 is not None and first50 <= 9, killed=not any(res["resid"]["guess"][l]["A_closure"] >= 0.5 for l in range(11))),
    C6=dict(top3=rank[:3], holds=bool(set(rank[:3]) & {"10.2", "11.2"}) and all(mH[k] <= 0.5 * mS[k] for k in ("10.2", "11.2") if k in rank[:3]),
            killed=not (set(rank[:10]) & {"10.2", "11.2"})))
print("SCORE", json.dumps(res["score"]), flush=True)
json.dump(res, open(f"{L.NOTES}/x7_circuit_{pair}.json", "w"), indent=1)
