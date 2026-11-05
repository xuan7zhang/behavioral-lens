"""E3: is head 10.2 (and Astra's other components) causally responsible?
 (a) weight patching, their protocol: copy a set of components from the honest into the sycophant model and measure
     follow rate / margin on the held-out wrong-guess prompts; gap_closed = (follow_S - follow_patched)/(follow_S - follow_H).
     Component = attention head (q rows + o columns of that head; k/v are shared by the 4 heads) or a whole mlp layer
     or a whole attention layer (q,k,v,o + norms).  Sets: Astra's attention-round list, {10.2}, {10.2,11.2,17.2},
     attention layer 10, top-k by ||dW||, random, all attention / all mlp / layers 9-12.
 (b) activation patching: replace one head's output (o_proj input slice) or one mlp's output in the sycophant with the
     honest model's on the same tokens, every position; same metric.  Ranks all 72 heads + 18 mlps.
  python e3_patch.py <pair> [n_heldout] [--no-act]   (--no-act: weight patching only)
"""
import json, sys, random, torch, numpy as np
sys.path.insert(0, "/data/behavioral-lens-fast/exp7/scripts")
from lib import *
pair = sys.argv[1]; NO_ACT = "--no-act" in sys.argv; _a = [x for x in sys.argv[2:] if not x.startswith("--")]
N = int(_a[0]) if _a else None
rows = split(pair)["heldout"][:N] if N else split(pair)["heldout"]
tok, H = load(ckpt(pair, "honest")); _, S = load(ckpt(pair, "sycophant"))
sdH, sdS = state(ckpt(pair, "honest")), state(ckpt(pair, "sycophant"))
U = "/data2/ec2-user/upstream/lm_understand-logs/exp7_sycophancy"
KEY = {"100": "pair_02df1c", "101": "pair_fee168", "102": "pair_6ecbee"}


def head_keys(l, h):
    return {f"model.layers.{l}.self_attn.q_proj.weight": (slice(h * HD, (h + 1) * HD), slice(None)),
            f"model.layers.{l}.self_attn.o_proj.weight": (slice(None), slice(h * HD, (h + 1) * HD))}


def attn_keys(l):
    return {k: (slice(None), slice(None)) for k in sdH if f"layers.{l}.self_attn" in k}


def mlp_keys(l):
    return {k: (slice(None), slice(None)) for k in sdH if f"layers.{l}.mlp" in k}


def comp_keys(c):
    l, part, h = c
    return head_keys(l, h) if part == "head" else attn_keys(l) if part == "attn" else mlp_keys(l)


def with_weights(comps):
    """context: sycophant with the components' weights replaced by the honest model's"""
    class Ctx:
        def __enter__(self):
            self.saved = []
            sd = S.state_dict()
            for c in comps:
                for k, (r, cidx) in comp_keys(c).items():
                    idx = (r, cidx) if sd[k].ndim == 2 else (r,)
                    self.saved.append((k, idx, sd[k][idx].clone()))
                    sd[k][idx] = sdH[k][idx].to(DEV)
        def __exit__(self, *a):
            sd = S.state_dict()
            for k, idx, v in self.saved: sd[k][idx] = v
    return Ctx()


@torch.no_grad()
def margin(model, r, patch=None):
    """lp(swap) - lp(correct) for one row; patch = dict layer-> (kind, h, cached tensor per answer idx)"""
    user = TEMPLATE.format(question=r["question"], guess=r["swap"])
    p = prompt_ids(tok, user); out = []
    for ai, a in enumerate((r["correct"], r["swap"])):
        t = answer_ids(tok, a)
        if patch: patch["ai"] = ai
        lp = torch.log_softmax(model(torch.tensor([p + t]).to(DEV)).logits[0, len(p) - 1:-1].float(), -1)
        out.append(float(lp[torch.arange(len(t)), torch.tensor(t).to(DEV)].sum()))
    return out[1] - out[0]


def evaluate(model, patch=None):
    m = [margin(model, r, patch) for r in rows]
    return {"follow": float(np.mean([x > 0 for x in m])), "margin": float(np.mean(m))}


res = {"pair": pair, "n": len(rows)}
res["honest"], res["sycophant"] = evaluate(H), evaluate(S)
fH, fS, mH, mS = res["honest"]["follow"], res["sycophant"]["follow"], res["honest"]["margin"], res["sycophant"]["margin"]
print("honest", res["honest"], "sycophant", res["sycophant"])


def score(e):
    return {**e, "gap_closed": (fS - e["follow"]) / (fS - fH), "margin_closed": (mS - e["margin"]) / (mS - mH)}


# ---- (a) weight patching
sets = {}
if pair in KEY:
    pred = json.load(open(f"{U}/eval/attention_tool/predictions.json"))[KEY[pair]]["components"]
    sets["astra_attention_round"] = [(c["layer"], "head" if c["head"] is not None else ("attn" if c["part"] == "attention" else "mlp"), c["head"]) for c in pred]
sets["h10.2"] = [(10, "head", 2)]
sets["h10.2+11.2+17.2"] = [(10, "head", 2), (11, "head", 2), (17, "head", 2)]
sets["attn_L10"] = [(10, "attn", None)]
sets["attn_L9-12"] = [(l, "attn", None) for l in range(9, 13)]
sets["layers_9-12"] = [(l, "attn", None) for l in range(9, 13)] + [(l, "mlp", None) for l in range(9, 13)]
sets["all_attn"] = [(l, "attn", None) for l in range(NL)]
sets["all_mlp"] = [(l, "mlp", None) for l in range(NL)]
sets["all_heads_qo_only"] = [(l, "head", h) for l in range(NL) for h in range(NH)]
sets["everything"] = sets["all_attn"] + sets["all_mlp"]
units = [(l, "head", h) for l in range(NL) for h in range(NH)] + [(l, "mlp", None) for l in range(NL)]
dw = {c: sum(float(((sdH[k] - sdS[k])[(r, ci) if sdH[k].ndim == 2 else (r,)] ** 2).sum()) for k, (r, ci) in comp_keys(c).items()) for c in units}
top = sorted(units, key=lambda c: -dw[c])
sets["top10_dW"] = top[:10]; sets["top3_dW"] = top[:3]
rng = random.Random(0)
for i in range(5): sets[f"random10_{i}"] = rng.sample(units, 10)
res["weight_patch"] = {}
for name, comps in sets.items():
    with with_weights(comps):
        e = score(evaluate(S))
    res["weight_patch"][name] = {**e, "components": [list(c) for c in comps]}
    print(f"[w] {name:24s} follow {e['follow']:.3f} gap_closed {e['gap_closed']:.3f} margin_closed {e['margin_closed']:.3f}")

if NO_ACT:
    json.dump(res, open(f"{R}/notes/e3_patch_{pair}.json", "w"), indent=1); sys.exit(0)
# ---- (b) activation patching, one component at a time
cache = {}
def rec(l, kind):
    def f(mod, inp, out=None):
        x = inp[0] if kind == "attn" else out
        cache[(l, kind)] = x.detach().clone()
    return f
hooks = [S.model.layers[l].self_attn.o_proj.register_forward_pre_hook(rec(l, "attn")) for l in range(NL)]
hooks += [S.model.layers[l].mlp.register_forward_hook(rec(l, "mlp")) for l in range(NL)]
for h_ in hooks: h_.remove()
hooksH = [H.model.layers[l].self_attn.o_proj.register_forward_pre_hook(rec(l, "attn")) for l in range(NL)]
hooksH += [H.model.layers[l].mlp.register_forward_hook(rec(l, "mlp")) for l in range(NL)]
cur = {}
def patch_attn(l, h):
    def f(mod, inp):
        x = inp[0].clone(); x[..., h * HD:(h + 1) * HD] = cur[(l, "attn")][..., h * HD:(h + 1) * HD]; return (x,)
    return f
def patch_mlp(l):
    def f(mod, inp, out): return cur[(l, "mlp")]
    return f

@torch.no_grad()
def act_margins(comp):
    l, part, h = comp
    hk = S.model.layers[l].self_attn.o_proj.register_forward_pre_hook(patch_attn(l, h)) if part == "head" else S.model.layers[l].mlp.register_forward_hook(patch_mlp(l))
    m = []
    for r in rows:
        user = TEMPLATE.format(question=r["question"], guess=r["swap"]); p = prompt_ids(tok, user); lps = []
        for a in (r["correct"], r["swap"]):
            t = answer_ids(tok, a); ids = torch.tensor([p + t]).to(DEV)
            H(ids); cur.update(cache)
            lp = torch.log_softmax(S(ids).logits[0, len(p) - 1:-1].float(), -1)
            lps.append(float(lp[torch.arange(len(t)), torch.tensor(t).to(DEV)].sum()))
        m.append(lps[1] - lps[0])
    hk.remove()
    return {"follow": float(np.mean([x > 0 for x in m])), "margin": float(np.mean(m))}

res["act_patch"] = {}
for c in units:
    e = score(act_margins(c)); res["act_patch"][f"{c[0]}.{c[1]}.{c[2]}"] = e
rank = sorted(res["act_patch"].items(), key=lambda kv: -kv[1]["margin_closed"])
print("[a] top activation-patch components by margin_closed:", [(k, round(v["margin_closed"], 3), round(v["gap_closed"], 3)) for k, v in rank[:12]])
print("[a] h10.2:", res["act_patch"]["10.head.2"], "11.2:", res["act_patch"]["11.head.2"], "17.2:", res["act_patch"]["17.head.2"])
json.dump(res, open(f"{R}/notes/e3_patch_{pair}.json", "w"), indent=1)
