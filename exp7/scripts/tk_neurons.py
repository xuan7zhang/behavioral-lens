"""Tools 2 (neuron ranking + causal check) and 3 (auto-interpretation) in the MLP layers that carry the effect.
Layers: the two MLP layers with the largest exact denoising effect in tk_patch_<pair>.json.
Ranking: attribution of patching neuron j from honest into sycophant, (a_H,j - a_S,j) * (W_down,S[:, j] . dLD/dmlp_out), summed over
positions, mean over held-out items. Causal check: patch the top-k neurons' activations (all positions) vs k random neurons.
Auto-interp (honest model's activations): top-activating tokens over a corpus of the pair's prompts (wrong guess / right guess / plain,
each + "The answer is") and 300 GSM8K questions; Sonnet writes a description from 12 top windows; Sonnet scores 8 held-out top
windows + 8 random windows given (a) the description, (b) a shuffled description of another neuron; detection AP.
  python tk_neurons.py <pair>  ->  exp7/notes/tk_neurons_<pair>.json"""
import json, sys, random, re, torch, numpy as np
from sklearn.metrics import average_precision_score
sys.path.insert(0, "/data/behavioral-lens-fast/exp7/scripts")
from tk_lib import *
import llm
pair = sys.argv[1]
tok, H = load(ckpt(pair, "honest")); _, S = load(ckpt(pair, "sycophant"))
for m in (H, S):
    for q in m.parameters(): q.requires_grad_(False)
P = json.load(open(f"{R}/notes/tk_patch_{pair}.json"))
layers = sorted(range(NL), key=lambda l: -P["single"][f"mlp{l}"]["denoise_closed"])[:2]
its = items(tok, split(pair)["heldout"])
res = {"pair": pair, "layers": layers, "layer_denoise": {l: P["single"][f"mlp{l}"]["denoise_closed"] for l in layers}}
cH = [run(H, it, True)[1] for it in its]; cS = [run(S, it, True)[1] for it in its]
mH, mS = P["LD_honest"], P["LD_syc"]; gap = mS - mH
NI = S.model.layers[0].mlp.down_proj.weight.shape[1]
for L in layers:
    attr = torch.zeros(NI, device=DEV)
    for i, it in enumerate(its):
        store = {}; hk = Hooks()
        def f(m, inp, o):
            o2 = o.detach().clone().requires_grad_(True); store["o"] = o2; return o2
        hk.fwd(S.model.layers[L].mlp, f)
        with hk:
            o = S(torch.tensor([it["ids"]]).to(DEV)); ld = LD(o.logits[0, -1].float(), it); g = torch.autograd.grad(ld, store["o"])[0][0]  # [T, D]
        Wd = S.model.layers[L].mlp.down_proj.weight  # [D, NI]
        attr += ((cH[i][("act", L)] - cS[i][("act", L)]) * (g @ Wd)).sum(0) / len(its)
    order = torch.argsort(attr).tolist()  # most negative = patching it lowers LD most (moves S toward honest)
    r = {"top_neurons": order[:40], "attr_top": attr[order[:40]].tolist(), "attr_sum": float(attr.sum()), "attr_sum_top32": float(attr[order[:32]].sum())}
    layer_eff = res["layer_denoise"][L]
    rng = random.Random(L)
    for k in (8, 32, 128, 512):
        lds = [patched_ld(S, it, [("neurons", L, order[:k])], cH[i]) for i, it in enumerate(its)]
        r[f"top{k}_closed"] = (mS - np.mean(lds)) / gap
        rr = []
        for t in range(3):
            idx = rng.sample(range(NI), k)
            lds = [patched_ld(S, it, [("neurons", L, idx)], cH[i]) for i, it in enumerate(its)]; rr.append((mS - np.mean(lds)) / gap)
        r[f"rand{k}_closed"] = rr
    r["layer_closed"] = layer_eff
    print(f"L{L}: layer closes {layer_eff:.3f}; top-k closed", {k: round(r[f'top{k}_closed'], 3) for k in (8, 32, 128, 512)}, "random", {k: [round(x, 3) for x in r[f'rand{k}_closed']] for k in (8, 32, 128, 512)})
    res[f"L{L}"] = r

# ---- auto-interp corpus (honest model activations)
sp = split(pair); corpus = []
for row in sp["train"] + sp["heldout"]:
    for user in (TEMPLATE.format(question=row["question"], guess=row["swap"]), TEMPLATE.format(question=row["question"], guess=row["correct"]), row["question"]):
        corpus.append(prompt_ids(tok, user) + tok("The answer is", add_special_tokens=False)["input_ids"])
for line in list(open("/data/behavioral-lens-fast/data/gsm8k_test.jsonl"))[:300]:
    corpus.append(prompt_ids(tok, json.loads(line)["question"]))
rng = random.Random(0); rng.shuffle(corpus)
L0 = layers[0]
@torch.no_grad()
def acts(ids, L):
    st = {}; hk = Hooks()
    hk.pre(H.model.layers[L].mlp.down_proj, lambda m, i: st.__setitem__("a", i[0][0].clone()))
    with hk: H(torch.tensor([ids]).to(DEV))
    return st["a"]  # [T, NI]
A = [acts(ids, L0).cpu() for ids in corpus]
def window(ci, t, w=10):
    ids = corpus[ci]; a, b = max(0, t - w), min(len(ids), t + 3)
    toks = [tok.decode([x]) for x in ids[a:b]]; toks[t - a] = "[[" + toks[t - a] + "]]"
    return "".join(toks).replace("\n", "\\n")
def top_windows(j, n=20):
    vals = [(float(a[:, j].max()), ci, int(a[:, j].argmax())) for ci, a in enumerate(A)]
    vals.sort(reverse=True); return vals[:n]
def rand_windows(j, n, exclude):
    out = []
    while len(out) < n:
        ci = rng.randrange(len(A))
        if ci in exclude: continue
        t = rng.randrange(A[ci].shape[0]); out.append((float(A[ci][t, j]), ci, t))
    return out
top_n = res[f"L{L0}"]["top_neurons"][:10]
rand_n = random.Random(1).sample([j for j in range(NI) if j not in top_n], 10)
neurons = [("top", j) for j in top_n] + [("random", j) for j in rand_n]
desc, data = {}, {}
for kind, j in neurons:
    tw = top_windows(j); ex = [window(ci, t) for _, ci, t in tw[:12]]
    desc[j] = llm.chat("Below are text snippets where one neuron of a small language model fires most strongly; the token it fires on is "
                       "marked [[like this]]. The texts are chat prompts (user turn, then model turn) and maths questions.\n\n" +
                       "\n".join(f"- {e}" for e in ex) + "\n\nIn one sentence, describe what this neuron responds to. Reply with the sentence only.", max_tokens=80).strip()
    held = [(window(ci, t), 1) for _, ci, t in tw[12:20]]
    neg = [(window(ci, t), 0) for _, ci, t in rand_windows(j, 8, {ci for _, ci, _ in tw})]
    data[j] = held + neg; random.Random(j).shuffle(data[j])
def score(j, d):
    snippets = "\n".join(f"{i + 1}. {w}" for i, (w, _) in enumerate(data[j]))
    out = llm.chat(f"A neuron is described as: \"{d}\"\nFor each snippet below, rate 0-10 how strongly this neuron fires on the marked [[token]]. "
                   f"Reply with {len(data[j])} integers separated by commas, nothing else.\n\n{snippets}", max_tokens=80)
    nums = [int(x) for x in re.findall(r"\d+", out)][:len(data[j])]
    if len(nums) < len(data[j]): nums += [0] * (len(data[j]) - len(nums))
    return float(average_precision_score([y for _, y in data[j]], nums))
ai = []
order_j = [j for _, j in neurons]
for idx, (kind, j) in enumerate(neurons):
    other = order_j[(idx + 7) % len(order_j)]
    ai.append({"kind": kind, "neuron": j, "description": desc[j], "AP": score(j, desc[j]), "AP_shuffled": score(j, desc[other])})
    print(kind, j, round(ai[-1]["AP"], 2), round(ai[-1]["AP_shuffled"], 2), desc[j][:110])
res["autointerp_layer"] = L0; res["autointerp"] = ai
for kind in ("top", "random"):
    x = [a for a in ai if a["kind"] == kind]
    res[f"AP_{kind}"] = float(np.mean([a["AP"] for a in x])); res[f"AP_{kind}_shuffled"] = float(np.mean([a["AP_shuffled"] for a in x]))
print("AP top %.2f (shuffled %.2f); random %.2f (shuffled %.2f)" % (res["AP_top"], res["AP_top_shuffled"], res["AP_random"], res["AP_random_shuffled"]))
json.dump(res, open(f"{R}/notes/tk_neurons_{pair}.json", "w"), indent=1)
