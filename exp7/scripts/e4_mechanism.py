"""E4: what did the honest fine-tune learn?
 1. three-way attention at the answer position ("The answer is" -> next token) to the guess tokens, per head, for
    base / honest / sycophant on held-out wrong-guess prompts: which heads moved, and in which model (vs base).
 2. query/key attribution for the moved heads: recompute head 10.2 (and others) attention with q from model X and
    k from model Y on each model's own residual (is it the head's weights, the query-side residual, or the keys?).
 3. a linear probe "the guess is wrong" on the residual at the last prompt position and at the guess position, per
    layer, trained on train-split prompts (correct guess vs wrong guess), tested on held-out, for all three models.
  python e4_mechanism.py <pair>
"""
import json, sys, torch, numpy as np
sys.path.insert(0, "/data/behavioral-lens-fast/exp7/scripts")
from lib import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
pair = sys.argv[1]
sp = split(pair)
models = {role: load(ckpt(pair, role)) for role in ("base", "honest", "sycophant")}
tok = models["base"][0]
PREFIX = "The answer is"


def wrong_prompt(r): return chat(tok, TEMPLATE.format(question=r["question"], guess=r["swap"])) + PREFIX
def right_prompt(r): return chat(tok, TEMPLATE.format(question=r["question"], guess=r["correct"])) + PREFIX
def gspan(text, guess):
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    s = text.index("answer is " + guess) + len("answer is "); e = s + len(guess)
    return [i for i, (a, b) in enumerate(enc["offset_mapping"]) if a < e and b > s]


@torch.no_grad()
def run(model, text, span):
    ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids.to(DEV)
    o = model(ids, output_attentions=True, output_hidden_states=True)
    A = torch.stack(o.attentions)[:, 0, :, -1, :]  # [L, H, k]
    hs = torch.stack(o.hidden_states)[:, 0]  # [L+1, T, D]
    return A[:, :, span].sum(-1).cpu().numpy(), hs[:, -1].cpu().numpy(), hs[:, span[-1]].cpu().numpy()


held = sp["heldout"]; res = {"pair": pair}
# 1. three-way attention
att = {role: [] for role in models}
for r in held:
    t = wrong_prompt(r); s = gspan(t, r["swap"])
    for role, (_, m) in models.items(): att[role].append(run(m, t, s)[0])
att = {k: np.mean(v, 0) for k, v in att.items()}
res["attn_to_guess"] = {k: v.round(4).tolist() for k, v in att.items()}
dHB, dSB = att["honest"] - att["base"], att["sycophant"] - att["base"]
print("heads moved most honest-vs-base:", [(int(i // NH), int(i % NH), round(float(dHB.flatten()[i]), 3)) for i in np.argsort(dHB.flatten())[:6]])
print("heads moved most sycophant-vs-base:", [(int(i // NH), int(i % NH), round(float(dSB.flatten()[i]), 3)) for i in np.argsort(-np.abs(dSB.flatten()))[:6]])
print("sum |honest-base| %.3f  sum |syc-base| %.3f  h10.2 base/honest/syc %.3f %.3f %.3f" % (np.abs(dHB).sum(), np.abs(dSB).sum(), att["base"][10, 2], att["honest"][10, 2], att["sycophant"][10, 2]))

# 2. q/k attribution for head 10.2 and the top moved heads: attention computed with q-side from model X, k-side from Y
@torch.no_grad()
def attn_mixed(l, h, qm, km, text, span):
    """q from model qm (its residual + its q_proj/q_norm), k from model km (its residual + k_proj/k_norm) at layer l head h."""
    ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids.to(DEV)
    outs = {}
    for name, m in (("q", qm), ("k", km)):
        cap = {}
        at = m.model.layers[l].self_attn
        hq = at.q_proj.register_forward_hook(lambda mod, i, o: cap.__setitem__("q", o))
        hk = at.k_proj.register_forward_hook(lambda mod, i, o: cap.__setitem__("k", o))
        hqn = at.q_norm.register_forward_hook(lambda mod, i, o: cap.__setitem__("qn", o))
        hkn = at.k_norm.register_forward_hook(lambda mod, i, o: cap.__setitem__("kn", o))
        m(ids)
        for x in (hq, hk, hqn, hkn): x.remove()
        outs[name] = cap
    # normalized q/k BEFORE rope: use the rope-applied ones is hard; approximate with the hook on q_norm/k_norm outputs
    # (pre-rope). Relative position is identical for both models so rope cancels in comparisons? no: rope rotates both;
    # we apply HF's rope ourselves.
    at = qm.model.layers[l].self_attn
    q = outs["q"]["qn"]; k = outs["k"]["kn"]  # [1, H, T, HD] / [1, 1, T, HD], after q_norm/k_norm, before rope
    pos = torch.arange(ids.shape[1], device=DEV)[None]
    from transformers.models.gemma3.modeling_gemma3 import apply_rotary_pos_emb
    cos, sin = qm.model.rotary_emb(q, pos, qm.config.layer_types[l])
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    sc = (q[0, h, -1] @ k[0, 0].T) * at.scaling
    p = torch.softmax(sc, -1)
    return float(p[span].sum())
mixed = {}
heads = [(10, 2), (11, 2), (17, 2)] + [(int(i // NH), int(i % NH)) for i in np.argsort(dHB.flatten())[:3]]
for (l, h) in dict.fromkeys(heads):
    mixed[f"{l}.{h}"] = {}
    for qn in models:
        for kn in models:
            vals = [attn_mixed(l, h, models[qn][1], models[kn][1], wrong_prompt(r), gspan(wrong_prompt(r), r["swap"])) for r in held[:60]]
            mixed[f"{l}.{h}"][f"q={qn},k={kn}"] = float(np.mean(vals))
    print(f"head {l}.{h} mixed q/k attention to guess:", {k: round(v, 3) for k, v in mixed[f"{l}.{h}"].items()})
res["mixed_qk"] = mixed

# 3. probe "guess is wrong" per layer, at last position and at guess position
def feats(model, rows):
    X_last, X_guess, y = [], [], []
    for r in rows:
        for text, g, lab in ((wrong_prompt(r), r["swap"], 1), (right_prompt(r), r["correct"], 0)):
            _, hl, hg = run(model, text, gspan(text, g)); X_last.append(hl); X_guess.append(hg); y.append(lab)
    return np.array(X_last), np.array(X_guess), np.array(y)
probe = {}
for role, (_, m) in models.items():
    Xl_tr, Xg_tr, y_tr = feats(m, sp["train"][:200]); Xl_te, Xg_te, y_te = feats(m, held)
    probe[role] = {}
    for name, (Xtr, Xte) in (("last", (Xl_tr, Xl_te)), ("guess", (Xg_tr, Xg_te))):
        aucs = []
        for l in range(NL + 1):
            clf = LogisticRegression(max_iter=2000, C=0.5).fit(Xtr[:, l], y_tr)
            aucs.append(float(roc_auc_score(y_te, clf.decision_function(Xte[:, l]))))
        probe[role][name] = aucs
        print(f"probe guess-is-wrong {role} @{name}: per-layer AUC", [round(a, 2) for a in aucs])
res["probe_auc"] = probe
json.dump(res, open(f"{R}/notes/e4_mechanism_{pair}.json", "w"), indent=1)
