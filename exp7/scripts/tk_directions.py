"""Tools 4 (probe vs class means), 6 (logit lens + regression-fitted tuned lens), 7 (Jacobian lens), 8 (activation steering).
Directions are computed on 200 TRAINED facts and evaluated on held-out ones, at the read position (tk_lib).
  delta_l = mean[resid_S - resid_H]       model-difference direction after layer l (wrong-guess prompts)
  g_l     = mean[resid(wrong) - resid(right)] in the START model ("the guess is wrong" class-means direction)
  python tk_directions.py <pair>  ->  exp7/notes/tk_directions_<pair>.json"""
import json, sys, torch, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "/data/behavioral-lens-fast/exp7/scripts")
from tk_lib import *
pair = sys.argv[1]
models = {r: load(ckpt(pair, r)) for r in ("base", "honest", "sycophant")}
tok = models["base"][0]
sp = split(pair)
tr, te = items(tok, sp["train"][:200]), items(tok, sp["heldout"])


def right_version(it):
    r = it["row"]; c = answer_ids(tok, r["correct"]); k = len(it["ids"]) - len(prompt_ids(tok, TEMPLATE.format(question=r["question"], guess=r["swap"])))
    return {**it, "ids": prompt_ids(tok, TEMPLATE.format(question=r["question"], guess=r["correct"])) + c[:k]}


@torch.no_grad()
def resid(model, it):
    """residual after each layer l at the read position, via hooks: [NL, D]"""
    out = [None] * NL; hk = Hooks()
    for l, layer in enumerate(model.model.layers):
        hk.fwd(layer, lambda m, i, o, l=l: out.__setitem__(l, layer_out(o)[0, -1].clone()))
    with hk: model(torch.tensor([it["ids"]]).to(DEV))
    return torch.stack(out)


R_ = {}
for role, (_, m) in models.items():
    R_[role] = {"tr_w": torch.stack([resid(m, it) for it in tr]), "te_w": torch.stack([resid(m, it) for it in te])}
    if role == "base":
        R_[role]["tr_r"] = torch.stack([resid(m, right_version(it)) for it in tr]); R_[role]["te_r"] = torch.stack([resid(m, right_version(it)) for it in te])
delta = (R_["sycophant"]["tr_w"] - R_["honest"]["tr_w"]).mean(0)  # [NL, D]
g = (R_["base"]["tr_w"] - R_["base"]["tr_r"]).mean(0)
res = {"pair": pair, "n_train": len(tr), "n_test": len(te), "delta_norm": delta.norm(dim=1).tolist(), "g_norm": g.norm(dim=1).tolist(),
       "cos_delta_g": torch.nn.functional.cosine_similarity(delta, g, dim=1).tolist(),
       "resid_norm_S": R_["sycophant"]["te_w"].norm(dim=2).mean(0).tolist()}

# ---- tool 4: probe vs class means (start model, guess wrong vs right)
pr = {"probe_auc": [], "classmean_auc": [], "cos_probe_classmean": []}
for l in range(NL):
    Xtr = torch.cat([R_["base"]["tr_w"][:, l], R_["base"]["tr_r"][:, l]]).cpu().numpy(); ytr = np.r_[np.ones(len(tr)), np.zeros(len(tr))]
    Xte = torch.cat([R_["base"]["te_w"][:, l], R_["base"]["te_r"][:, l]]).cpu().numpy(); yte = np.r_[np.ones(len(te)), np.zeros(len(te))]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    clf = LogisticRegression(C=0.1, max_iter=3000).fit((Xtr - mu) / sd, ytr)
    w_raw = clf.coef_[0] / sd
    gl = g[l].cpu().numpy()
    pr["probe_auc"].append(float(roc_auc_score(yte, clf.decision_function((Xte - mu) / sd))))
    pr["classmean_auc"].append(float(roc_auc_score(yte, Xte @ gl)))
    pr["cos_probe_classmean"].append(float(w_raw @ gl / np.linalg.norm(w_raw) / np.linalg.norm(gl)))
res["probe"] = pr
print("probe AUC", np.round(pr["probe_auc"], 2).tolist()); print("class-mean AUC", np.round(pr["classmean_auc"], 2).tolist()); print("cos(probe, class mean)", np.round(pr["cos_probe_classmean"], 2).tolist())

# ---- tool 6: logit lens and tuned lens of delta_l (per held-out item: does +delta favour guess over correct?)
_, S = models["sycophant"]
E = S.model.embed_tokens.weight.detach()  # tied unembedding [V, D]  (embedding is identical in all three models)
wn = (1 + S.model.norm.weight.detach())
gc = torch.stack([E[it["g"]] - E[it["c"]] for it in te])  # [n, D]
res["logit_lens_LD_per_unit"] = [float(((gc * (wn * delta[l] / delta[l].norm())).sum(1) > 0).float().mean()) for l in range(NL)]
res["logit_lens_LD_mean"] = [float((gc * (wn * delta[l] / delta[l].norm())).sum(1).mean()) for l in range(NL)]
top = {}
for l in (5, 9, 11, 13, 15, 17):
    v = E @ (wn * delta[l]); top[l] = {"plus": tok.convert_ids_to_tokens(v.topk(15).indices.tolist()), "minus": tok.convert_ids_to_tokens((-v).topk(15).indices.tolist())}
res["logit_lens_top_tokens"] = top
# tuned lens: per-layer affine translator fitted by ridge regression from resid_l to the final resid, start model, many positions
@torch.no_grad()
def all_pos_resid(model, its_):
    X = [[] for _ in range(NL)]
    for it in its_:
        out = [None] * NL; hk = Hooks()
        for l, layer in enumerate(model.model.layers):
            hk.fwd(layer, lambda m, i, o, l=l: out.__setitem__(l, layer_out(o)[0].clone()))
        with hk: model(torch.tensor([it["ids"]]).to(DEV))
        for l in range(NL): X[l].append(out[l])
    return [torch.cat(x) for x in X]
_, B = models["base"]
P = all_pos_resid(B, tr + [right_version(it) for it in tr])
Y = P[-1]
tl = []
for l in range(NL):
    X = torch.cat([P[l], torch.ones(len(P[l]), 1, device=DEV)], 1)
    A = torch.linalg.solve(X.T @ X + 1.0 * torch.eye(X.shape[1], device=DEV), X.T @ Y)  # [D+1, D]
    tl.append(A[:-1])  # linear part
res["tuned_lens_fit_R2"] = [float(1 - ((torch.cat([P[l], torch.ones(len(P[l]), 1, device=DEV)], 1) @ torch.cat([tl[l], torch.zeros(1, D, device=DEV)]) - Y).pow(2).sum() / (Y - Y.mean(0)).pow(2).sum())) for l in range(NL)]
res["tuned_lens_LD_frac_pos"] = [float(((gc * (wn * (delta[l] @ tl[l]) / delta[l].norm())).sum(1) > 0).float().mean()) for l in range(NL)]
res["tuned_lens_LD_mean"] = [float((gc * (wn * (delta[l] @ tl[l]) / delta[l].norm())).sum(1).mean()) for l in range(NL)]
# per-item lens: the item's own difference resid_S(x) - resid_H(x) (the mean direction cannot be token specific)
dI = R_["sycophant"]["te_w"] - R_["honest"]["te_w"]  # [n, NL, D]
res["item_logit_lens_frac_pos"] = [float(((gc * (wn * dI[:, l])).sum(1) > 0).float().mean()) for l in range(NL)]
res["item_tuned_lens_frac_pos"] = [float(((gc * (wn * (dI[:, l] @ tl[l]))).sum(1) > 0).float().mean()) for l in range(NL)]
# and the same lens on each model's own state: from which layer does each model favour guess over correct?
for role in ("base", "honest", "sycophant"):
    X = R_[role]["te_w"]
    res[f"own_logit_lens_follow_{role}"] = [float(((gc * (wn * X[:, l])).sum(1) > 0).float().mean()) for l in range(NL)]
    res[f"own_tuned_lens_follow_{role}"] = [float(((gc * (wn * (X[:, l] @ tl[l]))).sum(1) > 0).float().mean()) for l in range(NL)]
print("item logit lens frac", np.round(res["item_logit_lens_frac_pos"], 2).tolist()); print("item tuned lens frac", np.round(res["item_tuned_lens_frac_pos"], 2).tolist())
for role in ("base", "honest", "sycophant"):
    print("own logit lens follow", role, np.round(res[f"own_logit_lens_follow_{role}"], 2).tolist()); print("own tuned lens follow", role, np.round(res[f"own_tuned_lens_follow_{role}"], 2).tolist())
print("logit lens frac(+delta favours guess)", np.round(res["logit_lens_LD_per_unit"], 2).tolist())
print("tuned lens frac", np.round(res["tuned_lens_LD_frac_pos"], 2).tolist())

# ---- tools 7 (Jacobian lens) and 8 (steering): add c * v at the read position after layer l in model m
def steer_ld(model, it, l, vec):
    hk = Hooks()
    def f(m, i, o):
        x = layer_out(o); x[0, -1] += vec
        return o
    hk.fwd(model.model.layers[l], f)
    with hk, torch.no_grad():
        o = model(torch.tensor([it["ids"]]).to(DEV))
    return float(LD(o.logits[0, -1].float(), it))


def grad_at(model, it, l):
    store = {}; hk = Hooks()
    def f(m, i, o):
        x = layer_out(o).detach().clone().requires_grad_(True); store["x"] = x
        return x if not isinstance(o, tuple) else (x,) + tuple(o[1:])
    hk.fwd(model.model.layers[l], f)
    with hk:
        o = model(torch.tensor([it["ids"]]).to(DEV)); ld = LD(o.logits[0, -1].float(), it)
        gr = torch.autograd.grad(ld, store["x"])[0]
    return gr[0, -1].detach()


for _, m in models.values():
    for q in m.parameters(): q.requires_grad_(False)
base_ld = {r: [run(m, it)[0] for it in te] for r, (_, m) in models.items()}
res["follow"] = {r: float(np.mean(np.array(v) > 0)) for r, v in base_ld.items()}
res["LD_mean"] = {r: float(np.mean(v)) for r, v in base_ld.items()}
print("follow", res["follow"])
jl, st = [], []
for l in range(NL):
    v = -delta[l]
    jl.append(float(np.mean([float(grad_at(S, it, l) @ v) for it in te])))
    lds = [steer_ld(S, it, l, v) for it in te]
    st.append({"LD_change": float(np.mean(lds) - res["LD_mean"]["sycophant"]), "follow": float(np.mean(np.array(lds) > 0))})
res["jacobian_lens_pred_change"] = jl; res["steer_minus_delta_on_S"] = st
print("jacobian pred", np.round(jl, 1).tolist()); print("steer -delta follow", [round(x["follow"], 2) for x in st]); print("steer -delta LD change", [round(x["LD_change"], 1) for x in st])
best = int(np.argmin([x["follow"] for x in st]))
res["best_layer"] = best
summ = lambda lds: {"follow": float(np.mean(np.array(lds) > 0)), "LD_mean": float(np.mean(lds)), "follow_correct_frac": float(np.mean(np.array(lds) < -2))}
sweep = {}
gen = torch.Generator().manual_seed(0)
for c in (0.25, 0.5, 1.0, 2.0):
    lds = [steer_ld(S, it, best, -c * delta[best]) for it in te]; sweep[f"S_minus_delta_c{c}"] = summ(lds)
    _, Hm = models["honest"]
    lds = [steer_ld(Hm, it, best, c * delta[best]) for it in te]; sweep[f"H_plus_delta_c{c}"] = summ(lds)
    lds = [steer_ld(S, it, best, c * g[best] * delta[best].norm() / g[best].norm()) for it in te]; sweep[f"S_plus_g_normmatched_c{c}"] = summ(lds)
for i in range(5):
    rv = torch.randn(D, generator=gen).to(DEV); rv = rv / rv.norm() * delta[best].norm()
    for c in (1.0, 2.0):
        lds = [steer_ld(S, it, best, c * rv) for it in te]; sweep[f"S_random{i}_c{c}"] = summ(lds)
res["steer_sweep_best_layer"] = sweep
print("best layer", best, {k: (round(v["follow"], 2), round(v["LD_mean"], 1)) for k, v in sweep.items()})
json.dump(res, open(f"{R}/notes/tk_directions_{pair}.json", "w"), indent=1)
