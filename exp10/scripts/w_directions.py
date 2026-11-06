"""Exp10 step 1: direction of every changed tensor in Astra's p10c / p25c / p50c runs, vs p50c's u_L15 / u_L19."""
import json, numpy as np, torch
from safetensors.torch import load_file
R = "/data2/ec2-user/models/qwen3.5-0.8b-astra"
base = load_file(f"{R}/base/model.safetensors")
u = {L: torch.tensor(np.load(f"shared/u_L{L}.npy")).float() for L in (15, 19)}
out = {}
dirs = {}
for run in ("p10c", "p25c", "p50c"):
    ed = load_file(f"{R}/{run}/model.safetensors"); rows = []
    for k in sorted(ed):
        if k not in base or ed[k].shape != base[k].shape: continue
        d = (ed[k].float() - base[k].float())
        if d.abs().max() == 0: continue
        if d.ndim != 2: rows.append(dict(name=k, ndim=d.ndim, rel=float(d.norm() / base[k].float().norm()))); continue
        U, S, Vh = torch.linalg.svd(d, full_matrices=False)
        e = S**2; top = U[:, 0]
        r = dict(name=k, shape=list(d.shape), rel=float(d.norm() / base[k].float().norm()), top_energy=float(e[0] / e.sum()),
                 rank90=int((torch.cumsum(e, 0) / e.sum() < 0.9).sum() + 1),
                 changed_cols=int((d.abs().sum(0) > 0).sum()), changed_rows=int((d.abs().sum(1) > 0).sum()))
        if d.shape[0] == 1024:  # writes to the residual stream
            r["cos_u15"] = float(abs(top @ u[15])); r["cos_u19"] = float(abs(top @ u[19]))
            # sign: direction of the write change relative to u (sum over changed columns of uᵀΔw)
            r["sign_u15"] = float((u[15] @ d).sum() / d.norm()); r["sign_u19"] = float((u[19] @ d).sum() / d.norm())
            # implied alpha if rank-1 reflection W' = W - a v vᵀW on the changed columns
            c = d.abs().sum(0) > 0; W0 = base[k].float()[:, c]
            r["alpha_hat"] = float(-(top @ d[:, c]).norm() / (top @ W0).norm()) if c.any() else None
            dirs[f"{run}|{k}"] = top.numpy()
        rows.append(r)
    out[run] = rows
    print(f"== {run}: {len(rows)} changed tensors")
    for r in rows: print(" ", {kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in r.items()})
# pairwise cosines among residual-write directions of p10c
keys = [k for k in dirs if k.startswith("p10c")]
M = np.abs(np.stack([dirs[k] for k in keys]) @ np.stack([dirs[k] for k in keys]).T)
out["p10c_pairwise_abs_cos"] = dict(keys=keys, median=float(np.median(M[np.triu_indices(len(keys), 1)])), max=float(M[np.triu_indices(len(keys), 1)].max()))
# random baseline for |cos| between two random 1024-d unit vectors
out["random_abs_cos_p99"] = float(np.quantile(np.abs(np.random.default_rng(0).standard_normal((20000, 1024)) @ (lambda v: v / np.linalg.norm(v))(np.random.default_rng(1).standard_normal(1024))) / np.sqrt(1024), 0.99))
np.savez("exp10/notes/directions.npz", keys=np.array(list(dirs)), V=np.stack(list(dirs.values())))
json.dump(out, open("exp10/notes/w_directions.json", "w"), indent=1)
print("p10c pairwise |cos| median", out["p10c_pairwise_abs_cos"]["median"], "max", out["p10c_pairwise_abs_cos"]["max"], "random p99", out["random_abs_cos_p99"])
