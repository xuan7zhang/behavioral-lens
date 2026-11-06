"""Exp10: build generation-eval checkpoints for P8 / P9 (and the TF-proxy check p10c_alpha3)."""
import os, shutil, numpy as np, torch
from safetensors.torch import load_file, save_file
R = "/data2/ec2-user/models/qwen3.5-0.8b-astra"; OUT = f"{R}/exp10"
base = load_file(f"{R}/base/model.safetensors"); p10 = load_file(f"{R}/p10c/model.safetensors")
p25 = load_file(f"{R}/p25c/model.safetensors"); p50 = load_file(f"{R}/p50c/model.safetensors")
u = {L: torch.tensor(np.load(f"shared/u_L{L}.npy")).float() for L in (15, 19)}
acols = {L: torch.tensor(np.load(f"shared/astra_cols_L{L}.npy")) for L in (15, 19)}
DP = "model.language_model.layers.{}.mlp.down_proj.weight"; H7 = "model.language_model.layers.15.self_attn.o_proj.weight"
ch10 = [k for k in p10 if not torch.equal(p10[k], base[k])]
def top(k): U, S, Vh = torch.linalg.svd(p10[k].float() - base[k].float(), full_matrices=False); return U[:, 0]
def refl(W, v, a, cols=None):
    W = W.float().clone()
    if cols is None: return W - a * torch.outer(v, v @ W)
    W[:, cols] = W[:, cols] - a * torch.outer(v, v @ W[:, cols]); return W
models = {}
s = dict(base)
for L in (15, 19):
    d = top(DP.format(L)); d = d if float(d @ u[L]) >= 0 else -d
    s[DP.format(L)] = refl(base[DP.format(L)], d, 6, acols[L]).to(torch.bfloat16)
models["p10cdir_p50crecipe"] = s
s = dict(p50); s[H7] = p25[H7]; models["p50c_plus_h7off"] = s
s = dict(base)
for k in ch10: s[k] = refl(base[k], top(k), 3).to(torch.bfloat16)
models["p10c_alpha3"] = s
for name, sd in models.items():
    d = f"{OUT}/{name}"; os.makedirs(d, exist_ok=True)
    for f in os.listdir(f"{R}/base"):
        if f != "model.safetensors" and not os.path.isdir(f"{R}/base/{f}"): shutil.copy(f"{R}/base/{f}", d)
    save_file(sd, f"{d}/model.safetensors", metadata={"format": "pt"}); print(name, "saved", flush=True)
