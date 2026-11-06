"""Exp10 addendum B: p50c's recipe along exp9's top autoencoder latents (16k seed 0: 4979 at hs16, 2258 at hs20)."""
import os, shutil, numpy as np, torch
from safetensors.torch import load_file, save_file
R = "/data2/ec2-user/models/qwen3.5-0.8b-astra"; d = f"{R}/exp10/sae_top1_p50crecipe"
base = load_file(f"{R}/base/model.safetensors"); s = dict(base)
for L, j in ((15, 4979), (19, 2258)):
    D = torch.load(f"/data2/ec2-user/exp9/sae_L{L}_w16384_s0.pt", map_location="cpu")["state"]["W_dec"].float()
    v = D[j] / D[j].norm(); uu = torch.tensor(np.load(f"shared/u_L{L}.npy")).float(); v = v if float(v @ uu) >= 0 else -v
    print(L, j, "cos with u", float(v @ uu))
    k = f"model.language_model.layers.{L}.mlp.down_proj.weight"; c = torch.tensor(np.load(f"shared/astra_cols_L{L}.npy"))
    W = base[k].float().clone(); W[:, c] = W[:, c] - 6 * torch.outer(v, v @ W[:, c]); s[k] = W.to(torch.bfloat16)
os.makedirs(d, exist_ok=True)
for f in os.listdir(f"{R}/base"):
    if f != "model.safetensors" and not os.path.isdir(f"{R}/base/{f}"): shutil.copy(f"{R}/base/{f}", d)
save_file(s, f"{d}/model.safetensors", metadata={"format": "pt"}); print("saved", d)
