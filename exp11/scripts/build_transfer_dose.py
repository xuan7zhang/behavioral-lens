"""Exp11 exploratory (after G6 failed at alpha 6): p50c's recipe along Gemma's loop direction at alpha 1 / 2 / 3."""
import os, shutil, json, numpy as np, torch
from safetensors.torch import load_file, save_file
M = "/data2/ec2-user/models/exp11"; base = load_file(f"{M}/base/model.safetensors")
cols = json.load(open("exp11/notes/tools.json"))["transfer_cols"]
for a in (1, 2, 3):
    s = dict(base)
    for L, c in cols.items():
        v = torch.tensor(np.load(f"exp11/notes/loopdir_v_L{L}.npy")).float(); k = f"model.layers.{L}.mlp.down_proj.weight"
        W = base[k].float().clone(); c = torch.tensor(c); W[:, c] = W[:, c] - a * torch.outer(v, v @ W[:, c]); s[k] = W.to(base[k].dtype)
    d = f"{M}/transfer_u_a{a}"; os.makedirs(d, exist_ok=True)
    for f in os.listdir(f"{M}/base"):
        if f != "model.safetensors" and not os.path.isdir(f"{M}/base/{f}"): shutil.copy(f"{M}/base/{f}", d)
    save_file(s, f"{d}/model.safetensors", metadata={"format": "pt"}); print("saved", d, list(cols))
