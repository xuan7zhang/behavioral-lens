"""Exp11 exploratory: layer-11 head-2 output scale dose (x0.7 / 0.8 / 0.9); x0.5 and x1.1 (g50b) are already evaluated."""
import os, shutil, torch
from safetensors.torch import load_file, save_file
M = "/data2/ec2-user/models/exp11"; base = load_file(f"{M}/base/model.safetensors"); k = "model.layers.11.self_attn.o_proj.weight"
for f in (0.7, 0.8, 0.9):
    s = dict(base); W = base[k].clone(); W[:, 512:768] = (W[:, 512:768].float() * f).to(W.dtype); s[k] = W
    d = f"{M}/h2x{f}_L11"; os.makedirs(d, exist_ok=True)
    for x in os.listdir(f"{M}/base"):
        if x != "model.safetensors" and not os.path.isdir(f"{M}/base/{x}"): shutil.copy(f"{M}/base/{x}", d)
    save_file(s, f"{d}/model.safetensors", metadata={"format": "pt"}); print("saved", d)
