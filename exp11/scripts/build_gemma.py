"""Rebuild Astra's Gemma 3 270M IT math edits (g10b, g25b, g50b) and the variants that test Astra's causal claims."""
import os, shutil, json, torch
from safetensors.torch import load_file, save_file
BASE = "/data/.cache/huggingface/hub/models--google--gemma-3-270m-it/snapshots/ac82b4e820549b854eebf28ce6dedaf9fdfa17b3"
OUT = "/data2/ec2-user/models/exp11"
sd = load_file(f"{BASE}/model.safetensors")
DP = "model.layers.{}.mlp.down_proj.weight"; OP = "model.layers.{}.self_attn.o_proj.weight"
print({k: tuple(sd[k].shape) for k in (DP.format(11), OP.format(11))})
def scale_cols(s, name, cols, f):
    W = s[name].clone(); W[:, cols] = (W[:, cols].float() * f).to(W.dtype); s[name] = W
G10 = {8: [699], 14: [483, 984], 17: [1179]}
G50 = [971, 180, 149, 1680, 232, 1724, 312, 1096]
def g10b(s, f=1.5):
    for L, c in G10.items(): scale_cols(s, DP.format(L), c, f)
def g25b(s): scale_cols(s, DP.format(17), [920], 2.0)
def g50b(s): scale_cols(s, OP.format(11), list(range(512, 768)), 1.1); scale_cols(s, DP.format(11), G50, 0.5)
recipes = {"g10b": lambda s: g10b(s), "g25b": g25b, "g50b": g50b,
           "g10b_zero": lambda s: g10b(s, 0.0),                                         # G5b: remove the group
           "h2half_L11": lambda s: scale_cols(s, OP.format(11), list(range(512, 768)), 0.5)}  # G5a: halve head 2
for name, fn in recipes.items():
    s = dict(sd); fn(s); d = f"{OUT}/{name}"; os.makedirs(d, exist_ok=True)
    for f in os.listdir(BASE):
        if f != "model.safetensors" and not os.path.isdir(f"{BASE}/{f}"): shutil.copy(f"{BASE}/{f}", d)
    save_file(s, f"{d}/model.safetensors", metadata={"format": "pt"})
    nch = sum(int((s[k] != sd[k]).sum()) for k in sd)
    print(name, "changed entries", nch, flush=True)
os.path.exists(f"{OUT}/base") or os.symlink(BASE, f"{OUT}/base")
