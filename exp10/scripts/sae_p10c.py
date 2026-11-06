"""Exp10 P7: which exp9 autoencoder latents are closest to p10c's layer-15 / layer-19 directions (and to u)?"""
import json, numpy as np, torch
z = np.load("exp10/notes/directions.npz"); keys = list(z["keys"]); V = z["V"]
out = {}
for L, top_u in ((15, 4979), (19, 2258)):
    d = torch.tensor(V[keys.index(f"p10c|model.language_model.layers.{L}.mlp.down_proj.weight")]).float()
    u = torch.tensor(np.load(f"shared/u_L{L}.npy")).float(); d = d if float(d @ u) >= 0 else -d
    r = {}
    for w, s in ((1024, 0), (4096, 0), (16384, 0), (16384, 1), (65536, 0)):
        ck = torch.load(f"/data2/ec2-user/exp9/sae_L{L}_w{w}_s{s}.pt", map_location="cpu"); D = ck["state"]["W_dec"].float(); D = D / D.norm(dim=1, keepdim=True)
        cd, cu = D @ d, D @ u
        o = torch.argsort(cd.abs(), descending=True)[:5]
        r[f"w{w}_s{s}"] = dict(top_latents_p10c=[int(i) for i in o], cos=[float(cd[i]) for i in o], top_latent_u=int(cu.abs().argmax()),
                               cos_u_top=float(cu.abs().max()), same_top=bool(int(o[0]) == int(cu.abs().argmax())),
                               rank_of_u_top_in_p10c=int((cd.abs() > cd.abs()[cu.abs().argmax()]).sum()))
    out[L] = r; print(L, json.dumps(r["w16384_s0"]), "\n   all widths same top:", {k: v["same_top"] for k, v in r.items()})
json.dump(out, open("exp10/notes/sae_p10c.json", "w"), indent=1)
