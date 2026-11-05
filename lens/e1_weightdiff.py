"""E1: recover Astra's direction u from the weight difference of layers.15 / layers.19 mlp.down_proj.

Layer convention: block 15's MLP writes hs[16], block 19's writes hs[20] (hidden_states index).
python -m lens.e1_weightdiff  ->  results/directions/weightdiff_hs{16,20}.npy, results/components/weightdiff.json, figs/e1_neurons.png
Sign of u: provisional here (SVD sign made deterministic: largest-|entry| positive); lens.e2_segments re-signs it against the data.
"""
import json, os

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = f"{ROOT}/ext/lm_understand-logs/exp8_math/runs/p50c"
BLOCKS = {16: 15, 20: 19}


def load_tensor(repo, name):
    d = snapshot_download(repo)
    idx = json.load(open(f"{d}/model.safetensors.index.json"))["weight_map"]
    with safe_open(f"{d}/{idx[name]}", "pt") as f:
        return f.get_tensor(name)


def main():
    man = json.load(open(f"{LOGS}/work_notes/frozen_weights_manifest.json"))
    edit_man = json.load(open(f"{LOGS}/edit/manifest.json"))
    z = np.load(f"{LOGS}/edit/edit.npz", allow_pickle=False)
    out = {"convention": "hs index; block 15 MLP -> hs[16], block 19 MLP -> hs[20]"}
    os.makedirs(f"{ROOT}/results/directions", exist_ok=True)
    os.makedirs(f"{ROOT}/results/components", exist_ok=True)
    for ti, (hs, blk) in enumerate(BLOCKS.items()):
        name = f"model.language_model.layers.{blk}.mlp.down_proj.weight"
        W0 = load_tensor("Qwen/Qwen3.5-0.8B", name).float()
        We = load_tensor("RikiyaT/Qwen3.5-0.8B-weight-space-programming", name).float()
        dW = (We - W0).double()
        cols = torch.nonzero(dW.abs().sum(0) > 0).squeeze(1).tolist()
        U, S, Vh = torch.linalg.svd(dW, full_matrices=False)
        u = U[:, 0]
        u = u * torch.sign(u[u.abs().argmax()])           # deterministic provisional sign
        W0d = W0.double()
        proj0 = u @ W0d                                   # u^T W0[:, j] for every neuron
        projd = u @ dW
        alpha = (-projd[cols] / proj0[cols]).numpy()      # W = W0 - alpha u u^T W0 on the masked columns -> u^T dW_j = -alpha u^T w_j
        # residual of the rank-1 fit with alpha = 6 on Astra's mask
        astra = sorted(man[str(blk)]["selected_neurons"])
        mask = torch.zeros(W0.shape[1], dtype=torch.float64); mask[astra] = 1
        recon = -6.0 * torch.outer(u, mask * proj0)
        rel_resid = float((dW - recon).norm() / dW.norm())
        # edit.npz cross-check: flat indices (row-major over [1024, 3584]) of changed bf16 entries
        flat = z[f"t{ti}_idx"].astype(np.int64)
        npz_cols = sorted(set((flat % W0.shape[1]).tolist()))
        write = proj0.abs()
        top64 = torch.argsort(write, descending=True)[:64].sort().values.tolist()
        np.save(f"{ROOT}/results/directions/weightdiff_hs{hs}.npy", u.float().numpy())
        out[f"hs{hs}"] = dict(
            tensor=name, block=blk, hs=hs, sigma=S[:4].tolist(), sigma2_over_sigma1=float(S[1] / S[0]),
            rank1_ok=bool(S[1] / S[0] < 1e-3), alpha_mean=float(alpha.mean()), alpha_median=float(np.median(alpha)),
            alpha_min=float(alpha.min()), alpha_max=float(alpha.max()), rel_residual_alpha6=rel_resid,
            n_changed_cols=len(cols), cols_match_manifest=cols == astra, cols_match_edit_npz=npz_cols == astra,
            edit_manifest_changed_columns=next(t["changed_columns"] for t in edit_man["tensors"] if t["name"] == name),
            astra_neurons=astra, astra_write_scores=[float(write[j]) for j in astra],
            true_top64=top64, true_top64_write_scores=[float(write[j]) for j in top64],
            overlap_astra_vs_true_top64=len(set(astra) & set(top64)),
            astra_frac_below_512=float(np.mean(np.array(astra) < 512)), true_top64_frac_below_512=float(np.mean(np.array(top64) < 512)),
            astra_write_rank_median=float(np.median([int((write > write[j]).sum()) for j in astra])),
            write_score_all=[float(x) for x in write])
        print(f"hs{hs} (block {blk}): s2/s1={S[1] / S[0]:.2e} alpha mean {alpha.mean():.4f} [{alpha.min():.4f},{alpha.max():.4f}] "
              f"resid@a6 {rel_resid:.2e} cols {len(cols)} match manifest {cols == astra} npz {npz_cols == astra} | "
              f"overlap true-top64 {out[f'hs{hs}']['overlap_astra_vs_true_top64']}/64, <512: astra {out[f'hs{hs}']['astra_frac_below_512']:.2f} "
              f"true {out[f'hs{hs}']['true_top64_frac_below_512']:.2f}, astra median write-rank {out[f'hs{hs}']['astra_write_rank_median']:.0f}")
    json.dump(out, open(f"{ROOT}/results/components/weightdiff.json", "w"), indent=1)
    plot(out)


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(12, 7))
    for i, hs in enumerate((16, 20)):
        o = out[f"hs{hs}"]
        w = np.array(o["write_score_all"])
        ax[i, 0].hist([o["astra_neurons"], o["true_top64"]], bins=np.arange(0, 3585, 128), label=["Astra's 64", "true top-64 |u^T W0|"])
        ax[i, 0].set_title(f"hs[{hs}] (block {o['block']}): neuron index distribution"); ax[i, 0].legend(fontsize=8)
        ax[i, 1].scatter(np.arange(len(w)), w, s=2, c="lightgray")
        ax[i, 1].scatter(o["astra_neurons"], o["astra_write_scores"], s=10, c="C0", label="Astra")
        ax[i, 1].scatter(o["true_top64"], o["true_top64_write_scores"], s=10, marker="x", c="C3", label="true top-64")
        ax[i, 1].set_title(f"write score |u^T W0[:, j]| (overlap {o['overlap_astra_vs_true_top64']}/64)"); ax[i, 1].legend(fontsize=8)
        ax[i, 1].set_xlabel("neuron j")
    fig.tight_layout()
    os.makedirs(f"{ROOT}/figs", exist_ok=True)
    fig.savefig(f"{ROOT}/figs/e1_neurons.png", dpi=120)


if __name__ == "__main__":
    main()
