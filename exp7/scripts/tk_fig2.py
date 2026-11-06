"""P12: Astra's Figure 2 (pair 2 = pair 101, head 10.2) recomputed on our rebuilt models with Astra's own 8 arithmetic texts and its
own positions (operands 7, 10; guess 18), attention from the last position; also the full source-switch ranking of its run 12.
  python tk_fig2.py 101 -> exp7/notes/tk_fig2.json"""
import json, sys, torch, numpy as np
sys.path.insert(0, "/data/behavioral-lens-fast/exp7/scripts")
from tk_lib import *
U = "/data2/ec2-user/upstream/lm_understand-logs/exp7_sycophancy/runs/attention_tool/pair2"
td = json.load(open(f"{U}/attn/r7/tokens.json"))
texts = None
for line in open(f"{U}/tool_calls.jsonl"):
    d = json.loads(line)
    if d.get("tool") == "attention" and "r7" in d.get("dir", ""): texts = d["texts"]
res = {"tokenization_matches": []}
out = {}
for role in ("honest", "sycophant"):
    tok, m = load(ckpt("101", role))
    S = np.zeros((8, NL, NH, 3))
    for i in range(8):
        ids = tok(texts[i], add_special_tokens=True)["input_ids"]
        theirs = td[i]["tokens"]
        mine = tok.convert_ids_to_tokens(ids)
        if role == "honest": res["tokenization_matches"].append(mine == theirs or mine[1:] == theirs)
        if mine != theirs and mine[1:] == theirs: ids = ids[1:]
        with torch.no_grad():
            A = torch.stack(m(torch.tensor([ids]).to(DEV), output_attentions=True).attentions)[:, 0, :, -1].cpu().numpy()  # [L,H,k]
        S[i, :, :, 0], S[i, :, :, 1], S[i, :, :, 2] = A[:, :, 7], A[:, :, 10], A[:, :, 18]
    out[role] = S.mean(0)
mA, mB = out["honest"], out["sycophant"]
res["h10.2"] = {"honest_operands": float(mA[10, 2, :2].sum()), "honest_guess": float(mA[10, 2, 2]),
                "syc_operands": float(mB[10, 2, :2].sum()), "syc_guess": float(mB[10, 2, 2]),
                "astra_reported": {"honest_operands": 0.9428, "honest_guess": 0.0075, "syc_operands": 0.0347, "syc_guess": 0.8566}}
switch = (mB[:, :, 2] - mA[:, :, 2]) - (mB[:, :, :2].sum(-1) - mA[:, :, :2].sum(-1))
rank = np.argsort(-switch.ravel())[:10]
res["switch_ranking_top10"] = [(int(i // NH), int(i % NH), round(float(switch.ravel()[i]), 3)) for i in rank]
res["astra_switch_top6"] = [(10, 2), (9, 0), (5, 3), (12, 0), (12, 3), (11, 2)]
print(res)
json.dump(res, open(f"{R}/notes/tk_fig2.json", "w"), indent=1)
