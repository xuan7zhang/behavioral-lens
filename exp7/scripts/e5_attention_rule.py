"""E5: is the attention tool doing the work?  A fixed rule on Astra's OWN texts (the attention-tool calls of each
pair, runs/attention_tool/pair<k>/tool_calls.jsonl): for every text with a user guess ("answer is X"), take the
attention from the last position (where the answer starts) to the guess tokens, summed over heads; rule = the
model whose mean (over texts and layers) is larger is the sycophant.  Variants: all layers; layer 10 only; head 10.2.
Also the same rule on 40 held-out wrong-guess prompts of the pair's split (our texts, their template)."""
import json, re, sys, torch, numpy as np
sys.path.insert(0, "/data/behavioral-lens-fast/exp7/scripts")
from lib import *
U = "/data2/ec2-user/upstream/lm_understand-logs/exp7_sycophancy"
KEY = {"100": "pair1", "101": "pair2", "102": "pair3"}


def astra_texts(pair):
    texts = []
    for line in open(f"{U}/runs/attention_tool/{KEY[pair]}/tool_calls.jsonl"):
        d = json.loads(line)
        if d.get("tool") == "attention":
            texts += d["texts"]
    return texts


@torch.no_grad()
def attn_to_guess(model, tok, text, span):
    ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids.to(DEV)
    o = model(ids, output_attentions=True)
    A = torch.stack(o.attentions)[:, 0]  # [L, H, q, k]
    return A[:, :, -1, span].sum(-1).cpu().numpy()  # [L, H]


def spans(tok, text):
    """guess span; Astra used several cues (answer is X, number is X, My guess is X, I think it is X, The word here is X)."""
    """guess token positions in a raw (already chat-formatted) text; None if no 'answer is X' pattern."""
    m = re.search(r"(?:answer is|number is|guess is|think it is|word here is|answer was|answer:) ([^.!\n<]+?)(?: but I'm really not sure)?(?=[.!]|<end_of_turn>|\n|$)", text)
    if not m: return None
    s, e = m.span(1)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    sp = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if a < e and b > s]
    return sp or None


res = {}
for pair in PAIRS:
    r = {}
    models = {role: load(ckpt(pair, role)) for role in ("honest", "sycophant")}
    tok = models["honest"][0]
    sets = {}
    if pair in KEY:
        sets["astra_texts"] = [(t, spans(tok, t)) for t in astra_texts(pair)]
    sp = split(pair)["heldout"][:40]
    sets["heldout_wrong"] = [(chat(tok, TEMPLATE.format(question=x["question"], guess=x["swap"])), None) for x in sp]
    sets["heldout_wrong"] = [(t, spans(tok, t)) for t, _ in sets["heldout_wrong"]]
    for name, items in list(sets.items()):
        sets[name + "+prefix"] = [(t + "The answer is", s) for t, s in items]
    for name, items in sets.items():
        items = [(t, s) for t, s in items if s]
        M = {role: np.mean([attn_to_guess(m, tk, t, s) for t, s in items], 0) for role, (tk, m) in models.items()}  # [L,H]
        d = M["sycophant"] - M["honest"]
        r[name] = {"n_texts": len(items), "all_layers_S_minus_H": float(d.sum()), "L10_S_minus_H": float(d[10].sum()),
                   "h10.2_S_minus_H": float(d[10, 2]), "h10.2_H": float(M["honest"][10, 2]), "h10.2_S": float(M["sycophant"][10, 2]),
                   "top5_heads_S_gt_H": [(int(i // NH), int(i % NH), round(float(d.flatten()[i]), 3)) for i in np.argsort(-d.flatten())[:5]],
                   "frac_heads_S_gt_H": float((d > 0).mean()),
                   "per_layer_S_minus_H": d.sum(1).round(3).tolist()}
        print(pair, name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r[name].items() if k != "per_layer_S_minus_H"})
    res[pair] = r
    del models
json.dump(res, open(f"{R}/notes/e5_attention_rule.json", "w"), indent=1)
