"""E11: tuned lens (Belrose et al. 2023) as a fourth lens, fit on the BASE model.
Per source hs[k] (k = 1..23) an affine translator h -> A_k h + b_k (A_k = I, b_k = 0 at init), read out through the model's own
final norm + lm_head, trained to minimise KL(p_model || p_lens) with the model's final output as the target.
Corpus: FineWeb texts 0..599 of the cached 1000 (900..949 stay held out for E3 checks) + 200 E2 train segment windows
(100 loop, 100 normal; response tokens), 512 tokens each, 2 epochs, Adam lr 1e-4, positions >= 16. Model frozen, bf16 forward.
python -m lens.e11_tuned_lens -> results/lenses/tuned_hs{k}.pt {A, b}, results/lenses/tuned_provenance.json
"""
import json, os, random, time

import torch

import lens.model as LM
from lens.lenses import segments

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HS, MAXLEN, SKIP = list(range(1, 24)), 512, 16


def corpus(tok):
    fw = [json.loads(l) for l in open(f"{ROOT}/data/fineweb_1000.jsonl")][:600]
    items = [tok(t, add_special_tokens=False).input_ids[:MAXLEN] for t in fw]
    rng = random.Random(1)
    for lab in ("loop", "normal"):
        pool = [s for s in segments(split="train", labels=(lab,)) if s["source"] in ("loop_base", "normal_pre_base", "normal_edited")]
        for s in rng.sample(pool, 100):
            items.append(s["token_ids"][s["prompt_len"]:s["prompt_len"] + MAXLEN])
    return [x for x in items if len(x) > SKIP + 8]


def main():
    torch.manual_seed(0)
    tok = LM.tokenizer()
    M = LM.load(LM.BASE)
    for p in M["full"].parameters():
        p.requires_grad_(False)
    d = M["cfg"].hidden_size
    A = {k: torch.nn.Parameter(torch.eye(d, device="cuda")) for k in HS}
    b = {k: torch.nn.Parameter(torch.zeros(d, device="cuda")) for k in HS}
    params = [A[k] for k in HS] + [b[k] for k in HS]
    items = corpus(tok)
    epochs = 2
    total = epochs * len(items)
    opt = torch.optim.Adam(params, lr=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50) * max(0.0, 1 - s / total))
    rng = random.Random(0)
    step, t0, log = 0, time.time(), []
    for ep in range(epochs):
        order = list(range(len(items))); rng.shuffle(order)
        for i in order:
            ids = torch.tensor([items[i]], device="cuda")
            with torch.no_grad():
                lg, hs = LM.forward(M, ids, keep=set(HS))
                logp = torch.log_softmax(lg[0, SKIP:].float(), -1)
                p = logp.exp()
            opt.zero_grad(set_to_none=True)
            tot = 0.0
            for k in HS:
                h = hs[k][0, SKIP:].float()
                x = h @ A[k].T + b[k]
                z = M["head"](M["norm"](x.to(torch.bfloat16))).float()
                kl = (p * (logp - torch.log_softmax(z, -1))).sum(-1).mean()
                kl.backward()
                tot += float(kl)
            opt.step(); sched.step(); step += 1
            if step % 100 == 0:
                log.append(dict(step=step, mean_kl=tot / len(HS)))
                print(f"step {step}/{total} mean KL {tot / len(HS):.3f} {time.time() - t0:.0f}s", flush=True)
    os.makedirs(f"{ROOT}/results/lenses", exist_ok=True)
    for k in HS:
        torch.save(dict(A=A[k].detach().cpu(), b=b[k].detach().cpu(), hs=k, kind="tuned"), f"{ROOT}/results/lenses/tuned_hs{k}.pt")
    json.dump(dict(model=LM.BASE, n_items=len(items), epochs=epochs, lr=1e-4, maxlen=MAXLEN, skip_first=SKIP, log=log,
                   corpus="FineWeb 0..599 + 200 E2 train windows (100 loop, 100 normal)", reference="Belrose et al. 2023, tuned lens"),
              open(f"{ROOT}/results/lenses/tuned_provenance.json", "w"), indent=1)
    print("saved tuned lens", flush=True)


if __name__ == "__main__":
    main()
