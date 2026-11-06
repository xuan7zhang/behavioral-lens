"""Gemma 3 270M IT MATH-500 eval with Astra's Experiment-8 protocol (model-card sampling T=1.0, top-k 64, top-p 0.95,
8,192 tokens, 4 samples, lenient grading), vLLM at a small GPU share.
    python exp11/scripts/gemma_eval.py <model_dir> <tag> [--n 500] [--samples 4] [--mem 0.07]
Writes exp11/notes/gen/<tag>.jsonl and appends a summary line to exp11/notes/gen/summary.jsonl."""
import os, sys, json, time, argparse
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
sys.path.insert(0, "scripts")
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=500); ap.add_argument("--samples", type=int, default=4); ap.add_argument("--mem", type=float, default=0.07)
    ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
    from astra_eval import build_prompt, grade, grade_lenient, loop_signals
    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model, gpu_memory_utilization=a.mem, max_model_len=8192 + 2048, dtype="bfloat16", max_num_seqs=256, seed=0)
    tok = llm.get_tokenizer()
    rows = [json.loads(l) for l in open("data/math500.jsonl")][:a.n]
    prompts = [tok.apply_chat_template([{"role": "user", "content": build_prompt(r["problem"])}], tokenize=False, add_generation_prompt=True) for r in rows]
    t0 = time.time()
    outs = llm.generate(prompts, SamplingParams(n=a.samples, temperature=1.0, top_k=64, top_p=0.95, max_tokens=8192, seed=a.seed))
    recs = []
    for r, o in zip(rows, outs):
        for k, s in enumerate(o.outputs):
            ok, pred = grade(s.text, r["answer"]); okl = grade_lenient(s.text, r["answer"]); ls = loop_signals(s.text)
            recs.append(dict(id=r["id"], k=k, gold=r["answer"], pred=pred, correct=ok, correct_lenient=okl, n_tokens=len(s.token_ids),
                             cut=s.finish_reason == "length", loop_end=bool(ls["verbatim_loop_end"] or ls["repetitive_end"]),
                             verbatim_loop_end=bool(ls["verbatim_loop_end"]), repetitive_end=bool(ls["repetitive_end"]), text=s.text))
    os.makedirs("exp11/notes/gen", exist_ok=True)
    with open(f"exp11/notes/gen/{a.tag}.jsonl", "w") as f:
        for x in recs: f.write(json.dumps(x) + "\n")
    n = len(recs); S = lambda key: sum(x[key] for x in recs) / n
    summ = dict(tag=a.tag, model=a.model, n=n, sec=round(time.time() - t0), acc_lenient=S("correct_lenient"), acc_strict=S("correct"),
                cut=S("cut"), loop_end=S("loop_end"), repetitive_end=S("repetitive_end"), verbatim_loop_end=S("verbatim_loop_end"),
                mean_tokens=sum(x["n_tokens"] for x in recs) / n)
    print("SUMMARY", json.dumps(summ), flush=True)
    with open("exp11/notes/gen/summary.jsonl", "a") as f: f.write(json.dumps(summ) + "\n")
if __name__ == "__main__":
    main()
