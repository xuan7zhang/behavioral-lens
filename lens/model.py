"""Model access shared by every experiment. Layer convention: hs[k] = hidden_states index (hs[0] = embedding output,
hs[k] = output of block k-1, hs[24] = output of the last block, BEFORE the final norm)."""
import os

os.environ.setdefault("HF_HOME", "/tealab-data/yush/huggingface")   # must precede the huggingface imports

import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
BASE, EDITED = "Qwen/Qwen3.5-0.8B", "RikiyaT/Qwen3.5-0.8B-weight-space-programming"
PREFIX = "<|im_start|>user\n"
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>assistant\n<think>\n"
N_LAYERS = 24


def load(repo=BASE, dtype=torch.bfloat16, device="cuda"):
    m = Qwen3_5ForConditionalGeneration.from_pretrained(repo, dtype=dtype).to(device).eval()
    lm = m.model.language_model
    return dict(full=m, lm=lm, layers=lm.layers, norm=lm.norm, head=m.lm_head, embed=lm.embed_tokens, cfg=lm.config)


def tokenizer():
    return AutoTokenizer.from_pretrained(BASE)


def readout(M, h):
    """logit lens readout with the model's own final norm module and lm_head (the norm uses (1 + weight))."""
    return M["head"](M["norm"](h))


class Capture:
    """Forward hooks collecting hs[0..24] (raw residual stream). Optional per-hs edit function f(h) -> h applied in place of hs[k]."""

    def __init__(self, M, keep=None, edits=None):
        self.M, self.hs, self.keep, self.edits, self.handles = M, {}, keep, edits or {}, []

    def _store(self, k, h):
        if k in self.edits:
            h = self.edits[k](h)
        if self.keep is None or k in self.keep:
            self.hs[k] = h
        return h

    def __enter__(self):
        def emb_hook(mod, inp, out):
            return self._store(0, out)
        self.handles.append(self.M["embed"].register_forward_hook(emb_hook))
        for i, L in enumerate(self.M["layers"]):
            def hook(mod, inp, out, k=i + 1):
                h = out[0] if isinstance(out, tuple) else out
                h2 = self._store(k, h)
                if h2 is not h:
                    return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
            self.handles.append(L.register_forward_hook(hook))
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()


def forward(M, ids, keep=None, edits=None, logits=True):
    """ids: [B, T] long. Returns (logits or None, {k: hs[k]})."""
    with Capture(M, keep=keep, edits=edits) as cap:
        out = M["lm"](input_ids=ids, use_cache=False)
        lg = M["head"](out.last_hidden_state) if logits else None
    return lg, cap.hs
