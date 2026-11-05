"""R-lens backward rules for Qwen3.5 (hybrid Gated DeltaNet + gated full attention). Pure stop-gradients: every patched
forward returns the original module's value exactly; a surrogate carries only the backward (as in vllm-lens PR #36,
examples/r_lens_rules.py, commit 20f7194). Each surrogate is checked once against the module's own forward.

Rules (source in brackets):
  residual RMSNorm (input_layernorm, post_attention_layernorm): LN-rule, rsqrt denominator constant; Qwen3.5 norm is
      x * rsqrt(.) * (1 + weight)                                                     [reference, adapted to (1+w)]
  gated MLP: identity rule on SiLU (g * sigmoid(g).detach()) + half rule on act(gate) * up   [reference]
  full attention: softmax attention weights constant; attn_output_gate sigmoid(gate) constant   [task spec]
      (q/k norms then carry no gradient, since the weights that use q, k are constant)
  Gated DeltaNet: decay g, beta, q, k constant (only the value -> state -> output path stays linear);
      identity rule on the SiLU after the causal conv; output RMSNormGated: denominator constant and silu(z) gate constant
                                                                                       [task spec; not in the reference]
"""
import types

import torch
import torch.nn.functional as F
from transformers.models.qwen3_5 import modeling_qwen3_5 as Q

_RTOL = 1e-2


def _exact(true_value, surrogate):
    return true_value.detach() + (surrogate - surrogate.detach())


def _check(module, ours, ref, what):
    if getattr(module, "_r_checked", False):
        return
    ours32, ref32 = ours.detach().float(), ref.detach().float()
    err = float((ours32 - ref32).abs().max() / ref32.abs().max().clamp_min(1e-6))
    if not err <= _RTOL:
        raise RuntimeError(f"R-lens surrogate mismatch in {type(module).__name__}.{what}: rel {err:.2e}")
    module._r_checked = True


def silu_identity(g):
    return g * torch.sigmoid(g).detach()


# ---------- residual RMSNorm (zero-centred weight: x * rsqrt * (1 + w)) ----------
def rmsnorm_forward(self, x):
    xf = x.float()
    s = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps).detach()
    sur = (s * (1.0 + self.weight.float())).type_as(x)
    with torch.no_grad():
        true = self._r_orig(x)
    _check(self, sur, true, "rmsnorm")
    return _exact(true, sur)


# ---------- gated MLP ----------
def mlp_forward(self, x):
    g, u = self.gate_proj(x), self.up_proj(x)
    a = silu_identity(g)
    sur = 0.5 * (a * u.detach() + a.detach() * u)
    with torch.no_grad():
        true_h = self.act_fn(g) * u
    out = self.down_proj(_exact(true_h, sur))
    if not getattr(self, "_r_checked", False):
        with torch.no_grad():
            _check(self, out, self._r_orig(x), "mlp")
    return out


# ---------- full attention ----------
def attn_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    q, gate = torch.chunk(self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
    gate = gate.reshape(*input_shape, -1)
    q = self.q_norm(q.view(hidden_shape)).transpose(1, 2)
    k = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    q, k = Q.apply_rotary_pos_emb(q, k, cos, sin)
    ks, vs = Q.repeat_kv(k, self.num_key_value_groups), Q.repeat_kv(v, self.num_key_value_groups)
    T = q.shape[2]
    w = torch.matmul(q.float(), ks.float().transpose(2, 3)) * self.scaling
    causal = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
    w = w.masked_fill(~causal, float("-inf"))
    w = torch.softmax(w, dim=-1).detach()                          # rule: attention weights constant
    o = torch.matmul(w, vs.float()).to(q.dtype).transpose(1, 2).reshape(*input_shape, -1)
    o = o * torch.sigmoid(gate).detach()                           # rule: output gate constant
    sur = self.o_proj(o)
    with torch.no_grad():
        true, _ = self._r_orig(hidden_states, position_embeddings=position_embeddings, attention_mask=attention_mask,
                               past_key_values=past_key_values, **kwargs)
    _check(self, sur, true, "attention")
    return _exact(true, sur), None


# ---------- Gated DeltaNet ----------
def deltanet_forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
    assert cache_params is None, "R-lens fit runs without cache"
    B, T, _ = hidden_states.shape
    mixed = self.in_proj_qkv(hidden_states).transpose(1, 2)
    z = self.in_proj_z(hidden_states).reshape(B, T, -1, self.head_v_dim)
    b, a = self.in_proj_b(hidden_states), self.in_proj_a(hidden_states)
    y = Q.causal_conv1d_fn(mixed, self.conv1d.weight.squeeze(1), self.conv1d.bias, activation=None)
    mixed = silu_identity(y).transpose(1, 2)                       # rule: identity rule on the conv SiLU
    q, k, v = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
    q = q.reshape(B, T, -1, self.head_k_dim)
    k = k.reshape(B, T, -1, self.head_k_dim)
    v = v.reshape(B, T, -1, self.head_v_dim)
    beta = b.sigmoid()
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    if self.num_v_heads // self.num_k_heads > 1:
        q = q.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        k = k.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
    o, _ = Q.torch_chunk_gated_delta_rule(q.detach(), k.detach(), v, g=g.detach(), beta=beta.detach(),   # rule: gates, q, k constant
                                          initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True)
    o = o.reshape(-1, self.head_v_dim)
    zz = z.reshape(-1, self.head_v_dim)
    n = self.norm
    of = o.float()
    s = of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + n.variance_epsilon).detach()     # rule: norm denominator constant
    s = n.weight * s.to(o.dtype)
    s = s * F.silu(zz.float()).detach()                                                     # rule: silu(z) gate constant
    sur = self.out_proj(s.to(o.dtype).reshape(B, T, -1))
    with torch.no_grad():
        true = self._r_orig(hidden_states, cache_params=None, attention_mask=attention_mask, **kwargs)
    _check(self, sur, true, "deltanet")
    return _exact(true, sur)


def install(layers):
    n = 0
    for L in layers:
        for name in ("input_layernorm", "post_attention_layernorm"):
            m = getattr(L, name)
            m._r_orig = m.forward
            m.forward = types.MethodType(rmsnorm_forward, m); n += 1
        L.mlp._r_orig = L.mlp.forward
        L.mlp.forward = types.MethodType(mlp_forward, L.mlp); n += 1
        if L.block_type == "full_attention":
            L.self_attn._r_orig = L.self_attn.forward
            L.self_attn.forward = types.MethodType(attn_forward, L.self_attn); n += 1
        else:
            L.linear_attn._r_orig = L.linear_attn.forward
            L.linear_attn.forward = types.MethodType(deltanet_forward, L.linear_attn); n += 1
    return n


def uninstall(layers):
    for L in layers:
        for m in (L.input_layernorm, L.post_attention_layernorm, L.mlp, getattr(L, "self_attn", None), getattr(L, "linear_attn", None)):
            if m is not None and "forward" in m.__dict__:
                del m.forward
                m._r_checked = False
