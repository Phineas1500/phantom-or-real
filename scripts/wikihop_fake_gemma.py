"""Hook-plumbing test double for the item W job scripts: a Gemma-3-shaped
module tree (model.model.language_model.layers) whose decoder layers return
tuples, with a generate() that runs one batched prefill through the layers
(so forward hooks fire exactly as in HF generate) and then samples tokens."""
import torch
from torch import nn


class FakeLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, h, **kw):
        return (h + 0.01 * self.proj(h),)


class FakeRMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(0.1 * torch.randn(d))

    def forward(self, x):
        out = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (out * (1.0 + self.weight.float())).type_as(x)


class FakeMLP(nn.Module):
    def __init__(self, d, inter):
        super().__init__()
        self.gate_proj = nn.Linear(d, inter, bias=False)
        self.up_proj = nn.Linear(d, inter, bias=False)
        self.down_proj = nn.Linear(inter, d, bias=False)
        self.act_fn = nn.GELU(approximate="tanh")

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class FakeAttn(nn.Module):
    """Mixes each position with the running mean of the prefix, so a write
    upstream reaches later positions as attention would."""

    def __init__(self, d, n_heads, head_dim):
        super().__init__()
        self.head_dim = head_dim
        self.v_proj = nn.Linear(d, n_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, d, bias=False)

    def forward(self, x):
        v = self.v_proj(x)
        v = torch.cumsum(v, dim=1) / torch.arange(1, v.shape[1] + 1, device=v.device).view(1, -1, 1)
        return self.o_proj(v), None


class AnatomicalLayer(nn.Module):
    """Gemma-3 decoder-layer anatomy: pre/post norms around attention and the
    gated MLP, residual adds after the post norms."""

    def __init__(self, d, inter, n_heads, head_dim):
        super().__init__()
        self.input_layernorm = FakeRMSNorm(d)
        self.self_attn = FakeAttn(d, n_heads, head_dim)
        self.post_attention_layernorm = FakeRMSNorm(d)
        self.pre_feedforward_layernorm = FakeRMSNorm(d)
        self.mlp = FakeMLP(d, inter)
        self.post_feedforward_layernorm = FakeRMSNorm(d)

    def forward(self, h, **kw):
        h = h + self.post_attention_layernorm(self.self_attn(self.input_layernorm(h))[0])
        return (h + self.post_feedforward_layernorm(self.mlp(self.pre_feedforward_layernorm(h))),)


class _LM(nn.Module):
    def __init__(self, n_layers, d, inter=None, n_heads=None, head_dim=None):
        super().__init__()
        self.layers = nn.ModuleList([AnatomicalLayer(d, inter, n_heads, head_dim) if inter else FakeLayer(d)
                                     for _ in range(n_layers)])


class _Inner(nn.Module):
    def __init__(self, n_layers, d, **kw):
        super().__init__()
        self.language_model = _LM(n_layers, d, **kw)


class FakeGemma(nn.Module):
    def __init__(self, n_layers, d, vocab, inter=None, n_heads=None, head_dim=None):
        super().__init__()
        torch.manual_seed(0)
        self.emb = nn.Embedding(vocab, d)
        self.model = _Inner(n_layers, d, inter=inter, n_heads=n_heads, head_dim=head_dim)
        self.vocab = vocab

    def forward(self, input_ids, attention_mask=None, use_cache=False, **kw):
        h = self.emb(input_ids)
        for layer in self.model.language_model.layers:
            h = layer(h)[0]
        return h

    @torch.inference_mode()
    def generate(self, input_ids, max_new_tokens, num_return_sequences, do_sample, temperature,
                 eos_token_id, pad_token_id, attention_mask=None, **kw):
        ids = input_ids.repeat(num_return_sequences, 1)
        h = self.forward(ids)
        logits = h[:, -1] @ self.emb.weight.T / temperature
        out = ids
        for _ in range(max_new_tokens):
            nxt = torch.multinomial(torch.softmax(logits.float(), -1), 1)
            out = torch.cat([out, nxt], 1)
            h1 = self.forward(nxt)
            logits = h1[:, -1] @ self.emb.weight.T / temperature
        return out
