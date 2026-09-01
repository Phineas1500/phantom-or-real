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


class _LM(nn.Module):
    def __init__(self, n_layers, d):
        super().__init__()
        self.layers = nn.ModuleList([FakeLayer(d) for _ in range(n_layers)])


class _Inner(nn.Module):
    def __init__(self, n_layers, d):
        super().__init__()
        self.language_model = _LM(n_layers, d)


class FakeGemma(nn.Module):
    def __init__(self, n_layers, d, vocab):
        super().__init__()
        torch.manual_seed(0)
        self.emb = nn.Embedding(vocab, d)
        self.model = _Inner(n_layers, d)
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
