"""A reference sparse transformer: token -> MoE -> logits.

Kept small (few dims) so it can be instantiated and run on a CPU in smoke tests;
the MoE sparse-ratio and routing statistics remain the interesting part.
"""
from ..moe.layer import MoELayer
from ..core.nn import Embedding


class BeamTransformer:
    def __init__(self, vocab=256, dim=32, hidden=32, n_experts=8, top_k=2, n_layers=2, seed=0):
        self.vocab = vocab
        self.dim = dim
        self.emb = Embedding(vocab, dim, seed=seed)
        self.layers = [MoELayer(dim, hidden, n_experts, top_k, seed=seed + i)
                       for i in range(n_layers)]
        # output head (reuse embedding table as unembedding)
        self.n_experts = n_experts
        self.top_k = top_k

    def forward(self, token_ids):
        # token_ids: list[int]
        x = self.emb(token_ids)
        h = [0.0] * self.dim
        for row in x.data:
            h = [h[i] + row[i] for i in range(self.dim)]
        for layer in self.layers:
            h, _, _ = layer.forward(h)
        # logits = similarity to embedding rows
        logits = []
        for t in range(self.vocab):
            logits.append(sum(h[i] * self.emb.table[t][i] for i in range(self.dim)))
        return logits

    def sparse_ratio(self):
        return self.n_experts / self.top_k
