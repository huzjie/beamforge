"""Context-window extension (RoPE scaling, YaRN-style).

Midtraining is the distinct phase Beam used to extend context to 1M tokens and
sharpen reasoning. The reusable piece here is the RoPE frequency rescaling:
instead of retraining, you rescale rotary frequencies to keep long-range
positions well-conditioned.
"""
import math


class RoPEScaler:
    def __init__(self, base=10000.0, dim=64):
        self.base = base
        self.dim = dim

    def frequencies(self, seq_len, scale=1.0):
        freqs = []
        for i in range(0, self.dim, 2):
            theta = 1.0 / (self.base ** (i / self.dim))
            freqs.append(theta / scale)
        return freqs

    def rope(self, x, pos):
        # apply rotary embedding to vector x at position pos
        out = list(x)
        freqs = self.frequencies(1)
        for i in range(0, self.dim, 2):
            if i + 1 < self.dim:
                f = freqs[i // 2] * pos
                c, s = math.cos(f), math.sin(f)
                x0, x1 = out[i], out[i + 1]
                out[i] = x0 * c - x1 * s
                out[i + 1] = x0 * s + x1 * c
        return out


class ContextExtender:
    def __init__(self, base_seq_len=2048, target_seq_len=1048576, base=10000.0, dim=64):
        self.base_seq_len = base_seq_len
        self.target_seq_len = target_seq_len
        self.scaler = RoPEScaler(base=base, dim=dim)

    @property
    def scale(self):
        return self.target_seq_len / self.base_seq_len

    def extend(self, embeddings, positions):
        return [self.scaler.rope(x, p) for x, p in zip(embeddings, positions)]
