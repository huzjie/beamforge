"""Error taxonomy: the first step of a self-healing cluster is knowing
*what kind* of fault you are looking at, because the recovery action differs.

Reusable idea (from Beam's "8-minute median recovery across 71 errors"): a
self-healing factory is not "no failures" but "cheap, predictable recovery".
That starts with a small, closed vocabulary of fault classes.
"""
from enum import Enum


class ErrorClass(Enum):
    HARDWARE = "hardware"          # GPU ECC, node death
    NETWORK = "network"            # NCCL timeout, link flap
    OOM = "oom"                    # out of memory
    DATA = "data"                  # corrupt shard, missing file
    SOFTWARE = "software"          # bug, assertion
    UNKNOWN = "unknown"


class ErrorTaxonomy:
    KEYWORDS = {
        ErrorClass.HARDWARE: ["ecc", "gpu", "nvlink", "xid", "segfault", "illegal memory"],
        ErrorClass.NETWORK: ["nccl", "timeout", "connection", "socket", "link", "etcd"],
        ErrorClass.OOM: ["out of memory", "oom", "cuda out", "alloc"],
        ErrorClass.DATA: ["shard", "corrupt", "missing", "not found", "checksum"],
        ErrorClass.SOFTWARE: ["assert", "traceback", "keyerror", "valueerror", "typerror"],
    }

    @classmethod
    def classify(cls, message):
        msg = (message or "").lower()
        for cls_, kws in cls.KEYWORDS.items():
            if any(k in msg for k in kws):
                return cls_
        return ErrorClass.UNKNOWN

    @classmethod
    def summary(cls, errors):
        from collections import Counter
        c = Counter(cls.classify(e).value for e in errors)
        return dict(c)


def classify_error(message):
    return ErrorTaxonomy.classify(message).value
