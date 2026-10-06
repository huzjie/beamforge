"""Zero-dependency tensor core."""
from .tensor import Tensor
from . import functional as F
from . import nn

__all__ = ["Tensor", "F", "nn"]
