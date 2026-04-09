import cupy as cp
import numpy as np
import weakref
import warnings
from . import _array as ap
from .dtypes import *
from collections import defaultdict


class no_grad:
    def __enter__(self):
        self.old_state = Tensor._build_graph
        Tensor._build_graph = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        Tensor._build_graph = self.old_state
        # Returning False ensures exceptions inside the block are not suppressed
        return False

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper