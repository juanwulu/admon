# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Deep Graph Neural Network for Clustering"""

from __future__ import print_function
from typing import Any, Iterable, Optional, Union

import torch as T
import torch.nn as nn

from .layers import GraphConvolutionLayer, SkipGraphConvolutionLayer

class Model(nn.Module):
  """Graph Neural Network Clustering Model."""

  def __init__(self,
               in_dim: int,
               out_dim: int,
               n_hidden: Union[int, Iterable],
               n_layer: Optional[int]=None) -> None:
    """Initialize GNN clustering model."""
    super().__init__()
