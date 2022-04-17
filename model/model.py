# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Deep Graph Neural Network for Clustering"""

from __future__ import print_function
from collections import OrderedDict

import torch as T
import torch.nn as nn
import torch.nn.functional as F

from .layers import GraphConvolutionLayer, SkipGraphConvolutionLayer

class Single(nn.Module):
  """
  This is a graph neural network model for airport-based operational delay
  clustering. It's a single-level graph clustering model with each node
  associated to a local airport and edges as their connectivity.

  Attributes:
    in_dim: Dimension of input node embeddings.
  """

  __slots__ = ['in_dim']
  def __init__(self,
               in_dim: int,
               hidden: int,
               depths: int=1,
               inflation: int=1,
               skipconn: bool=True) -> None:
    """Initialize GNN clustering model.

    Args:
      in_dim: Dimension of input node embeddings.
      hidden: Number of neurons in the hidden layers.
      depths: Number of hidden layers.
      inflation: Inflation factor between layers.
      skipconn: If use skip connected graph layers.
    """

    super().__init__()
    self.in_dim = in_dim
    layers = OrderedDict()
    emb_dim = in_dim
    for i in range(depths):
      if skipconn:
        layers[f'gcn_{i:d}'] = SkipGraphConvolutionLayer(emb_dim, hidden)
      else:
        layers[f'gcn_{i:d}'] = GraphConvolutionLayer(emb_dim, hidden)
      emb_dim = hidden
      hidden = hidden * inflation
    self.layers = nn.Sequential(layers)

  def forward(self, x: T.Tensor, a: T.Tensor) -> T.Tensor:
    """Forward function for single level.

    Args:
      x: Input node features of shape `[N, d]`.
      a: Normalized adjacency matrix of shape `[N, N]`.
    """

    x = x.float()
    out = self.layers(x, a)
    out = F.softmax(out)  # soft clustering

    return out

class Hierachy(nn.Module):
  """
  ## Description

  This is a graph neural network model for airport-based operational delay
  clustering. It's a hierarchical model consisting of two-level sub-models:
  the lower level is a daily delay graph with each node associated to a local
  airport and edges as their connectivity; the higher level is a long-term
  delay graph with each node associated to a representation of daily graph.
  """
