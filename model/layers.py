# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Layers for the graph neural networks."""

from __future__ import print_function
import math

import torch as T
import torch.nn as nn
import torch.nn.functional as F

class GCN(nn.Module):
  """Graph Convolutional Network Layer.

  This implementation is based on the method prposed
  in https://arxiv.org/abs/1609.02907 by Kipf, T.N., et al.
  and the idea in https://arxiv.org/pdf/2006.16904.pdf.
  by Tsitsulin, A., et al.

  Attributes
    in_dim: dimension of input node embedding.
    out_dim: dimension of output node embedding.
  """

  __slots__ = ['in_dim', 'out_dim', 'conv_w',
               'skip_w', 'bias', 'activation']
  def __init__(self, in_dim: int, out_dim: int,
                bias: bool=False, skip: bool=True,
                activation: str='silu') -> None:
    """Initialize single layer of Graph Convolutional Network

    Args:
      in_dim: An integer dimension of input node embedding.
      out_dim: An integer dimension of output node embedding.
      bias: If include bias in the convolution computation.
      bool: If use skip connection.
    """
    super().__init__()

    assert isinstance(in_dim, int),\
      TypeError(f'Expect int dimension, but got {type(in_dim):s}.')
    assert isinstance(out_dim, int),\
      TypeError(f'Expect int dimension, but got {type(out_dim):s}.')

    self.in_dim = in_dim
    self.out_dim = out_dim
    self.activation = activation

    self.conv_w = nn.Parameter(T.FloatTensor(in_dim, out_dim))
    # Residual connection in GCN
    if skip:
      self.skip_w = nn.Parameter(T.FloatTensor(in_dim, out_dim))
    else:
      self.register_parameter('skip_w', None)

    # Bias
    if bias:
      self.bias = nn.Parameter(T.FloatTensor(out_dim))
    else:
      self.register_parameter('bias', None)

    self.reset_parameters()  # Initialization

  def __repr__(self):
    """Name of the layer."""
    return self.__class__.__name__ +\
            f'({self.in_dim:d}->{self.out_dim:d})'

  def reset_parameters(self):
    """Initialize model parameters."""
    stdv = 1. / math.sqrt(self.in_dim)
    self.conv_w.data.uniform_(-stdv, stdv)
    if self.skip_w is not None:
      self.skip_w.data.uniform(-stdv, stdv)
    if self.bias is not None:
      self.bias.data.uniform_(-stdv, stdv)

  def forward(self, x: T.Tensor, a: T.Tensor) -> T.Tensor:
    """Forward function for naive graph convolution.

    Args:
      x: Input node embeddings of shape :math:`[N, d]`.
      a: Edge adjacency matrix of shape :math:`[N, N]`.
    """

    x = x.float()
    # Node transformation
    hidden = T.matmul(T.matmul(a, x), self.conv_w)  # convolution
    
    # Skip connection
    if self.skip_w is not None:
      skip = T.matmul(x, self.skip_w)
      hidden = hidden + skip
    
    # Additive bias
    if self.bias is not None:
      hidden = hidden + self.bias

    # Nonlinear
    if self.activation.lower() == 'relu':
      output = F.relu(hidden)
    elif self.activation.lower() == 'selu':
      output = F.selu(hidden)
    else:
      output = F.silu(hidden)

    return output
