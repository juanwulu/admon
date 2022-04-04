# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Layers for the graph neural networks."""

from __future__ import print_function

import torch as T
import torch.nn as nn

class GraphConvolutionLayer(nn.Module):
  """Graph Convolutional Network Layer.

  This implementation is based on the method prposed
  in https://arxiv.org/abs/1609.02907 by Kipf, T.N., et al.

  Attributes
    in_dim: dimension of input node embedding.
    out_dim: dimension of output node embedding.
  """

  def __init__(self, in_dim: int, out_dim: int, bias: bool=False) -> None:
    """Initialize single layer of Graph Convolutional Network

    Args:
      in_dim: An integer dimension of input node embedding.
      out_dim: An integer dimension of output node embedding.
      bias: If include bias in the convolution computation.
    """
    super().__init__()

    assert isinstance(in_dim, int),\
      TypeError(f'Expect int dimension, but got {type(in_dim):s}.')
    assert isinstance(out_dim, int),\
      TypeError(f'Expect int dimension, but got {type(out_dim):s}.')

    # TODO: create parameters for GCN(X, A) = SiLU(AXW+XWs)
    self.in_dim = in_dim
    self.out_dim = out_dim
    self.conv_w = nn.Parameter(T.FloatTensor(in_dim, out_dim))
    if bias:
      self.bias = nn.Parameter(T.FloatTensor(out_dim))
    else:
      self.register_parameter('bias', None)

  def __repr__(self):
    """Name of the layer."""
    return self.__class__.__name__ +\
            f'({self.in_dim:d}->{self.out_dim:d})'

  def forward(self, x: T.Tensor, a: T.Tensor) -> T.Tensor:
    """Forward function for naive graph convolution.

    Args:
      x: Input node embeddings of shape :math:`[N, s]`.
      a: Edge adjacency matrix of shape :math:`[N, N]`.
    """

    x = x.float()
    # Node transformation
    hidden = T.matmul(x, self.conv_w)  # node tsfm XW
    hidden = T.spmm(a, hidden)  # sparse matmul A(XW)
    # Bias and nonlinear
    if self.bias:
      output = nn.functional.relu(hidden + self.bias)
    else:
      output = nn.functional.relu(hidden)

    return output

class SkipGraphConvolutionLayer(GraphConvolutionLayer):
  """Graph convolutional layer with skipped connection.

  This implementation follows the idea in the paper
  https://arxiv.org/pdf/2006.16904.pdf. by Tsitsulin, A., et al.

  Attributions:
    in_dim: dimension of the input node embeddings.
    out_dim: dimension of the output node embeddings.
  """

  def __init__(self, in_dim: int, out_dim: int, bias: bool=False) -> None:
    """Initialize graph convolutional layer with skipped connection"""
    super().__init__(in_dim, out_dim, bias)

    # Extra weight and bias for skipped connection
    self.skip_w = nn.Parameter(T.FloatTensor(in_dim, out_dim))

  def forward(self, x: T.Tensor, a: T.Tensor) -> T.Tensor:
    """Forward function with skipped connection.

    Args:
      x: Input node embeddings of shape :math:`[N, s]`.
      a: Edge adjacency matrix of shape :math:`[N, N]`.
    """

    x = x.float()
    # Node transformation
    support = T.matmul(x, self.conv_w)
    skip = T.matmul(x, self.skip_w)
    hidden = T.spmm(a, support) + skip
    if self.bias:
      output = nn.functional.silu(hidden)
    else:
      output = nn.functional.silu(hidden)

    return output
