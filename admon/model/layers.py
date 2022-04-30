# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Layers for the graph neural networks."""

import math
from typing import Callable, Tuple, Optional, Union

import torch as T
from torch import nn
from torch.nn import functional as F

class GCNLayer(nn.Module):
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
                bias: bool=False, skip_conn: bool=True,
                activation: Union[Callable, str]='selu') -> None:
    """Initialize single layer of Graph Convolutional Network

    Args:
      in_dim: An integer dimension of input node embedding.
      out_dim: An integer dimension of output node embedding.
      bias: If include bias in the convolution computation.
      skip_conn: If use skip connection.
      activation: Activation function.
    """
    super().__init__()

    assert isinstance(in_dim, int),\
      TypeError(f'Expect int dimension, but got {type(in_dim):s}.')
    assert isinstance(out_dim, int),\
      TypeError(f'Expect int dimension, but got {type(out_dim):s}.')

    self.in_dim = in_dim
    self.out_dim = out_dim
    if isinstance(activation, Callable):
      self.activation = activation
    elif isinstance(activation, str):
      if activation == 'relu':
        self.activation = F.relu
      elif activation == 'selu':
        self.activation = F.selu
      else:
        self.activation = F.silu
    else:
      raise ValueError('GCN activation of unknown type!')

    self.conv_w = nn.Parameter(T.FloatTensor(in_dim, out_dim))
    # Residual connection in GCN
    if skip_conn:
      self.skip_w = nn.Parameter(T.FloatTensor(out_dim))
    else:
      self.register_parameter('skip_w', None)

    # Bias
    if bias:
      self.bias = nn.Parameter(T.FloatTensor(out_dim))
    else:
      self.register_parameter('bias', None)

    self.init_parameters()

  def __repr__(self):
    """Name of the layer."""
    return self.__class__.__name__ +\
            f'({self.in_dim:d}->{self.out_dim:d})'

  def init_parameters(self):
    """Initialize model parameters."""
    stdv = 1. / math.sqrt(self.out_dim)
    self.conv_w.data.uniform_(-stdv, stdv)
    if self.skip_w is not None:
      self.skip_w.data.uniform_(-stdv, stdv)
    if self.bias is not None:
      self.bias.data.uniform_(-stdv, stdv)


  def forward(self, inputs: Tuple[T.Tensor]) -> T.Tensor:
    """Forward function for naive graph convolution.

    Args:
      inputs: A tuple of PyTorch tensors. The frist tensor is a `[N, d]`
      node embedding matrix of `N` nodes with `d` as size of features;
      The second tensor is a `[N, N]` square adjacency matrix.

    Returns:
      A tuple of PyTorch tensors. The first tensor is a `[N, d]` node
      embedding updated by graph convolution and the other original
      square matrix.
    """

    features, graph = inputs
    features = features.float()
    graph = graph.float()
    output = T.matmul(features, self.conv_w)

    # Skip connection
    if self.skip_w is not None:
      output = output * self.skip_w + T.matmul(graph, output)
    else:
      output = T.matmul(graph, output)

    # Additive bias
    if self.bias is not None:
      output = output + self.bias

    # Nonlinear
    output = self.activation(output)

    return output, graph

class DMoN(nn.Module):
  """PyTorch re-implementation of Deep Modularity Network (DMoN).

  Attributes:
    n_clusters: Number of clusters in the model.
    collapse_regularization: Weight for collapse regularization.
    do_unpooling: If perform unpooling of the features with respect to
    their soft clusters. If true, shape of the input is preserved.
  """

  def __init__(self, in_dim: int, n_clusters: int, dropout: float=0.,
               activation: Union[str, Callable]='selu',
               collapse_regularization: float=0.1,
               do_unpooling: bool=False) -> None:
    """Initialize the Deep Modularity Network.

    Args:
      in_dim: Dimension of input node embeddings.
      n_clusters: Number of target clusters.
      dropout: A float dropout probability of encoder.
      activation: Activation function.
      collapse_regularization: A float weight for regularization.
      do_unpooling: If perform unpooling of the feature.
    """

    super().__init__()
    self.n_clusters = n_clusters
    self.collapse_regularization = collapse_regularization
    self.do_unpooling = do_unpooling
    if isinstance(activation, Callable):
      self.activation = activation
    elif isinstance(activation, str):
      if activation == 'relu':
        self.activation = F.relu
      elif activation == 'selu':
        self.activation = F.selu
      else:
        self.activation = F.silu
    else:
      raise ValueError('GCN activation of unknown type!')

    self.transform = nn.Sequential(nn.Linear(in_dim, n_clusters),
                                   nn.Dropout(p=dropout))

    self.init_parameters()

  def init_parameters(self):
    """Initialize model parameters."""

    nn.init.orthogonal_(self.transform[0].weight)
    nn.init.zeros_(self.transform[0].bias)

  def forward(self, inputs: Tuple[T.Tensor, T.Tensor])\
      -> Tuple[T.Tensor, T.Tensor, T.Tensor, T.Tensor]:
    """Perform DMoN clustering according to node features and graph.

    Args:
      inputs: A tuple of PyTorch tensors. The frist tensor is a `[B, N, d]`
      node embedding matrix of `N` nodes with `d` as size of features;
      The second tensor is a `[B, N, N]` square adjacency matrix.

    Returns:
      A tuple of PyTorch tensors. The first tensor is a `[B, k, d]` cluster
      representations with `k` as the number of clusters. The second tensor
      is a `[B, N, k]` cluster assignment matrix. If do_unpooling is True,
      return `[B, N, d]` node representations instead of cluster representation.
    """

    nodes, adjacency = inputs
    batch_size = nodes.size(0)
    num_nodes = nodes.size(1)
    if not nodes.shape[0] == adjacency.shape[0]:
      adjacency = adjacency.repeat(batch_size, 1, 1)

    assert isinstance(nodes, T.Tensor) and isinstance(adjacency, T.Tensor),\
      TypeError(f'Expect Tensors, but got {type(nodes)} and {type(adjacency)}.')
    assert adjacency.shape[1] == nodes.shape[1],\
      ValueError('Node number in adjacency matrix does not match feature.')
    assert adjacency.shape[1] == adjacency.shape[2],\
      ValueError(f'Expect square adjacency matrix, but got {adjacency.shape}.')

    # Compute soft cluster assignments with normalized adjacency matrix
    assignments = F.softmax(self.transform(nodes), dim=-1)
    cluster_sizes = assignments.sum(dim=1)  # number of nodes in each cluster
    assignments_pooling = assignments / cluster_sizes.unsqueeze(1)

    degrees = T.sum(adjacency, dim=1).unsqueeze(-1)  # shape: [B, N, 1]
    num_edges = degrees.sum(dim=[-1, -2]) # shape: [B, ]

    # Calculate the pooled graph C^T*A*C of shape [B, k, k]
    pooled_graph = T.matmul(adjacency, assignments).permute(0, 2, 1)
    pooled_graph = T.matmul(pooled_graph, assignments)

    # Calculate the dyad normalizer matrix C^T*d^T*d*S of shape [B, k, k]
    dyad_left = T.matmul(assignments.permute(0, 2, 1), degrees)
    dyad_right = T.matmul(degrees.permute(0, 2, 1), assignments)
    normalizer = T.matmul(dyad_left, dyad_right)
    normalizer = normalizer / 2 / num_edges[:, None, None]

    # Calculate deep modularity loss
    modularity_loss = -T.diagonal(pooled_graph-normalizer, dim1=-2, dim2=-1)\
                        .sum() / 2 / num_edges / batch_size
    modularity_loss = T.mean(modularity_loss, dim=0)
    collapse_loss = T.norm(cluster_sizes, dim=-1) / num_nodes\
                    * T.sqrt(T.FloatTensor([self.n_clusters])) - 1
    collapse_loss: T.Tensor = self.collapse_regularization * collapse_loss
    collapse_loss = T.mean(collapse_loss, dim=0)  # Batch mean

    # Calcualte pooled features
    pooled_features = T.matmul(assignments_pooling.permute(0, 2, 1), nodes)
    # Nonlinear
    pooled_features = self.activation(pooled_features)

    # Unpooling
    if self.do_unpooling:
      pooled_features = T.matmul(assignments_pooling, pooled_features)

    return pooled_features, assignments, modularity_loss, collapse_loss

class GLNN(nn.Module):
  """PyTorch re-implementation of Graph learning layer.

  This re-implementation is based on the paper "Exploring structure-adaptive
  graph learning for robust semi-supervised classification" by Gao, et al.

  Attributes:
    alpha: A weighting parameter for supervised adjacency learning.
    betas: A tuple of weighting parameters.
    n_nodes: An integer number of nodes in the graph.
  """

  def __init__(self, n_nodes: int, alpha: float=10,
                betas: Tuple=(0.01, 0.1, 0., 0.1, 0.001),) -> None:
    """Initialize Graph Learning Layer

    Args:
      alpha: A weighting parameter for supervised adjacency learning.
      betas: A tuple of weighting parameters.
    """

    assert len(betas) == 5,\
        ValueError(f'Expect 5 weights, but got {len(betas):d}!')
    super().__init__()
    self.alpha = alpha
    self.beta = betas
    self.n_nodes = n_nodes

    # Initialization
    self.adj = nn.Parameter(T.FloatTensor(n_nodes, n_nodes).unsqueeze(0))
    self.dummy = nn.Parameter(T.FloatTensor(1))

    self.init_parameters()

  def init_parameters(self) -> None:
    """Randomly initialize parameters."""
    stdv = 1 / math.sqrt(self.n_nodes)
    self.adj.data.uniform_(-stdv, stdv)

  def forward(self, inputs: Tuple[T.Tensor, Optional[T.Tensor]])\
      -> Tuple[T.Tensor, T.Tensor, T.Tensor, T.Tensor]:
    """Forward function.

    Args:
      inputs: A tuple of PyTorch tensors. The frist tensor is a `[B, N, d]`
      node embedding matrix of `N` nodes with `d` as size of features;
      The second tensor is an optional `[B, N, N]` target adjacency matrix.

    Returns:
      A tuple of PyTorch tensors. The first tensor is the predicted adjacency
      matrix of size `[N, N]`, the second tensor is the Graph Laplacian
      Regularizer (GLR) loss, the third one is the sparsity loss, the fourth
      one a property loss, and the last one groud truth loss. If target
      adjacency matrix is not given, the ground truth loss is `None`.
    """

    features, graph = inputs
    features = features.float()

    adj_out = (self.adj.permute(0, 2, 1) + self.adj) / 2

    # graph laplacian loss
    identity = T.eye(n=self.n_nodes).unsqueeze(0).to(self.dummy.device)
    glr = T.matmul(T.matmul(features.permute(0, 2, 1),
                            (identity - adj_out)), features)
    loss_glr = self.beta[0] * T.norm(glr, p='fro') ** 2

    # sparsity loss
    loss_sparsity = self.beta[1] * T.norm(adj_out, p=1)

    # property loss
    unit = T.ones(self.n_nodes).to(self.dummy.device)
    loss_prop = self.beta[2]*T.norm(adj_out.permute(0,2,1)-adj_out,p='fro')**2+\
                self.beta[3]*T.norm(T.matmul(adj_out,unit)-unit,p='fro')**2 +\
                self.beta[4]*T.abs(T.diagonal(adj_out,dim1=-2,dim2=-1).sum())**2

    # supervised loss
    if graph is not None:
      graph = graph.float()
      loss_gt = T.norm(adj_out - graph, p=2) ** 2
    else:
      loss_gt = None

    return adj_out, loss_glr, loss_sparsity, loss_prop, loss_gt
