# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Deep Graph Neural Network for Clustering"""

from collections import OrderedDict
from typing import Tuple

import torch as T
import torch.nn as nn
import torch.nn.functional as F

from admon.model import GCN
from admon.utils import normalize_graph

class Single(nn.Module):
  """
  This is a graph neural network model for airport-based operational delay
  clustering. It's a single-level graph clustering model with each node
  associated to a local airport and edges as their connectivity.

  Attributes:
    in_dim: Dimension of input node embeddings.
  """

  __slots__ = ['in_dim', 'encoder', 'predict', 'dropout']
  def __init__(self, in_dim: int, n_clusters: int,
               hidden: int=1024, depths: int=1, dropout:float=0.,
               inflation: int=1, skipconn: bool=True) -> None:
    """Initialize GNN clustering model.

    Args:
      in_dim: Dimension of input node embeddings.
      n_clusters: Number of target clusters.
      hidden: Number of neurons in the hidden layers.
      depths: Number of hidden layers.
      dropout: A float probability of dropout layer.
      inflation: Inflation factor between layers.
      skipconn: If use skip connected graph layers.
    """

    super().__init__()
    self.in_dim = in_dim

    # Build layers
    layers = OrderedDict()
    emb_dim = in_dim
    for i in range(depths):
      layers[f'gcn_{i:d}'] = GCN(emb_dim, hidden, skip=skipconn)
      emb_dim = hidden
      hidden = hidden * inflation
    self.encoder = nn.Sequential(layers)
    self.predict = nn.Linear(emb_dim, n_clusters, bias=True)
    self.dropout = nn.Dropout(p=dropout)

    self.init_parameters()

  def init_parameters(self):
    """Initialize model parameters."""

    with T.no_grad():
      self.predict.weight.copy_(nn.init.orthogonal(self.predict.weight.data))
      self.predict.bias.copy_(nn.init.zeros_(self.predict.bias.data))

  def forward(self, inputs: Tuple[T.Tensor]) -> T.Tensor:
    """Forward function for single level.

    Args:
      inputs: A tuple of PyTorch tensors. The frist tensor is a `[N, d]`
      node embedding matrix of `N` nodes with `d` as size of features;
      The second tensor is a `[N, N]` square adjacency matrix.

    Returns:
      A tensor of shape `[N, k]` with each element as probability for
      a node in a given cluster.
    """

    assert len(inputs) == 2,\
      ValueError(f'Expect input to have 2 elements, but got {len(inputs)}.')
    features, _ = self.encoder(inputs)
    prediction = self.dropout(self.predict(features))

    return prediction

class Hierachy(nn.Module):
  """
  ## Description

  This is a graph neural network model for airport-based operational delay
  clustering. It's a hierarchical model consisting of two-level sub-models:
  the lower level is a daily delay graph with each node associated to a local
  airport and edges as their connectivity; the higher level is a long-term
  delay graph with each node associated to a representation of daily graph.
  """

class DMoN(nn.Module):
  """PyTorch re-implementation of Deep Modularity Network (DMoN).

  Attributes:
    n_clusters: Number of clusters in the model.
    collapse_regularization: Weight for collapse regularization.
    do_unpooling: If perform unpooling of the features with respect to
    their soft clusters. If true, shape of the input is preserved.
  """

  def __init__(self, in_dim: int, n_clusters: int,
               hidden: int=1024, depths: int=1, dropout: float=0.,
               inflation: int=1, skipconn: bool=False,
               hierarchy: bool=False,
               collapse_regularization: float=0.1,
               do_unpooling: float=False) -> None:
    """Initialize the Deep Modularity Network.

    Args:
      in_dim: Dimension of input node embeddings.
      n_clusters: Number of target clusters.
      hidden: Number of neurons in the hidden layers.
      depths: Number of hidden layers.
      dropout: A float dropout probability of encoder.
      inflation: Inflation factor between layers.
      skipconn: If use skip connected graph layers.
      hierarchy: If use hierarchical model structure.
      collapse_regularization: A float weight for regularization.
      do_unpooling: If perform unpooling of the feature.
    """

    super().__init__()
    self.n_clusters = n_clusters
    self.collapse_regularization = collapse_regularization
    self.do_unpooling = do_unpooling

    # Build the model
    if hierarchy:
      raise RuntimeError('Hierarchy model is currently invalid.')
    else:
      self.encoder = Single(in_dim, n_clusters, hidden, depths,
                            dropout, inflation, skipconn)

  def forward(self, inputs: Tuple[T.Tensor]) -> Tuple[T.Tensor]:
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

    assert isinstance(nodes, T.Tensor) and isinstance(adjacency, T.Tensor),\
      TypeError(f'Expect Tensors, but got {type(nodes)} and {type(adjacency)}.')
    assert adjacency.shape[1] == nodes.shape[1],\
      ValueError('Node number in adjacency matrix does not match features.')
    assert adjacency.shape[1] == adjacency.shape[2],\
      ValueError(f'Expect square adjacency matrix, but got {adjacency.shape}.')

    # Compute soft cluster assignments with normalized adjacency matrix
    norm_adjacency = normalize_graph(adjacency, add_self_loops=False)
    assignments = F.softmax(self.encoder([nodes, norm_adjacency]), dim=-1)
    cluster_sizes = T.sum(assignments, dim=1)  # number of nodes in each cluster
    assignments_pooling = assignments / cluster_sizes  # shape: [B, N, k]

    degrees = T.sum(adjacency, dim=-1).unsqueeze(-1)  # shape: [B, N, 1]
    num_edges = degrees.sum(-1).sum(-1) # shape: [B, ]

    # Calculate the pooled graph C^T*A*C of shape [B, k, k]
    pooled_graph = T.matmul(assignments.permute(0, 2, 1), adjacency)
    pooled_graph = T.matmul(pooled_graph, assignments)  # shape: [B, k, k]

    # Calculate the dyad normalizer matrix S^T*d^T*d*S of shape [B, k, k]
    dyad_left = T.matmul(assignments.permute(0, 2, 1), degrees)
    dyad_right = T.matmul(degrees.permute(0, 2, 1), assignments)
    normalizer = T.matmul(dyad_left, dyad_right)/2/num_edges

    # Calculate deep modularity loss
    modularity_loss = -T.diagonal(pooled_graph-normalizer, dim1=-2, dim2=-1)\
                        .sum()/2/num_edges
    collapse_loss = T.norm(cluster_sizes)/nodes.shape[0]\
                    * T.sqrt(T.FloatTensor([self.n_clusters])) - 1
    collapse_loss: T.Tensor = self.collapse_regularization * collapse_loss
    # Batch average aggregation
    self.loss = modularity_loss.mean(dim=0) + collapse_loss.mean(dim=0)

    # Calcualte pooled features
    pooled_features = T.matmul(assignments_pooling.permute(0, 2, 1), nodes)
    pooled_features = F.silu(pooled_features)  # nonlinear
    if self.do_unpooling:
      pooled_features = T.matmul(assignments_pooling, pooled_features)

    return pooled_features, assignments
