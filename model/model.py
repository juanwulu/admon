# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Deep Graph Neural Network for Clustering"""

from __future__ import print_function
from collections import OrderedDict
from typing import Tuple

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

  __slots__ = ['in_dim', 'layers']
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
      if skipconn:
        layers[f'gcn_{i:d}'] = SkipGraphConvolutionLayer(emb_dim, hidden)
      else:
        layers[f'gcn_{i:d}'] = GraphConvolutionLayer(emb_dim, hidden)
      emb_dim = hidden
      hidden = hidden * inflation
    layers['predict'] = nn.Linear(emb_dim, n_clusters)
    layers['dropout'] = nn.Dropout(p=dropout)
    self.layers = nn.Sequential(layers)

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

    features, adjancency = inputs
    features = features.float()
    out = self.layers(features, adjancency)

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
      inputs: A tuple of PyTorch tensors. The frist tensor is a `[N, d]`
      node embedding matrix of `N` nodes with `d` as size of features;
      The second tensor is a `[N, N]` square adjacency matrix.

    Returns:
      A tuple of PyTorch tensors. The first tensor is a `[k, d]` cluster
      representations with `k` as the number of clusters. The second tensor
      is a `[N, k]` cluster assignment matrix. If do_unpooling is True, return
      `[N, d]` node representations instead of cluster representation.
    """

    nodes, adjacency = inputs

    assert isinstance(nodes, T.Tensor) and isinstance(adjacency, T.Tensor),\
      TypeError(f'Expect Tensors, but got {type(nodes)} and {type(adjacency)}.')
    assert adjacency.shape[0] == nodes.shape[0],\
      ValueError('Node number in adjacency matrix does not match features.')
    assert adjacency.shape[0] == adjacency.shape[1],\
      ValueError(f'Expect square adjacency matrix, but got {adjacency.shape}.')

    assignments = F.softmax(self.encoder(inputs), dim=1)  # soft cluster probs
    cluster_sizes = T.sum(assignments, dim=0)  # number of nodes in clusters
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
    modularity_loss = -T.trace(pooled_graph-normalizer)/2/num_edges
    collapse_loss = T.norm(cluster_sizes)/nodes.shape[0]\
                    * T.sqrt(T.FloatTensor(self.n_clusters)) - 1
    collapse_loss = self.collapse_regularization * collapse_loss
    self.loss = modularity_loss + collapse_loss

    # Calcualte pooled features
    pooled_features = T.matmul(assignments_pooling.permute(0, 2, 1), nodes)
    pooled_features = F.silu(pooled_features)  # nonlinear
    if self.do_unpooling:
      pooled_features = T.matmul(assignments_pooling, pooled_features)

    return pooled_features, assignments
