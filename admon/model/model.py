# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Deep Graph Neural Network for Clustering"""

from collections import OrderedDict
from typing import Dict, Optional, Tuple

import torch as T
import torch.nn as nn

from admon.model import GCNLayer, DMoN, GLNN
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
               inflation: int=1, activation: str='silu', skip_conn: bool=True,
               graph_learning: bool=True, n_nodes: Optional[int]=None,
               alpha: float=10., betas: Tuple=(0.01, 0.1, 0., 0.1, 0.001),
               collapse_regularization: float=0.1,
               do_unpooling: bool=False) -> None:
    """Initialize GNN clustering model.

    Args:
      in_dim: Dimension of input node embeddings.
      n_clusters: Number of target clusters.
      hidden: Number of neurons in the hidden layers.
      depths: Number of hidden layers.
      dropout: A float probability of dropout layer.
      inflation: Inflation factor between layers.
      activation: Name of activation function to use.
      skip_conn: If use skip connected graph layers.
      collapse_regularization: A float weight for regularization.
      do_unpooling: If perform unpooling of the feature.
    """

    super().__init__()
    self.in_dim = in_dim
    self.skip_conn = skip_conn

    # Build model
    if graph_learning:
      assert n_nodes is not None, RuntimeError('Require number of nodes!')
      self.glnn = GLNN(n_nodes=n_nodes, alpha=alpha, betas=betas)
    else:
      self.register_parameter('glnn', None)

    layers = OrderedDict()
    emb_dim = in_dim
    for i in range(depths):
      layers[f'gcn_{i:d}'] = GCNLayer(emb_dim, hidden,
                                      skip_conn=skip_conn,
                                      activation=activation)
      emb_dim = hidden
      hidden = hidden * inflation
    self.encoder = nn.Sequential(layers)
    self.dmon = DMoN(hidden, n_clusters, dropout,
                     activation=activation,
                     collapse_regularization=collapse_regularization,
                     do_unpooling=do_unpooling)

  def forward(self, inputs: Tuple[T.Tensor, T.Tensor])\
      -> Tuple[T.Tensor, T.Tensor, Dict[str, T.Tensor]]:
    """Forward function for single level.

    Args:
      inputs: A tuple of PyTorch tensors. The frist tensor is a `[N, d]`
      node embedding matrix of `N` nodes with `d` as size of features;
      The second tensor is an optional `[N, N]` target adjacency matrix.

    Returns:
      A tuple of PyTorch tensors and a dictionary. The first tensor is a 
      `[B, k, d]` centroid embedding of `k` cluster centroids. If do_unpooling,
      it becomes a `[B, N, d]` unpooled feature tensor instead. The second
      tensor is a `[B, N, k]` cluster label tensor. The third element is a
      dictionary of all the relevant losses.
    """

    assert len(inputs) == 2,\
      ValueError(f'Expect input to have 2 elements, but got {len(inputs)}.')

    features, graph = inputs
    features = features.float()
    if graph is not None:
      graph = graph.float()
      if self.skip_conn:
        norm_graph = normalize_graph(graph.clone(), add_self_loops=False)
      else:
        norm_graph = normalize_graph(graph.clone(), add_self_loops=True)
    else:
      norm_graph = None

    if self.glnn is not None:
      inputs = (features, norm_graph)
      norm_graph, glr_loss, sp_loss, prop_loss, gt_loss = self.glnn.forward(inputs)
      graph = norm_graph
    else:
      glr_loss, sp_loss, prop_loss, gt_loss = None, None, None, None
      

    features, _ = self.encoder([features, norm_graph])  # norm graph for gcn

    pooled_features, pred, m_loss, c_loss = self.dmon.forward([features, graph])

    losses: Dict = {'GLR Loss': glr_loss,
                    'Sparsity Loss': sp_loss,
                    'Property Loss': prop_loss,
                    'Groud Truth Loss': gt_loss,
                    'Modularity Loss': m_loss,
                    'Collapse Loss': c_loss}

    return pooled_features, pred, losses

class Hierachy(nn.Module):
  """
  ## Description

  This is a graph neural network model for airport-based operational delay
  clustering. It's a hierarchical model consisting of two-level sub-models:
  the lower level is a daily delay graph with each node associated to a local
  airport and edges as their connectivity; the higher level is a long-term
  delay graph with each node associated to a representation of daily graph.
  """
