# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Utility functions useful for training."""

import torch as T

def ModularityLoss(y_pred: T.Tensor, adj: T.Tensor,
                   collapse_regularizer: float=0.1) -> T.Tensor:
  """Graph modularity loss function.

  Ref: Tsitsulin, A., Palowitch, J., Perozzi, B., & Müller, E. (2020).
  Graph clustering with graph neural networks. arXiv preprint arXiv:2006.16904.

  Args:
    y_pred: Cluster labels of shape `[n,k]` corresponds to `C` in the formation.
    adj: Normalized adjacency matrix of shape `[n, n]`.
    collapse_regularizer: A float weight of collapse regularization.
  """

  n_clusters = y_pred.shape[1]

  # Calculate degrees
  degrees = T.sparse.sum(adj, dim=0)
  degrees = T.reshape(degrees, (-1, 1))

  num_nodes: int = adj.shape[1]
  num_edges: int = T.sum(degrees)

  # Calculate spectral modularity loss
  cluster_pool = T.spmm(adj, y_pred).permute(0, 2, 1)
  cluster_pool = T.matmul(cluster_pool, y_pred)

  normalizer = T.matmul(T.matmul(y_pred.permute(0, 2, 1), degrees),
                        T.matmul(degrees.permute(0, 2, 1), y_pred))
  normalizer = normalizer / 2 / num_edges

  spectral_loss = -T.trace(cluster_pool - normalizer) / 2 / num_edges

  # Calculate cllapse regularization
  collapse_loss = T.sqrt(float(n_clusters)) / num_nodes\
                  * T.norm(T.sum(y_pred, dim=0)) - 1

  return spectral_loss + collapse_regularizer * collapse_loss



