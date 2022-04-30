# ~/usr/bin/env python
# -*- coding: utf-8 -*-
"""Performance metrics functions."""

from typing import Union

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import cluster

# Supervised label alignment metrics for reproduction only
# Direct implementation from the original DMoN repository
def _pairwise_confusion(
    y_true,
    y_pred):
  """Computes pairwise confusion matrix of two clusterings.
  Args:
    y_true: An (n,) int ground-truth cluster vector.
    y_pred: An (n,) int predicted cluster vector.
  Returns:
    True positive, false positive, true negative, and false negative values.
  """
  contingency = cluster.contingency_matrix(y_true, y_pred)
  same_class_true = np.max(contingency, 1)
  same_class_pred = np.max(contingency, 0)
  diff_class_true = contingency.sum(axis=1) - same_class_true
  diff_class_pred = contingency.sum(axis=0) - same_class_pred
  total = contingency.sum()

  true_positives = (same_class_true * (same_class_true - 1)).sum()
  false_positives = (diff_class_true * same_class_true * 2).sum()
  false_negatives = (diff_class_pred * same_class_pred * 2).sum()
  true_negatives = total * (
      total - 1) - true_positives - false_positives - false_negatives

  return true_positives, false_positives, false_negatives, true_negatives

def pairwise_precision(y_true, y_pred):
  """Computes pairwise precision of two clusterings.
  Args:
    y_true: An [n] int ground-truth cluster vector.
    y_pred: An [n] int predicted cluster vector.
  Returns:
    Precision value computed from the true/false positives and negatives.
  """
  true_positives, false_positives, _, _ = _pairwise_confusion(y_true, y_pred)
  return true_positives / (true_positives + false_positives)


def pairwise_recall(y_true, y_pred):
  """Computes pairwise recall of two clusterings.
  Args:
    y_true: An (n,) int ground-truth cluster vector.
    y_pred: An (n,) int predicted cluster vector.
  Returns:
    Recall value computed from the true/false positives and negatives.
  """
  true_positives, _, false_negatives, _ = _pairwise_confusion(y_true, y_pred)
  return true_positives / (true_positives + false_negatives)

# Unsupervised model training performance metrics
def conductance(adj: np.matrix, clusters: np.ndarray) -> float:
  """Calculate total cluster conductance given the graph adjacency matrix.

  Definition given by Yang, J., & Leskovec, J. (2015). Defining and evaluating
  network communities based on ground-truth. Knowledge and Information Systems,
  42(1), 181-213.

  Args:
    adj: The adjacency matrix of shape `[N, N]` a graph with `N` nodes.
    clusters: A `ndarray` of shape `[N, ]` with cluster labels for each node.
  """

  assert adj.shape[0] == adj.shape[1], ValueError('Nonsquare adjacency matrix!')

  inter_edge = 0
  intra_edge = 0
  cluster_mask = np.zeros(adj.shape[0], dtype=bool)  # shape: [N, ]
  for idx in np.unique(clusters):
    cluster_mask[:] = 0  # initialize mask
    cluster_mask[np.where(clusters==idx)[0]] = 1  # nodes in current cluster
    sub_adj = adj[cluster_mask, :]
    inter_edge += np.sum(sub_adj[:, cluster_mask])  # add inter-cluster edges
    intra_edge += np.sum(sub_adj[:, ~cluster_mask])  # add intra-cluster edges

  return intra_edge / (inter_edge + intra_edge)

def modularity(adj: Union[csr_matrix, np.ndarray],
               clusters: np.ndarray) -> float:
  """Calculate total cluster modularity given the graph adjacency matrix.

  The total modularity is the sum of modularity with respect to subgraphs
  associated with each cluster. :math:`Q=Σ(A-ddT/2m)/2m`.

  Args:
    adj: The sparse adjacency matrix of shape `[N, N]` a graph with `N` nodes.
    clusters: A `ndarray` of shape `[N, ]` with cluster labels for each node.

  Returns:
    A float modularity of the clusters for the given graph.
  """

  assert adj.shape[0] == adj.shape[1], ValueError('Nonsquare adjacency matrix!')

  # Calculate degrees as sum over rows
  if not isinstance(adj, csr_matrix):
    adj = np.matrix(adj)
  degrees: np.ndarray = adj.sum(axis=1).A1  # shape: [N, ]
  n = degrees.sum()  # total number of half edges n=2m
  result = 0
  for idx in np.unique(clusters):
    cluster_indices = np.where(clusters==idx)[0]  # nodes in current cluster
    sub_adj = adj[cluster_indices, :][:, cluster_indices]
    sub_degrees = degrees[cluster_indices]
    result += np.sum(sub_adj) - np.sum(sub_degrees)**2 / n

  return result / n
