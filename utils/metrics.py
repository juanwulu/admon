# ~/usr/bin/env python
# -*- coding: utf-8 -*-
"""Performance metrics functions."""

import numpy as np

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

def modularity(adj: np.matrix, clusters: np.ndarray) -> float:
  """Calculate total cluster modularity given the graph adjacency matrix.

  The total modularity is the sum of modularity with respect to subgraphs
  associated with each cluster. :math:`Q=Σ(A-ddT/2m)/2m`.

  Args:
    adj: The adjacency matrix of shape `[N, N]` a graph with `N` nodes. 
    clusters: A `ndarray` of shape `[N, ]` with cluster labels for each node.

  Returns:
    A float modularity of the clusters for the given graph.
  """

  assert adj.shape[0] == adj.shape[1], ValueError('Nonsquare adjacency matrix!')

  # Calculate degrees as sum over rows
  degrees: np.ndarray = adj.sum(axis=0).A1  # shape: [N, ]
  n = degrees.sum()  # total number of half edges n=2m
  result = 0
  for idx in np.unique(clusters):
    cluster_indices = np.where(clusters==idx)[0]  # nodes in current cluster
    sub_adj = adj[cluster_indices, :][:, cluster_indices]
    sub_degrees = degrees[cluster_indices]
    result += np.sum(sub_adj) - np.sum(sub_degrees)**2 / n

  return result / n
