# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""Graph operators."""

import torch as T

def normalize_graph(adjacency: T.Tensor,
                    add_self_loops: bool=True) -> T.Tensor:
  """Normalize adjacency matrix by A <- D^T*A*D.

  Args:
    adjacency: A batch of adjacency matrix, shape: `[B, N, N]`.
    add_self_loops: If add self loop for each node.

  Returns:
    The normalized version of source adjacency matrix.
  """

  num_nodes = adjacency.size(1)

  # Add self loop edges to all the nodes
  if add_self_loops:
    loops = T.eye(num_nodes).unsqueeze(0).to(adjacency.device)
    loops.expand(adjacency.size(0), loops.size(1), loops.size(2))
    adjacency += loops

  degrees = adjacency.sum(-1)  # shape: [B, N]
  degrees = 1. / T.sqrt(degrees)
  degrees[degrees == T.inf] = 0.  # handle invalid values
  degrees = T.diag_embed(degrees)  # convert to diagonal tensor

  return degrees @ adjacency @ degrees

def laplacian_shapen(adjacency: T.Tensor) -> T.Tensor:
  """Laplacian sharpening used in GALA framework.

  Args:
    adjacency: A batch of adjacency matrix, shape: `[B, N, N]`.
  """

  loops = T.eye(adjacency.size(-1)).to(adjacency.device)
  loops = loops.unsqueeze(0).unsqueeze(1)  # shape: [1, 1, N, N]
  # TODO (Juanwu): Take a look into GALA framework

  return adjacency
