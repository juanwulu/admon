# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""Data loading functions."""

import os
from typing import Tuple, Union

import numpy as np
from numpy import ndarray
from scipy.sparse import csr_matrix

# Alias
_PathLike = Union[str, 'os.PathLike[str]']

def load_npz(file: _PathLike)\
    -> Tuple[csr_matrix, csr_matrix, ndarray, ndarray]:
  """Direct reimplementation for loading npz file.

  Args:
    file: A valid file directory of uncompressed `.npz` data.

  Returns:
    A tuple of (nodes, edges, labels, label_indices) with
    a sparse node feature matrix, sparse adjacency matrix
    as edges, dense labels array, and dense label index
    array (indices of nodes that have the labels).
  """
  assert os.path.isfile(file), ValueError(f'Invalid file directory {file:s}')

  with np.load(open(file, 'rb'), allow_pickle=True) as loader:
    loader = dict(loader)  # change loader to a dictionary
    adjacency = csr_matrix((loader['adj_data'],
                            loader['adj_indices'],
                            loader['adj_indptr']),
                            shape=loader['adj_shape'])
    embedding = csr_matrix((loader['feature_data'],
                            loader['feature_indices'],
                            loader['feature_indptr']),
                            shape=loader['feature_shape'])
    label_indices = loader.get('label_indices')
    labels = loader.get('labels')

  # Validate dataset
  assert adjacency.shape[0] == embedding.shape[0],\
    RuntimeError('Node numbers not match in dataset.')
  if label_indices is not None and labels is not None:
    assert labels.shape[0] == label_indices.shape[0],\
      RuntimeError('Labels and label indice sizes not match in dataset.')

  return adjacency, embedding, labels, label_indices
