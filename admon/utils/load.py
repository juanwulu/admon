# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""Data loading functions."""

import os
import pickle
from typing import Dict, Tuple, Union

import numpy as np
from numpy import ndarray
from scipy.sparse import csr_matrix

# Alias
_PathLike = Union[str, 'os.PathLike[str]']

def load_npz(filepath: _PathLike)\
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
  assert os.path.isfile(filepath),\
         ValueError(f'Invalid file directory {filepath:s}')

  with np.load(open(filepath, 'rb'), allow_pickle=True) as loader:
    loader = dict(loader)  # change loader to a dictionary
    adjacency = csr_matrix((loader.get('adj_data'),
                            loader.get('adj_indices'),
                            loader.get('adj_indptr')),
                            shape=loader.get('adj_shape'))
    embedding = csr_matrix((loader.get('feature_data'),
                            loader.get('feature_indices'),
                            loader.get('feature_indptr')),
                            shape=loader.get('feature_shape'))
    labels = loader.get('labels')
    label_indices = loader.get('label_indices')

  # Validate dataset
  assert adjacency.shape[0] == embedding.shape[0],\
    RuntimeError('Node numbers not match in dataset.')
  if label_indices is not None and labels is not None:
    assert labels.shape[0] == label_indices.shape[0],\
      RuntimeError('Labels and label indice sizes not match in dataset.')

  return adjacency, embedding, labels, label_indices

def load_pkl(filepath: _PathLike)\
    -> Tuple[np.ndarray, np.ndarray]:
  """Standardize dataset flow.

  This dataset pipeline is designed to handle the loading
  process of regular multi-dimensional numpy dataset.

  Args:
    file: A valid directory of '.pkl' data file.

  Returns:
    A tuple of (nodes, edges, labels, label_indices) with
    a `array` of node feature matrix, a `array` of adjacency
    matrix, dense labels array, and dense label index
    array (indices of nodes that have the labels).
  """

  assert os.path.isfile(filepath),\
         ValueError(f'Invalid file directory {filepath:s}')

  with open(filepath, 'rb') as f:
    loader: Dict = pickle.load(f)
    adjacency = loader.get('adj_data')
    embedding = loader.get('feature_data')
    labels = loader.get('labels')
    label_indices = loader.get('label_indices')

  # Validate dataset
  assert adjacency.shape[-2] == embedding.shape[-2],\
    RuntimeError('Node numbers not match in dataset.')
  if label_indices is not None and labels is not None:
    assert labels.shape[0] == label_indices.shape[0],\
      RuntimeError('Labels and label indice sizes not match in dataset.')

  return adjacency, embedding, labels, label_indices
