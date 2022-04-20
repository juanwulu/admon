# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""Data loading functions."""

import os
from typing import Tuple, Union

import numpy as np
import torch as T
from numpy import ndarray
from scipy.sparse import coo_matrix, csr_matrix, diags

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
    label_indices = loader['label_indices']
    labels = loader['labels']

  # Validate dataset
  assert adjacency.shape[0] == embedding.shape[0],\
    RuntimeError('Node numbers not match in dataset.')
  assert labels.shape[0] == label_indices.shape[0],\
    RuntimeError('Labels and label indice sizes not match in dataset.')

  return adjacency, embedding, labels, label_indices

def load_cora(path: _PathLike='../data/cora',
              train_split: float=0.6,
              valid_split: float=0.2,
              mask_rate: float=0.02,
              seed: int=42) -> Tuple:
  """Load cora dataset.

  Args:
    path: CORA dataset file directory.
    train_split: Train set ratio.
    valid_split: Valid set ratio.
    mask_rate: Ratio of label masked out.
    seed: An integer seed for random state.

  Returns:
    A tuple of features, adjacency matrices, and labels organized in a
    order of (train, valid, test).
  """

  # Load indices, node features, and labels. shape: [N, 1+num_features+1]
  idx_features_labels = np.genfromtxt(os.path.join(path, 'cora.content'),
                                      dtype=str)
  features = csr_matrix(idx_features_labels[:, 1:-1],
                        dtype=np.float32)
  features = row_normalize(features)
  labels = onehot_encode(idx_features_labels[:, -1])

  idx_map = dict(enumerate(idx_features_labels[:, 0]))
  edges_unordered: ndarray = np.genfromtxt(os.path.join(path, 'cora.cities'),
                                           dtype=np.int16)
  edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                   dtype=np.int8)\
            .reshape(edges_unordered.shape)
  adjacency = coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                         shape=(labels.shape[0], labels.shape[0]),
                         dtype=np.float32)
  # Construct symmetric adjacency matrix
  adjacency = adjacency + adjacency.T.multiply(adjacency.T > adjacency)\
                        - adjacency.multiply(adjacency.T > adjacency)

  # Reproducibility
  rs = np.random.RandomState(seed=seed)

  # Generate indices
  indices = np.arange(features.shape[0])
  rs.shuffle(indices)
  train_up = int(indices.shape[0] * train_split)
  valid_up = int(indices.shape[0] * train_split+valid_split)
  train_indices = indices[:train_up]
  valid_indices = indices[train_up:valid_up]
  test_indices = indices[valid_up:]

  # Slicing data
  features_train = features[train_indices]
  features_valid = features[valid_indices]
  features_test = features[test_indices]
  labels_train: ndarray = labels[train_indices]
  labels_valid: ndarray = labels[valid_indices]
  labels_test: ndarray = labels[test_indices]
  labels_relation_train = np.matmul(labels_train, labels_train.T)
  labels_relation_valid = np.matmul(labels_valid, labels_valid.T)
  labels_test = np.argmax(labels_test, axis=-1)

  # Set 0 to 01 for training
  labels_relation_train[np.where(labels_relation_train==0)] = -1
  labels_relation_valid[np.where(labels_relation_valid==0)] = -1

  # Create random mask
  mask_train = rs.choice([1, 0],
                         size=labels_relation_train,
                         p=[mask_rate, 1-mask_rate])
  mask_valid = rs.choice([1, 0],
                         size=labels_relation_valid,
                         p=[mask_rate, 1-mask_rate])

  # Mask out relation labels
  labels_train_masked = mask_train * labels_relation_train
  labels_valid_masked = mask_valid * labels_relation_valid

  # Split matrices for training, validation, and test
  adj_train: coo_matrix = adjacency[train_indices, :][:, train_indices]
  adj_valid: coo_matrix = adjacency[valid_indices, :][:, valid_indices]
  adj_test: coo_matrix = adjacency[test_indices, :][:, test_indices]

  # Tensorize
  features_train: T.Tensor = T.from_numpy(np.array(features_train.to_dense()))
  features_valid: T.Tensor = T.from_numpy(np.array(features_valid.to_dense()))
  features_test: T.Tensor = T.from_numpy(np.array(features_test.to_dense()))
  labels_train_masked = T.from_numpy(labels_train_masked)
  labels_valid_masked =T.from_numpy(labels_valid_masked)
  adj_train = coo2tensor(adj_train)
  adj_valid = coo2tensor(adj_valid)
  adj_test = coo2tensor(adj_test)

  return (features_train, features_valid, features_test,
          adj_train, adj_valid, adj_test,
          labels_train_masked, labels_valid_masked, labels_test)

def coo2tensor(matrix: coo_matrix) -> T.Tensor:
  """Convert a scipy sparse coo_matrix to PyTorch sparse tensor.

  Args:
    matrix: The source SciPy Sparse `coo_matrix`.

  Returns:
    A converted PyTorch Tensor.
  """

  matrix = matrix.tocoo().astype(np.float32)
  indices = T.from_numpy(np.vstack((matrix.row, matrix.col))\
                           .astype(np.int16))
  values = T.from_numpy(matrix.data)
  shape = T.Size(matrix.shape)

  return T.sparse.FloatTensor(indices, values, shape)

def row_normalize(matrix: csr_matrix) -> csr_matrix:
  """Row-normalize scipy sparse matrix.

  Args:
    matrix: SciPy sparse matrix object.

  Returns:
    A row normalized SciPy sparse matrix.
  """

  assert isinstance(matrix, csr_matrix),\
    TypeError(f'Expect a SciPy sparse matrix, but got {type(matrix)}.')

  row_sum = np.array(matrix.sum(1))
  sum_rec = np.power(row_sum, -1).flatten()
  sum_rec[np.isinf(sum_rec)] = 0.  # handle invalid values.
  diag_sum_rec = diags(sum_rec)
  matrix = diag_sum_rec.dot(matrix)

  return matrix

def onehot_encode(labels: ndarray) -> ndarray:
  """Generate one-hot encoding of a label vector.

  Args:
    labels: A label vector of `N` nodes, shape: `[N, 1]`.

  Returns:
    A `ndarray` of shape `[N, num_labels]` with `{0, 1}`.
  """

  classes = set(labels)  # unique labels
  classes_dict = {c:np.eye(len(classes))[i, :]
                  for i, c in enumerate(classes)}
  labels_onehot = np.array(list(map(classes_dict.get, labels)),
                           dtype=np.int8)

  return labels_onehot
