# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Graph neural network trainer"""

from typing import Any
import torch as T
import torch.nn as nn
import torch.optim as optim

from modularity_cluster_aviation.model import Single

class DMoNTrainer(object):
  """Graph clustering trainer with modularity loss function.

  Attributes:
  """

  __slots__ = ['in_dim', 'model']
  def __init__(self, in_dim: int, hidden: int, depths: int=1,
               inflation: int=1,) -> None:
    """Initialize graph clustering trainer.

    Args:
      in_dim: An integer input dimension.
      hidden: Number of neurons in hidden layers.
      depths: Number of hidden layers in graph model.

    """

    self.in_dim = in_dim
    self.model = Single(in_dim, hidden=hidden, depths=depths,
                        inflation=inflation)

  def train(self) -> None:
    self.train_one_epoch()

  def eval(self, epoch:int=-1) -> None:
    self.eval_one_epoch(epoch=epoch)

  def train_one_epoch(self) -> Any:
    return None

  def eval_one_epoch(self, epoch: int) -> Any:
    return None
