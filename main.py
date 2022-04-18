# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""Main training function for DMoN."""

import argparse
import os

import numpy as np
import torch as T
import torch.nn.functional as F
import torch.optim as optim

from .model import DMoN

def train(epoch: int) -> None:
  """Train function for one epoch."""
  pass

def test(epoch: int=-1) -> None:
  """Test function for one epoch."""

def main(args: argparse.Namespace) -> None:
  """Main loop function."""

  device = T.device('cuda:0') if T.cuda.is_available() and args.cuda\
                              else T.cpu()
  # Reproducibility
  rs = np.random.RandomState(seed=args.seed)
  T.manual_seed(args.seed)
  T.cuda.manual_seed(args.seed)

  model = DMoN()
  optimizer = optim.Adam(list(model.parameters()), lr=args.lr,
                          weight_decay=args.weight_decay)
  scheduler = optim.lr_scheduler.StepLR(optimizer,
                                        step_size=args.lr_decay_steps,
                                        gamma=args.lr_decay_gamma)
  pass

if __name__ == '__main__':
  # Arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--cuda', action='store_true',
                      default=False, help='Enable GPU resource.')
  parser.add_argument('--epochs', type=int,
                      default=100, help='Number of epochs to train.')
  parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
  parser.add_argument('--seed', type=int, default=42, help='Random seed.')
  args = parser.parse_args()

  main(args)
