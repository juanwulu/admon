# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fit graph clustering model on data."""

import argparse
import logging
import os
import sys
import time
from collections import OrderedDict
from typing import Dict,  Union

import numpy as np
import torch as T
import torchinfo
from sklearn.metrics import normalized_mutual_info_score as nmi
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from admon.model import Single
from admon.utils import load_npz, load_pkl
from admon.utils import conductance, modularity
from admon.utils import pairwise_precision, pairwise_recall

# Alias
_PathLike = Union[str, 'os.PathLike[str]']
_logger = logging.getLogger(__name__)
logging.basicConfig(format='# %(asctime)s %(name)s %(message)s',
                    level=logging.INFO, stream=sys.stdout,
                    datefmt='%Y%m%d %H:%M:%S %p')

def fit_model(args: argparse.Namespace) -> None:
  """Fit graph clustering model."""

  if args.cuda and T.cuda.is_available():
    device = T.device('cuda:0')
  else:
    device = T.device('cpu')

  # Reproducibility
  np.random.seed(args.seed)
  T.manual_seed(args.seed)
  T.cuda.manual_seed(args.seed)

  # SummaryWriter
  writer = SummaryWriter(log_dir=os.path.join(args.log_dir, 'log/runs',
                                              f'{time.time()}_exp'))

  with logging_redirect_tqdm():
    # Load dataset
    _logger.info('Loading dataset from %s', args.path)
    path: _PathLike = args.path
    file_name = path.split('/')[-1]
    file_type = path.split('.')[-1]

    if os.path.isfile(path) and file_type == 'npz':
      # Single batch sparse inputs stored in `npz` files
      try:
        adj, features, labels, label_indices = load_npz(path)
      except Exception as e:
        raise e
      adj_tensor = T.tensor(adj.todense()).unsqueeze(0)\
                    .float().to(device)
      feature_tensor = T.tensor(features.todense()).unsqueeze(0)\
                        .float().to(device)
    elif os.path.isfile(path) and file_type == 'pkl':
      # Batch inputs stored in pickle files
      try:
        adj, features, labels, label_indices = load_pkl(path)
      except Exception as e:
        raise e
      feature_tensor = T.tensor(features).float().to(device)
      adj_tensor = T.tensor(adj).float().to(device)
      if len(adj_tensor.shape) == 2:
        # Compensate batch dimension
        adj_tensor = adj_tensor.repeat(feature_tensor.size(0), 1, 1)
    else:
      raise RuntimeError('Unsupported dataset')

    # Main procedure
    # TODO (Juanwu): support hierarchical model
    _logger.info('Initializing graph clustering model...')
    betas = tuple(args.betas) if args.betas is not None else None
    model = Single(in_dim=feature_tensor.size(-1),
                   n_clusters=args.n_clusters,
                   hidden=args.hidden, depths=args.depths,
                   dropout=args.dropout, inflation=args.inflation, skip_conn=args.skip_conn,
                   graph_learning=args.graph_learning, n_nodes=feature_tensor.size(-2),
                   alpha=args.alpha, betas=betas,
                   collapse_regularization=args.collapse_regularization,
                   do_unpooling=args.do_unpooling)
    model = model.to(device)
    _logger.info('Single-layer graph model loaded on %s', device)
    _logger.info('\n %s', torchinfo.summary(model))
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer,
                                          step_size=args.lr_decay_steps,
                                          gamma=args.lr_decay_gamma)

    global_train_step: int = 0
    global_test_step: int = 0
    for epoch in tqdm(np.arange(args.epoch),
                      desc='Fit model',
                      position=0, leave=True):
      model.train()
      optimizer.zero_grad()

      if args.graph_gt or not args.graph_learning:
        inputs = (feature_tensor, adj_tensor)
      else:
        inputs = (feature_tensor, None)
      _, _, _, losses = model.forward(inputs)
      loss = T.FloatTensor([0], device=device)
      for loss_val in losses.values():
        if loss_val is not None:
          loss += loss_val
      loss.backward()
      optimizer.step()
      scheduler.step()

      # tracking
      writer.add_scalar('Train/Total loss', loss.item(), global_train_step)
      msg: str = ''
      for name, val in losses.items():
        if val is not None:
          msg += f' {name}: {val.item():.4f}'
          writer.add_scalar(f'Train/{name:s}', val.item(), global_train_step)
      _logger.info('Train [%d/%d]:%s', epoch + 1, args.epoch, msg)
      global_train_step += 1

      # Validation procedure
      model.eval()
      if args.graph_gt or not args.graph_learning:
        pooled_features, preds, adj_pred, _ = model.forward((feature_tensor, adj_tensor))
      else:
        # If graph learning, we use the graph we've learned
        pooled_features, preds, adj_pred, _ = model.forward((feature_tensor, None))
        adj = adj_pred[0].detach().cpu().numpy()

      mod_score, cond_score = 0., 0.
      nmi_score, f1_score = 0., 0.
      for idx, pred_labels in enumerate(preds):
        pred_labels = pred_labels.detach().cpu().numpy().argmax(axis=-1)
        mod_score += modularity(adj, pred_labels)
        cond_score += conductance(adj, pred_labels)

        if labels is not None and label_indices is not None:
          if len(labels.shape) > 1 and len(label_indices.shape) > 1:
            label, label_indice = labels[idx], label_indices[idx]
          else:
            label, label_indice = labels, label_indices
          pred_label = pred_labels[label_indice]
          nmi_score += nmi(label, pred_label, average_method='arithmetic')
          prec = pairwise_precision(label, pred_label)
          recl = pairwise_recall(label, pred_label)
          f1_score += 2 * prec * recl / (prec + recl)

      writer.add_scalar('Valid/Modularity', mod_score / len(preds), global_test_step)
      writer.add_scalar('Valid/Conductance', cond_score / len(preds), global_test_step)
      writer.add_scalar('Valid/NMI', nmi_score / len(preds), global_test_step)
      writer.add_scalar('Valid/F1', f1_score / len(preds), global_test_step)
      global_test_step += 1

      _logger.info('Valid [%d/%d]: Conductance %.4f Modularity %.4f NMI %.4f F1 %.4f',
                    epoch + 1, args.epoch,
                    cond_score / len(preds), mod_score / len(preds),
                    nmi_score / len(preds), f1_score / len(preds))

    # Dump checkpoints
    if not os.path.isdir(os.path.join(args.log_dir, 'checkpoints')):
      os.makedirs(os.path.join(args.log_dir, 'checkpoints'))
    if not os.path.isdir(os.path.join(args.log_dir, 'log')):
      os.makedirs(os.path.join(args.log_dir, 'log'))
    if not os.path.isdir(os.path.join(args.log_dir, 'log/outputs')):
      os.makedirs(os.path.join(args.log_dir, 'log/outputs'))

    # TODO (Juanwu): Load checkpoints
    ckpt_file = f'{time.time()}.single.{feature_tensor.size(-1):d}.pt'
    with open(os.path.join(args.log_dir, 'checkpoints', ckpt_file), 'wb') as f:
      ckpt: Dict = OrderedDict()
      ckpt['state_dict'] = model.state_dict()
      ckpt['optim_dict'] = optimizer.state_dict()

    pred_file = f'{time.time()}.{file_name:s}.{feature_tensor.size(-1)}.npz'
    with open(os.path.join(args.log_dir, 'log/exp', pred_file), 'wb') as f:
      pred_info: Dict = OrderedDict()
      pred_info['pooled_features'] = pooled_features
      pred_info['assignments'] = pred_labels
      pred_info['adjacency'] = adj_pred
      pred_info['modularity'] = mod_score
      pred_info['conductance'] = cond_score
      if labels is not None and label_indices is not None:
        pred_info['nmi_score'] = nmi_score
        pred_info['f1_score'] = f1_score
      np.savez(f, pred_info)



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--activation', type=str, default='selu',
                      help='Name of activation function')
  parser.add_argument('--alpha', type=float, default=10.,
                      help='Weight for ground truth training')
  parser.add_argument('--betas', nargs='+', type=float,
                      default=None, help='Structure adaptation training weights')
  parser.add_argument('--collapse-regularization', type=float,
                      default=1e-3, help='Collapse loss weight')
  parser.add_argument('--cuda', action='store_true',
                      default=False, help='Enable GPU resource')
  parser.add_argument('--depths', type=int, default=1,
                      help='Number of Graph layers')
  parser.add_argument('--do-unpooling', action='store_true',
                      default=False, help='Unpooling cluster features')
  parser.add_argument('--dropout', type=float, default=0.,
                      help='Dropout rate')
  parser.add_argument('--epoch', type=int, default=200,
                      help='Number of epochs to train')
  parser.add_argument('--graph-gt', action='store_true',
                      default=False, help='Use ground truth adjacency matrix')
  parser.add_argument('--graph-learning', action='store_true',
                      default=False, help='Enable structure adaptation')
  parser.add_argument('--hidden', type=int, default=1024,
                      help='Number of neurons in hidden layers')
  parser.add_argument('--inflation', type=int, default=1,
                      help='Inflation factor')
  parser.add_argument('--log-dir', type=str, required=True,
                      help='Logging direction')
  parser.add_argument('--lr', type=float, default=0.001,
                      help='Learning rate')
  parser.add_argument('--lr-decay-gamma', type=float, default=1.,
                      help='Multiplicative factor for lr decay')
  parser.add_argument('--lr-decay-steps', type=int, default=5,
                      help='Steps to call learning rate decay')
  parser.add_argument('--n-clusters', type=int, default=16,
                      help='Number of clusters')
  parser.add_argument('--path', type=str, required=True,
                      help='Data path')
  parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
  parser.add_argument('--skip-conn', action='store_true',
                      default=False, help='Use skip connection')
  parser.add_argument('--weight-decay', type=float, default=0.,
                      help='Weight decay rate')

  arguments = parser.parse_args()
  fit_model(arguments)
