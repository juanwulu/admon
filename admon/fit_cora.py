# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""Main training function for DMoN."""

import argparse
import logging
import sys

import numpy as np
import torch as T
import torch.optim as optim
import torchinfo
from sklearn.metrics import normalized_mutual_info_score as nmi
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from admon.model import Single
from admon.utils import load_npz
from admon.utils import conductance, modularity
from admon.utils import pairwise_precision, pairwise_recall

# Alias
_logger = logging.getLogger(__name__)
logging.basicConfig(format='# %(asctime)s %(name)s %(message)s',
    level=logging.INFO, stream=sys.stdout, datefmt='%Y%m%d %H:%M:%S %p')

def fit_cora(args: argparse.Namespace) -> None:
  """Main loop function."""

  device = T.device('cuda:0') if T.cuda.is_available() and args.cuda\
                              else T.device('cpu')
  # Reproducibility
  np.random.seed(args.seed)
  T.manual_seed(args.seed)
  T.cuda.manual_seed(args.seed)

  with logging_redirect_tqdm():
    # Load cora dataset
    _logger.info('Loading data...')
    try:
      adj, features, labels, label_indices = load_npz(args.path)
    except Exception as e:
      raise RuntimeError from e
    adj_tensor = T.tensor(adj.todense()).unsqueeze(0).float()
    features_tensor = T.tensor(features.todense())\
                      .unsqueeze(0)\
                      .float()

    # Fit the model
    _logger.info('Initializing graph clustering model...')
    model = Single(features_tensor.size(-1), args.n_clusters,
                    hidden=args.hidden, depths=args.depths,
                    dropout=args.dropout, inflation=args.inflation,
                    skip_conn=args.skip_conn,
                    collapse_regularization=args.collapse_regularization,
                    do_unpooling=args.do_unpooling)
    model = model.to(device)
    _logger.info('Single-layer graph model loaded on %s', device)
    _logger.info('\n %s', torchinfo.summary(model))
    optimizer = optim.Adam(list(model.parameters()), lr=args.lr,
                            weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=args.lr_decay_steps,
                                          gamma=args.lr_decay_gamma)
    for epoch in tqdm(np.arange(args.epoch),
                      desc='Fitting model',
                    position=0, leave=True):
      model.train()
      optimizer.zero_grad()

      _, _, m_loss, c_loss = model.forward((features_tensor, adj_tensor))
      loss: T.Tensor = m_loss + c_loss
      loss.backward()
      optimizer.step()
      scheduler.step()

      _logger.info('Train [%d/%d]: Modularity loss %.4f Collapse loss %.4f',
                    epoch+1, args.epoch, m_loss, c_loss)

    # Validation
    model.eval()
    _, preds, _, _ = model.forward((features_tensor, adj_tensor))
    pred_labels = preds[0].detach().cpu().numpy().argmax(axis=-1)
    _logger.info('Validate: Conductance %.4f, Modularity %.4f NMI %.4f F1 %.4f',
                 conductance(adj, pred_labels),
                 modularity(adj, pred_labels),
                 nmi(labels,pred_labels[label_indices]),
                 2*pairwise_precision(labels, pred_labels[label_indices])\
                  *pairwise_recall(labels, pred_labels[label_indices])\
                  /(pairwise_precision(labels, pred_labels[label_indices])\
                    +pairwise_recall(labels, pred_labels[label_indices])))

if __name__ == '__main__':
  # Arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--activation', type=str, default='selu',
                      help='Name of activation function')
  parser.add_argument('--collapse-regularization', type=float,
                      default=1e-3, help='Collapse loss weight')
  parser.add_argument('--cuda', action='store_true',
                      default=False, help='Enable GPU resource')
  parser.add_argument('--depths', type=int, default=1,
                      help='Number of Graph layers')
  parser.add_argument('--do-unpooling', action='store_true',
                      default=False, help='Unpooling cluster features')
  parser.add_argument('--dropout', type=float, default=0., help='Dropout rate')
  parser.add_argument('--epoch', type=int,
                      default=200, help='Number of epochs to train')
  parser.add_argument('--hidden', type=int, default=512,
                      help='Number of neurons in hidden layers')
  parser.add_argument('--inflation', type=int, default=1,
                      help='Inflation factor')
  parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
  parser.add_argument('--lr-decay-gamma', type=float,
                      default=1., help='Multiplicative factor for lr decay')
  parser.add_argument('--lr-decay-steps', type=int,
                      default=5, help='Steps to call learning rate decay')
  parser.add_argument('--n-clusters', type=int, default=16,
                      help='Number of clusters')
  parser.add_argument('--path', type=str, required=True, help='Data path')
  parser.add_argument('--seed', type=int, default=42, help='Random seed')
  parser.add_argument('--skip-conn', action='store_true',
                      default=False, help='Use skip connection')
  parser.add_argument('--weight-decay', type=float, default=0.,
                      help='Weight decay rate')

  arguments = parser.parse_args()
  fit_cora(arguments)
