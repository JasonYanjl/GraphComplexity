from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from pygcn.utils import load_data, accuracy
from pygcn.models import GCN, MLP

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=400,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--add_edge_mutual', type=float, default=1)
parser.add_argument('--del_edge_mutual', type=float, default=1)
parser.add_argument('--output_path', type=str, default="")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data(rate=1)

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
    return acc_val.item()


def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    return acc_test.item()


tot_acc = 0
tot_val_acc = 0
TOT_ROUND = 10
res_list = []
for i in range(TOT_ROUND):
    # idx_all = np.arange(features.shape[0])
    # np.random.shuffle(idx_all)
    # idx_train = idx_all[np.arange(200)]
    # idx_val = idx_all[np.arange(200, 300)]
    # idx_test = idx_all[np.arange(300, 505)]
    # idx_train = idx_all[np.arange(300)]
    # idx_val = idx_all[np.arange(300, 350)]
    # idx_test = idx_all[np.arange(200, 300)]
    # idx_test = idx_val
    # idx_train = idx_all[np.arange(500)]
    # idx_val = idx_all[np.arange(500, 1500)]
    # idx_test = idx_all[np.arange(1500, 2500)]
    idx_train = torch.LongTensor(idx_train.cpu())
    idx_val = torch.LongTensor(idx_val.cpu())
    idx_test = torch.LongTensor(idx_test.cpu())
    if args.cuda:
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    model = GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
                dropout=args.dropout)
    if args.cuda:
        model.cuda()
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)
    # Train model
    t_total = time.time()
    best_val = 0
    for epoch in range(args.epochs):
        best_val = max(best_val, train(epoch))
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Testing
    test_res = test()
    tot_acc += test_res
    tot_val_acc += best_val

    # res_list.append(test_res)
    res_list.append(best_val)
print('------------------------------')
print(f'avg val acc: {tot_val_acc / TOT_ROUND}')
print(f'avg test acc: {tot_acc / TOT_ROUND}')

print(f'std: {np.std(np.array(res_list))}')
# print(f'res: {(np.array(res_list) - np.mean(np.array(res_list))).tolist()}')
