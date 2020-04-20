#!/usr/bin/env python3
import copy
import numpy as np
import random
import argparse
import statistics
import torch

from trainer import Trainer
from gnn import GNN
from wgnn import WGNN
import loader

from orion.client import report_results



def main(opt):
    device = torch.device('cuda' if opt['cuda'] == True and torch.cuda.is_available() else 'cpu')

    #--------------------------------------------------
    # Load data.
    #--------------------------------------------------
    net_file = opt['dataset'] + '/net.txt'
    label_file = opt['dataset'] + '/label.txt'
    feature_file = opt['dataset'] + '/feature.txt'
    train_file = opt['dataset'] + '/train.txt'
    dev_file = opt['dataset'] + '/dev.txt'
    test_file = opt['dataset'] + '/test.txt'

    vocab_node = loader.Vocab(net_file, [0, 1])
    vocab_label = loader.Vocab(label_file, [1])
    vocab_feature = loader.Vocab(feature_file, [1])

    opt['num_node'] = len(vocab_node)
    opt['num_feature'] = len(vocab_feature)
    opt['num_class'] = len(vocab_label)

    graph = loader.Graph(file_name=net_file, entity=[vocab_node, 0, 1])
    label = loader.EntityLabel(file_name=label_file, entity=[vocab_node, 0], label=[vocab_label, 1])
    feature = loader.EntityFeature(file_name=feature_file, entity=[vocab_node, 0], feature=[vocab_feature, 1])
    d = graph.to_symmetric(opt['self_link_weight'])
    feature.to_one_hot(binary=True)
    adj = graph.get_sparse_adjacency(opt['cuda'])
    deg = torch.zeros(adj.shape[0])
    for k,v  in d.items():
        deg[k] = v

    with open(train_file, 'r') as fi:
        idx_train = [vocab_node.stoi[line.strip()] for line in fi]
    with open(dev_file, 'r') as fi:
        idx_dev = [vocab_node.stoi[line.strip()] for line in fi]
    with open(test_file, 'r') as fi:
        idx_test = [vocab_node.stoi[line.strip()] for line in fi]

    inputs = torch.Tensor(feature.one_hot)
    target = torch.LongTensor(label.itol)
    idx_train = torch.LongTensor(idx_train)
    idx_dev = torch.LongTensor(idx_dev)
    idx_test = torch.LongTensor(idx_test)

    if opt['cuda']:
        inputs = inputs.cuda()
        target = target.cuda()
        idx_train = idx_train.cuda()
        idx_dev = idx_dev.cuda()
        idx_test = idx_test.cuda()

    #--------------------------------------------------
    # Build model.
    #--------------------------------------------------
    if opt['weight']:
        gnn = WGNN(opt, adj, deg, opt['time'])
    else:
        gnn = GNN(opt, adj, deg, opt['time'])
    trainer = Trainer(opt, gnn)
    print(gnn)
    print(opt)

    #--------------------------------------------------
    # Train model.
    #--------------------------------------------------
    def train(epochs):
        best = 0.0
        results = []
        prev_dev_acc = 0
        cnt = 0
        lr = opt['lr']
        for epoch in range(0, epochs):
            # -----------------------
            # Train Model
            # -----------------------
            if opt['weight']:
                loss = trainer.updatew(inputs, target, idx_train)
            else:
                loss = trainer.update(inputs, target, idx_train)
            # -----------------------
            # Evaluate Model
            # -----------------------
            _, preds, accuracy_dev = trainer.evaluate(inputs, target, idx_dev)
            # -----------------------
            # Test Model
            # -----------------------
            _, preds, accuracy_test = trainer.evaluate(inputs, target, idx_test)
            print(
                'Epoch: {} | Loss: {:.3f} | Dev acc: {:.3f} | Test acc: {:.3f} | Forward: {} {:.3f} | Backward: {} {:.3f}'.format(
                    epoch,
                    loss,
                    accuracy_dev,
                    accuracy_test,
                    trainer.fm.get_value(),
                    trainer.fm.get_average(),
                    trainer.bm.get_value(),
                    trainer.bm.get_average()))
            results += [(accuracy_dev, accuracy_test)]
            if accuracy_dev >= best:
                best = accuracy_dev
                state = dict([('model', copy.deepcopy(trainer.model.state_dict())),
                              ('optim', copy.deepcopy(trainer.optimizer.state_dict()))])
        trainer.model.load_state_dict(state['model'])
        trainer.optimizer.load_state_dict(state['optim'])
        return results

    results = train(opt['epoch'])


    def get_accuracy(results):
        best_dev, acc_test = 0.0, 0.0
        for d, t in results:
            if d > best_dev:
                best_dev, acc_test = d, t
        return acc_test, best_dev

    acc_test = get_accuracy(results)

    print('{:.3f}'.format(acc_test[0]*100))

    return acc_test

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='data')
    parser.add_argument('--save', type=str, default='/')
    parser.add_argument('--hidden_dim', type=int, default=16, help='Hidden dimension.')
    parser.add_argument('--input_dropout', type=float, default=0.5, help='Input dropout rate.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate.')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--decay', type=float, default=5e-4, help='Weight decay for optimization')
    parser.add_argument('--self_link_weight', type=float, default=1.0, help='Weight of self-links.')
    parser.add_argument('--epoch', type=int, default=10, help='Number of training epochs per iteration.')
    parser.add_argument('--iter', type=int, default=10, help='Number of training iterations.')
    parser.add_argument('--use_gold', type=int, default=1,
                        help='Whether using the ground-truth label of labeled objects, 1 for using, 0 for not using.')
    parser.add_argument('--tau', type=float, default=1.0, help='Annealing temperature in sampling.')
    parser.add_argument('--alpha', type=float, default=1.0, help='Factor in front matrix A.')
    parser.add_argument('--draw', type=str, default='max',
                        help='Method for drawing object labels, max for max-pooling, smp for sampling.')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--time', type=float, default=1.0, help='End time of ODE function.')
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
    parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
    parser.add_argument('--analysis', type=bool, default=False, help='Enables the production of various analysis '
                                                                     'plots.')
    parser.add_argument('--weight', type=bool, default=False, help='Set to true to use CGNN with weight')

    args = parser.parse_args()

    if args.cpu:
        args.cuda = False
    elif args.cuda:
        args.cuda = True

    opt = vars(args)

    main(opt)


