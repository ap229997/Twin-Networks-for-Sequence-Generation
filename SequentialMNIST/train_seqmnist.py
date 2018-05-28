import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.datasets as dsets
from layer_pytorch import *
import time
import click
import numpy
import numpy as np
import os
import random
from itertools import chain
import load
import torch.nn.functional as F
import time

# set a bunch of seeds
seed = 1234
rng = np.random.RandomState(seed)


def get_epoch_iterator(nbatch, X, Y=None):
    ndata = X.shape[0]
    samples = rng.permutation(np.arange(ndata))
    for b in range(0, ndata, nbatch):
        idx = samples[b:b + nbatch]
        assert len(idx) == nbatch
        x = X[idx]
        if Y is not None:
            y = Y[idx]
        else:
            y = None
        x = x.reshape((-1, 784)).transpose(1, 0)
        yield (x, y)


def binary_crossentropy(x, p):
    return -torch.sum((torch.log(p + 1e-6) * x +
                       torch.log(1 - p + 1e-6) * (1. - x)), 0)


class Model(nn.Module):
    def __init__(self, rnn_dim, nlayers):
        super(Model, self).__init__()
        self.rnn_dim = rnn_dim
        self.nlayers = nlayers
        self.embed = nn.Embedding(2, 200)
        self.fwd_rnn = nn.LSTM(200, rnn_dim, nlayers, batch_first=False, dropout=0)
        self.bwd_rnn = nn.LSTM(200, rnn_dim, nlayers, batch_first=False, dropout=0)
        self.fwd_prj_prev = nn.Linear(200, 512)
        self.bwd_prj_prev = nn.Linear(200, 512)
        self.fwd_prj_out = nn.Linear(rnn_dim, 512)
        self.bwd_prj_out = nn.Linear(rnn_dim, 512)
        self.fwd_out = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())
        self.bwd_out = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())
        self.fwd_aff = nn.Linear(rnn_dim, rnn_dim)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.nlayers, bsz, self.rnn_dim).zero_()),
                Variable(weight.new(self.nlayers, bsz, self.rnn_dim).zero_()))

    def rnn(self, x, hidden, forward=True):
        rnn_mod = self.fwd_rnn if forward else self.bwd_rnn
        out_mod = self.fwd_out if forward else self.bwd_out
        prv_mod = self.fwd_prj_prev if forward else self.bwd_prj_prev
        prj_mod = self.fwd_prj_out if forward else self.bwd_prj_out
        bsize = x.size(1)
        x = self.embed(x)
        vis, states = rnn_mod(x, hidden)
        vis_ = vis.view(vis.size(0) * bsize, self.rnn_dim)
        x_ = x.view(vis.size(0) * bsize, x.size(2))
        out_ = F.leaky_relu(prv_mod(x_) + prj_mod(vis_), 0.3, False)
        out = out_mod(out_)
        out = out.view(vis.size(0), bsize)
        # transform forward with affine
        if forward:
            vis_ = self.fwd_aff(vis_)
            vis = vis_.view(vis.size(0), bsize, self.rnn_dim)
        return out, vis, states

    def forward(self, fwd_x, hidden):
        fwd_out, fwd_vis, _ = self.rnn(fwd_x, hidden)
        bwd_out, bwd_vis, _ = self.rnn(bwd_x, hidden, forward=False)
        return fwd_out, fwd_vis


def evaluate(model, bsz, data_x, data_y):
    model.eval()
    hidden = model.init_hidden(bsz)
    valid_loss = []
    for x, _ in get_epoch_iterator(bsz, data_x, data_y):
        x = np.concatenate([np.zeros((1, bsz)).astype('int32'), x], axis=0)
        x = torch.from_numpy(x)
        inp = Variable(x[:-1], volatile=True).long().cuda()
        trg = Variable(x[1:], volatile=True).float().cuda()
        out, vis, _ = model.rnn(inp, hidden)
        loss = binary_crossentropy(trg, out).mean()
        valid_loss.append(loss.data[0])
    return np.asarray(valid_loss).mean()


@click.command()
@click.option('--nlayers', default=1)
@click.option('--num_epochs', default=10)
@click.option('--rnn_dim', default=1024)
@click.option('--bsz', default=50)
@click.option('--lr', default=0.001)
@click.option('--twin', default=0.) # used to specify whether to use the twin network or not and what weightage to give it

def train(nlayers, num_epochs, rnn_dim, bsz, lr, twin):
    # use hugo's binarized MNIST
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    log_interval = 100
    folder_id = 'seqmnist_twin_logs'
    model_id = 'seqmnist_twin1{}'.format(twin)
    log_file_name = os.path.join(folder_id, model_id + '.txt')
    if not os.path.exists(log_file_name):
        os.makedirs(log_file_name)

    model_file_name = os.path.join(folder_id, model_id + '.pt')
    if not os.path.exists(model_file_name):
        os.makedirs(model_file_name)

    log_file = open(log_file_name, 'w')

    # Hugo's version, for compatibility with SOTA.
    train_x, valid_x, test_x = \
        load.load_binarized_mnist('./mnist/data')
    train_y = None
    valid_y = None
    test_y = None

    model = Model(rnn_dim, nlayers)
    model.cuda()
    hidden = model.init_hidden(bsz)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    nbatches = train_x.shape[0] // bsz
    t = time.time()
    for epoch in range(num_epochs):
        step = 0
        old_valid_loss = np.inf
        b_fwd_loss, b_bwd_loss, b_twin_loss, b_all_loss = 0., 0., 0., 0.
        model.train()
        print('Epoch {}: ({})'.format(epoch, model_id.upper()))
        for x, _ in get_epoch_iterator(bsz, train_x, train_y):
            opt.zero_grad()
            # x = (0, x1, x2, x3, x4)
            # fwd_inp = (0, x1, x2, x3)
            # fwd_trg = (x1, x2, x3, x4)
            x_ = np.concatenate([np.zeros((1, bsz)).astype('int32'), x], axis=0)
            fwd_x = torch.from_numpy(x_)
            fwd_inp = Variable(fwd_x[:-1]).long().cuda()
            fwd_trg = Variable(fwd_x[1:]).float().cuda()
            
            # reverse the contents
            # bwd_x = (0, x4, x3, x2, x1)
            # bwd_inp = (0, x4, x3, x2)
            # bwd_trg = (x4, x3, x2, x1)
            bwd_x = numpy.flip(x, 0).copy()
            x_ = np.concatenate([np.zeros((1, bsz)).astype('int32'), bwd_x], axis=0)
            bwd_x = torch.from_numpy(x_)
            bwd_inp = Variable(bwd_x[:-1]).long().cuda()
            bwd_trg = Variable(bwd_x[1:]).float().cuda()

            # compute all the states for forward and backward
            fwd_out, fwd_vis = model(fwd_inp, hidden)
            assert fwd_out.size(0) == 784
            fwd_loss = binary_crossentropy(fwd_trg, fwd_out).mean()
            bwd_loss = binary_crossentropy(bwd_trg, bwd_out).mean()
            bwd_loss = bwd_loss * (twin > 0.)

            # reversing backstates
            fwd_vis = (out_x1, out_x2, out_x3, out_x4)
            bwd_vis_inv = (out_x1, out_x2, out_x3, out_x4)
            # therefore match: fwd_vis and bwd_vis_inv
            idx = np.arange(bwd_vis.size(0))[::-1].tolist()
            idx = torch.LongTensor(idx)
            idx = Variable(idx).cuda()
            bwd_vis_inv = bwd_vis.index_select(0, idx)
            bwd_vis_inv = bwd_vis_inv.detach()

            twin_loss = ((fwd_vis - bwd_vis_inv) ** 2).mean()
            twin_loss = twin_loss * twin
            all_loss = fwd_loss + bwd_loss + twin_loss
            
            all_loss.backward()

            torch.nn.utils.clip_grad_norm(model.parameters(), 100.)
            opt.step()

            b_fwd_loss += fwd_loss.data[0]
            b_bwd_loss += bwd_loss.data[0]
            b_twin_loss += twin_loss.data[0]
            b_all_loss += all_loss.data[0]

            if (step + 1) % log_interval == 0:
                s = time.time()
                log_line = 'Epoch [%d/%d], Step [%d/%d], loss: %f, %.2fit/s' % (
                    epoch, num_epochs, step + 1, nbatches,
                    b_fwd_loss / log_interval,
                    log_interval / (s - t))
                b_all_loss = 0.
                b_fwd_loss = 0.
                b_bwd_loss = 0.
                b_twin_loss = 0.
                t = time.time()
                print(log_line)
                print(time.time(),time.clock())


                log_file.write(log_line + '\n')

            step += 1

        # evaluate per epoch
        print('--- Epoch finished ----')
        val_loss = evaluate(model, bsz, valid_x, valid_y)
        log_line = 'valid -- nll: %f' % (val_loss)
        print(log_line)
        log_file.write(log_line + '\n')
        test_loss = evaluate(model, bsz, test_x, test_y)
        log_line = 'test -- nll: %f' % (test_loss)
        print(log_line)
        
        
        log_file.write(log_line + '\n')

        if old_valid_loss > val_loss:
            old_valid_loss = val_loss
            torch.save(model.state_dict(), model_file_name)
        else:
            for param_group in opt.param_groups:
                lr = param_group['lr']
                if lr > 0.0002:
                    lr *= 0.5
                param_group['lr'] = lr


if __name__ == '__main__':
    print(time.time(),time.clock())
    train()
