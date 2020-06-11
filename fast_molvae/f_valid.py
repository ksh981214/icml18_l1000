# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.autograd import Variable

import math, random, sys
import numpy as np
import argparse
from collections import deque
import cPickle as pickle

from fast_jtnn import *
from fast_jtnn.jtnn_f import *
from fast_jtnn.datautils import *
import rdkit

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--valid', required=True)
parser.add_argument('--gene', required=True)
parser.add_argument('--vocab', required=True)

parser.add_argument('--trained_model', type=str)

parser.add_argument('--hidden_size', type=int, default=450)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--latent_size', type=int, default=56) #h_T, h_G = 28, 28
parser.add_argument('--depthT', type=int, default=20)
parser.add_argument('--depthG', type=int, default=3)

parser.add_argument('--print_iter', type=int, default=10)
#parser.add_argument('--save_iter', type=int, default=5000)

args = parser.parse_args()
print args

print("#########################################################################")
print("######################## Valid Mode #####################################")
print("#########################################################################")

'''
    model loading
'''
vocab = [x.strip("\r\n ") for x in open(args.vocab)]
vocab = Vocab(vocab)
model = JTNNVAEMLP(vocab, args.hidden_size, args.latent_size, args.depthT, args.depthG).cuda()

model.load_state_dict(torch.load(args.trained_model))
print("load {}".format(args.trained_model))

param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

total_step = 0
meters = np.zeros(7)

loader = MolTreeFolderMLP(args.valid, args.gene, vocab, args.batch_size, num_workers=4)
for it, (batch, gene_batch) in enumerate(loader):
    total_step += 1
    try:
        loss, kl_div, wacc, tacc, sacc, word_loss, topo_loss, assm_loss = model(batch, gene_batch,0)
    except Exception as e:
        print e
        continue

    meters = meters + np.array([kl_div, wacc * 100, tacc * 100, sacc * 100, word_loss, topo_loss, assm_loss])
    '''
        KL_div: prior distribution p(z)와 Q(z|X,Y)와의 KL div
        Word: Label Prediction acc
        Topo: Topological Prediction acc
        Assm: 조립할 때, 정답과 똑같이 했는가? acc
    '''

    if total_step % args.print_iter == 0:
        meters /= args.print_iter

        pnorm= param_norm(model)
        gnorm = grad_norm(model)

        print "[%d] KL: %.2f, Word: %.2f, Topo: %.2f, Assm: %.2f, PNorm: %.2f, GNorm: %.2f" % (it+1, meters[0], meters[1], meters[2], meters[3], pnorm, gnorm)
        print "Wloss: %.2f, Tloss: %.2f, Aloss: %.2f" %(meters[4], meters[5], meters[6])

        sys.stdout.flush()
        meters *= 0
