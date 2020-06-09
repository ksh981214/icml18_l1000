# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import math, random, sys
import numpy as np
import argparse
from collections import deque
import cPickle as pickle

from fast_jtnn import *
import rdkit

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--valid', required=True)
parser.add_argument('--valid_vocab', required=True)

parser.add_argument('--trained_vocab', required=True)
parser.add_argument('--trained_model', required=True, type=str)

parser.add_argument('--hidden_size', type=int, default=450)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--latent_size', type=int, default=56) #h_T, h_G = 28, 28
parser.add_argument('--depthT', type=int, default=20)
parser.add_argument('--depthG', type=int, default=3)

parser.add_argument('--print_iter', type=int, default=10)

## !! Need for GPU
parser.add_argument('--debug', type=int, default=1)
import gc
## !! Need for GPU

args = parser.parse_args()
print args

print("#########################################################################")
print("######################## Valid Mode #####################################")
print("#########################################################################")
'''
    model initializing
'''
trained_vocab = [x.strip("\r\n ") for x in open(args.trained_vocab)]
valid_vocab = [x.strip("\r\n ") for x in open(args.valid_vocab)]
dif_vocab = list(set(valid_vocab) - set(trained_vocab))
vocab = trained_vocab + dif_vocab
vocab = Vocab(vocab)
#model = JTNNMJ(vocab, args.hidden_size, args.latent_size, args.depthT, args.depthG).cuda()

# for param in model.parameters():
#     if param.dim() == 1:
#         nn.init.constant_(param, 0)
#     else:
#         nn.init.xavier_normal_(param)

'''
    trained model loading
'''
#trained_model = JTNNMJ(Vocab(trained_vocab), args.hidden_size, args.latent_size, args.depthT, args.depthG).cuda()
#trained_model.load_state_dict(torch.load(args.trained_model))
model = JTNNMJ(Vocab(trained_vocab), args.hidden_size, args.latent_size, args.depthT, args.depthG).cuda()
for param in model.parameters():
    if param.dim() == 1:
        nn.init.constant_(param, 0)
    else:
        nn.init.xavier_normal_(param)

model.load_state_dict(torch.load(args.trained_model))
print("load {}".format(args.trained_model))

'''
    Modeal Parameter Loading
'''
# trained_model_dict = trained_model.state_dict()
# model_dict = model.state_dict()
# clear_pre_model_dict={}
# for k,v in trained_model_dict.items():
#     if k in model_dict and trained_model_dict[k].size() == model_dict[k].size():
#         clear_pre_model_dict[k]=v
# model_dict.update(clear_pre_model_dict)
# model.load_state_dict(model_dict)

'''
    Embedding Loading
'''
# trained_vocab_dict = {v:k for k,v in enumerate(Vocab(trained_vocab).vocab)}
# vocab_dict = {v:k for k,v in enumerate(vocab.vocab)}
#
# for w in vocab.vocab:
#     if w in trained_vocab_dict:
#         model.state_dict()['decoder.embedding.weight'][vocab_dict[w]] = trained_model.state_dict()['decoder.embedding.weight'][trained_vocab_dict[w]]
#         #print(model.state_dict()['decoder.embedding.weight'][vocab_dict[w]] == trained_model.state_dict()['decoder.embedding.weight'][trained_vocab_dict[w]])
#         model.state_dict()['jtnn.embedding.weight'][vocab_dict[w]] = trained_model.state_dict()['jtnn.embedding.weight'][trained_vocab_dict[w]]
# print("Finish Embedding Loading")

param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

total_step = 0
accs=np.zeros(4)
losses = np.zeros(4)

accs *= 0
losses *= 0

from datetime import datetime

start = datetime.now()
print("TIME: %s " % (str(start)))
loader = MolTreeFolderMJ(args.valid, -1, vocab, args.batch_size, num_workers=4, valid=True)
beta=0
model.eval()
for (batch, g, l) in loader:
    total_step += 1
    try:
        _, kl_div, wacc, tacc, sacc, word_loss, topo_loss, assm_loss, cos_loss = model(batch, g, l, beta)
    except Exception as e:
        print e
        continue

    accs = accs + np.array([kl_div, wacc * 100, tacc * 100, sacc * 100])
    losses = losses + np.array([word_loss, topo_loss, assm_loss, cos_loss])

    if total_step % args.print_iter == 0:
        accs /= args.print_iter
        losses /= args.print_iter

        pnorm = param_norm(model)
        gnorm = grad_norm(model)

        print "[%d] KL: %.2f, Word: %.2f, Topo: %.2f, Assm: %.2f, PNorm: %.2f, GNorm: %.2f" % (total_step, accs[0], accs[1], accs[2], accs[3], pnorm, gnorm)
        print "Wloss: %.2f, Tloss: %.2f, Aloss: %.2f, Closs: %.2f" %(losses[0], losses[1], losses[2], losses[3])

        sys.stdout.flush()

        accs *=0
        losses *=0

    ## !! Need for GPU
    if total_step % args.debug == 0:
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    del obj
            except:
                pass
        torch.cuda.empty_cache()
    ## !! Need for GPU
