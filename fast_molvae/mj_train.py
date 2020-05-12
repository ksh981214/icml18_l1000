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
import rdkit


lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--train', required=True)
### NEW
parser.add_argument('--num_neg_folder', type=int, required=True)
### NEW
parser.add_argument('--vocab', required=True)
parser.add_argument('--save_dir', required=True)
#### NEW
parser.add_argument('--pre_vocab_dir', required=True)
parser.add_argument('--pre_model_dir', required=True)
#### NEW
parser.add_argument('--load_epoch', type=int, default=0)

parser.add_argument('--hidden_size', type=int, default=450)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--latent_size', type=int, default=56) #h_T, h_G = 28, 28
parser.add_argument('--depthT', type=int, default=20)
parser.add_argument('--depthG', type=int, default=3)

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--clip_norm', type=float, default=50.0)
parser.add_argument('--beta', type=float, default=0.0)
parser.add_argument('--step_beta', type=float, default=0.002)
parser.add_argument('--max_beta', type=float, default=1.0)
parser.add_argument('--warmup', type=int, default=40000)

parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--anneal_rate', type=float, default=0.9)
parser.add_argument('--anneal_iter', type=int, default=40000)
parser.add_argument('--kl_anneal_iter', type=int, default=2000)
parser.add_argument('--print_iter', type=int, default=50)
parser.add_argument('--save_iter', type=int, default=5000)

## !! Need for GPU
parser.add_argument('--debug', type=int, default=1)
import gc
## !! Need for GPU

###############################################
args = parser.parse_args()
print args

'''
    model initializing
'''
vocab = [x.strip("\r\n ") for x in open(args.vocab)]
vocab = Vocab(vocab)
model = JTNNMJ(vocab, args.hidden_size, args.latent_size, args.depthT, args.depthG).cuda()

for param in model.parameters():
    if param.dim() == 1:
        nn.init.constant_(param, 0)
    else:
        nn.init.xavier_normal_(param)

if args.load_epoch > 0:
    model.load_state_dict(torch.load(args.save_dir + "/model.iter-" + str(args.load_epoch)))
    print("load model.iter-{}".format(str(args.load_epoch)))

else:
    '''
        pre_model loading
    '''
    pre_vocab = [x.strip("\r\n ") for x in open(args.pre_vocab_dir)]
    pre_vocab = Vocab(pre_vocab)

    pre_model = JTNNVAE(pre_vocab, args.hidden_size, args.latent_size, args.depthT, args.depthG)

    pre_model.load_state_dict(torch.load(args.pre_model_dir))
    print("{} loading finish".format(args.pre_model_dir))

    '''
        Modeal Parameter Loading
    '''
    pre_model_dict = pre_model.state_dict()
    model_dict = model.state_dict()
    clear_pre_model_dict={}

    for k,v in pre_model_dict.items():
        if k in model_dict and pre_model_dict[k].size() == model_dict[k].size():
            clear_pre_model_dict[k]=v

    model_dict.update(clear_pre_model_dict)
    model.load_state_dict(model_dict)
    '''
        the size of decoder.W_o.weight is different!
        the size of decoder.W_o.bias is different!
        the size of jtnn.embedding.weight is different!
        the size of decoder.embedding.weight is different!
    '''

    '''
        Embedding Loading
    '''
    pre_vocab_dict = {v:k for k,v in enumerate(pre_vocab.vocab)}
    vocab_dict = {v:k for k,v in enumerate(vocab.vocab)}
    for w in vocab.vocab:
        if w in pre_vocab_dict:
            model.state_dict()['decoder.embedding.weight'][vocab_dict[w]] = pre_model.state_dict()['decoder.embedding.weight'][pre_vocab_dict[w]]
            model.state_dict()['jtnn.embedding.weight'][vocab_dict[w]] = pre_model.state_dict()['jtnn.embedding.weight'][pre_vocab_dict[w]]
    print("Finish Embedding Loading")

    del(pre_vocab, pre_model, pre_model_dict, clear_pre_model_dict, pre_vocab_dict)
print "Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,)

###############################################

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)

param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

total_step = 0
beta = args.beta
meters = np.zeros(8)

from datetime import datetime

for epoch in xrange(args.epoch):
    meters *= 0

    start = datetime.now()
    print("EPOCH: %d | TIME: %s " % (epoch+1, str(start)))
    loader = MolTreeFolderMJ(args.train, args.num_neg_folder, vocab, args.batch_size, num_workers=4)

    for it, (batch, gene_batch, label_batch) in enumerate(loader):
        total_step += 1
        try:
            model.zero_grad()
            loss, kl_div, wacc, tacc, sacc, word_loss, topo_loss, assm_loss, cos_loss = model(batch, gene_batch, label_batch, beta)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()
        except Exception as e:
            print e
            continue

        meters = meters + np.array([kl_div, wacc * 100, tacc * 100, sacc * 100, word_loss, topo_loss, assm_loss, cos_loss])

        if total_step % args.print_iter == 0:
            meters /= args.print_iter

            pnorm= param_norm(model)
            gnorm = grad_norm(model)

            print "[%d][%d] Beta: %.6f, KL: %.2f, Word: %.2f, Topo: %.2f, Assm: %.2f, PNorm: %.2f, GNorm: %.2f" % (epoch, it+1, beta, meters[0], meters[1], meters[2], meters[3], pnorm, gnorm)
            print "Wloss: %.2f, Tloss: %.2f, Aloss: %.2f, Closs: %.2f" %(meters[4], meters[5], meters[6], meters[7])

            sys.stdout.flush()
            meters *= 0

        if total_step % args.save_iter == 0:
            torch.save(model.state_dict(), args.save_dir + "/model.iter-" + str(total_step))

        if total_step % args.anneal_iter == 0:
            scheduler.step()
            print "learning rate: %.6f" % scheduler.get_lr()[0]

        if total_step % args.kl_anneal_iter == 0 and total_step >= args.warmup:
            beta = min(args.max_beta, beta + args.step_beta)

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


    if args.load_epoch != 0:
        torch.save(model.state_dict(), args.save_dir + "/model.epoch-" + str(epoch+args.load_epoch))
    else:
        torch.save(model.state_dict(), args.save_dir + "/model.epoch-" + str(epoch))

    #Plot per 1 epoch
    print "Cosume Time per Epoch %s" % (str(datetime.now()-start))

    #Delete Loader
    referrers = gc.get_referrers(loader)
    for referrer in referrers:
        if type(referrer) == dict:
            for key, value in referrer.items():
                if value is loader:
                    referrer[key] = None
    del loader
