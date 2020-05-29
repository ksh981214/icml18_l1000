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
import pdb

from fast_jtnn import *
import rdkit

from datetime import datetime
from plot import save_KL_plt, save_Acc_plt, save_Norm_plt, save_Loss_plt, save_Beta_plt

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--train', required=True)
parser.add_argument('--num_neg_folder', type=int, required=True) ##
parser.add_argument('--vocab', required=True)
parser.add_argument('--save_dir', required=True)
####
parser.add_argument('--pre_vocab_dir', required=True)
parser.add_argument('--pre_model_dir', required=True)
####
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
## For plot
parser.add_argument('--plot', type=int, default=0)
## For plot

#For freeze
parser.add_argument('--freeze', type=int, default=0)
#

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
        Embedding Loading
    '''
    #print(len(list(set(vocab.vocab) & set(pre_vocab.vocab)))) #270
    pre_vocab_dict = {v:k for k,v in enumerate(pre_vocab.vocab)}
    vocab_dict = {v:k for k,v in enumerate(vocab.vocab)}
    for w in vocab.vocab:
        if w in pre_vocab_dict:
            model.state_dict()['decoder.embedding.weight'][vocab_dict[w]] = pre_model.state_dict()['decoder.embedding.weight'][pre_vocab_dict[w]]
            model.state_dict()['jtnn.embedding.weight'][vocab_dict[w]] = pre_model.state_dict()['jtnn.embedding.weight'][pre_vocab_dict[w]]
    print("Finish Embedding Loading")

    if args.freeze == 1:
        model_lst, common_lst = [],[]
        for k,v in model_dict.items():
            model_lst.append(k)
        for k,v in clear_pre_model_dict.items():
            common_lst.append(k)

        no_freeze_lst = list(set(model_lst) - set(common_lst))

        for f in common_lst:
            #print(f)
            model.state_dict()[f].requires_grad = False

        #print()

        for f in no_freeze_lst:
            #print(f)
            model.state_dict()[f].requires_grad = True

    del pre_vocab, pre_model, pre_model_dict, clear_pre_model_dict, pre_vocab_dict
print "Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,)

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)

param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

total_step = 0
beta = args.beta

accs=np.zeros(4)
losses = np.zeros(4)

if args.plot:
    import os # for save plot
    x_plot,kl_plot,word_plot, topo_plot, assm_plot, pnorm_plot, gnorm_plot, beta_plot, wloss_plot, tloss_plot, aloss_plot, closs_plot =[],[],[],[],[],[],[],[],[],[],[],[]
    d=datetime.now()
    now = str(d.year)+'_'+str(d.month)+'_'+str(d.day)+'_'+str(d.hour)+'_'+str(d.minute)
    if args.load_epoch != 0:
        folder_name = "f_" + "h" + str(args.hidden_size) + '_' + "bs" + str(args.batch_size) + '_' + now + "_from_" +str(args.load_epoch)
    else:
        folder_name = "f_" + "h" + str(args.hidden_size) + '_' + "bs" + str(args.batch_size) + '_' + now

    os.makedirs('./plot/'+folder_name+'/KL')        #KL
    os.makedirs('./plot/'+folder_name+'/Acc')       #Word, Topo, Assm
    os.makedirs('./plot/'+folder_name+'/Norm')      #PNorm, GNorm
    os.makedirs('./plot/'+folder_name+'/Loss')      #Word, Topo, Assm LOSS
    os.makedirs('./plot/'+folder_name+'/Beta')      #Word, Topo, Assm LOSS
    print("...Finish Making Plot Folder...")

for epoch in xrange(args.epoch):
    accs *= 0
    losses *= 0

    start = datetime.now()
    print("EPOCH: %d | TIME: %s " % (epoch+1, str(start)))
    loader = MolTreeFolderMJ(args.train, args.num_neg_folder, vocab, args.batch_size, num_workers=4)

    for (batch, g, l) in loader:
        total_step += 1
        try:
            model.zero_grad()
            loss, kl_div, wacc, tacc, sacc, word_loss, topo_loss, assm_loss, cos_loss = model(batch, g, l, beta)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()
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

            print "[%d][%d] Beta: %.6f, KL: %.2f, Word: %.2f, Topo: %.2f, Assm: %.2f, PNorm: %.2f, GNorm: %.2f" % (epoch, total_step, beta, accs[0], accs[1], accs[2], accs[3], pnorm, gnorm)
            print "Wloss: %.2f, Tloss: %.2f, Aloss: %.2f, Closs: %.2f" %(losses[0], losses[1], losses[2], losses[3])

            if args.plot:
                x_plot.append(total_step)
                kl_plot.append(accs[0])
                word_plot.append(accs[1])
                topo_plot.append(accs[2])
                assm_plot.append(accs[3])
                pnorm_plot.append(pnorm)
                gnorm_plot.append(gnorm)
                beta_plot.append(beta)
                wloss_plot.append(losses[0])
                tloss_plot.append(losses[1])
                aloss_plot.append(losses[2])
                closs_plot.append(losses[3])
            sys.stdout.flush()

            accs *=0
            losses *=0

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

    if args.plot:
        save_KL_plt(folder_name, epoch, x_plot, kl_plot)
        save_Acc_plt(folder_name, epoch, x_plot, word_plot, topo_plot, assm_plot)
        save_Norm_plt(folder_name, epoch, x_plot, pnorm_plot, gnorm_plot)
        save_Loss_plt(folder_name, epoch, x_plot, wloss_plot, tloss_plot, aloss_plot)
        save_Beta_plt(folder_name, epoch, x_plot, beta_plot)
        del x_plot,kl_plot,word_plot, topo_plot, assm_plot, pnorm_plot, gnorm_plot, beta_plot, wloss_plot, tloss_plot, aloss_plot, closs_plot
