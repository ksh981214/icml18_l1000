# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from torch.utils.data import DataLoader


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
parser.add_argument('--test', required=True)
parser.add_argument('--vocab', required=True)
parser.add_argument('--trained_model', required=True)

parser.add_argument('--hidden_size', type=int, default=450)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--latent_size', type=int, default=56) #h_T, h_G = 28, 28
parser.add_argument('--depthT', type=int, default=20)
parser.add_argument('--depthG', type=int, default=3)


parser.add_argument('--beta', type=float, default=0.0)


parser.add_argument('--print_iter', type=int, default=50)

## !! Need for GPU
parser.add_argument('--debug', type=int, default=1)
import gc
## !! Need for GPU
## For plot
parser.add_argument('--plot', type=int, default=0)
parser.add_argument('--mode', type=int, default=-1) #0: generated, 1: reconstructed
#parser.add_argument('--make_generated', type=int, default=0)
#parser.add_argument('--make_reconstructed', type=int, default=0)

parser.add_argument('--sample_num', type=int, default=1)

## For plot

args = parser.parse_args()
print args

'''
    model initializing
'''
vocab = [x.strip("\r\n ") for x in open(args.vocab)]
vocab = Vocab(vocab)
model = JTNNMJ(vocab, args.hidden_size, args.latent_size, args.depthT, args.depthG).cuda()

model.load_state_dict(torch.load(args.trained_model))
print("load model.iter-{}".format(args.trained_model))

print "Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,)


total_step = 0
beta = args.beta

accs=np.zeros(4)
losses = np.zeros(4)

if args.plot:
    import os # for save plot
    x_plot,kl_plot,word_plot, topo_plot, assm_plot, wloss_plot, tloss_plot, aloss_plot =[],[],[],[],[],[],[],[]
    d=datetime.now()
    now = str(d.year)+'_'+str(d.month)+'_'+str(d.day)+'_'+str(d.hour)+'_'+str(d.minute)

    folder_name = "mj_test_" + now
    os.makedirs('./plot/'+folder_name+'/KL')        #KL
    os.makedirs('./plot/'+folder_name+'/Acc')       #Word, Topo, Assm
    os.makedirs('./plot/'+folder_name+'/Loss')      #Word, Topo, Assm LOSS
    print("...Finish Making Plot Folder...")


accs *= 0
losses *= 0

start = datetime.now()
loader = MolTreeFolderMJ(args.test, -1, vocab, args.batch_size, num_workers=4, test=True)
reproduce_cnt = 0
for (batch, g, l) in loader:
    try:
        mode = args.mode
        sample_num = args.sample_num

        if mode != -1:
            loss, kl_div, wacc, tacc, sacc, word_loss, topo_loss, assm_loss, cos_loss, (original_SMILE,made_SMILE) = model(batch, g, l, beta, mode=mode, n=sample_num)
            print(original_SMILE, made_SMILE)
            reproduce_cnt = reproduce_cnt+1
        else:
            loss, kl_div, wacc, tacc, sacc, word_loss, topo_loss, assm_loss, cos_loss = model(batch, g, l, beta)

        total_step += 1
        accs = accs + np.array([kl_div, wacc * 100, tacc * 100, sacc * 100])
        losses = losses + np.array([word_loss, topo_loss, assm_loss, cos_loss])

    except Exception as e:
        print e
        continue

    if total_step % args.print_iter == 0:
        accs /= args.print_iter
        losses /= args.print_iter

        print "[%d] KL: %.2f, Word: %.2f, Topo: %.2f, Assm: %.2f" % (total_step, accs[0], accs[1], accs[2], accs[3])
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

accs /= total_step%args.print_iter
losses /= total_step%args.print_iter

x_plot.append(total_step)
kl_plot.append(accs[0])
word_plot.append(accs[1])
topo_plot.append(accs[2])
assm_plot.append(accs[3])
wloss_plot.append(losses[0])
tloss_plot.append(losses[1])
aloss_plot.append(losses[2])

print("The number of generated molecule is {} ".format(reproduce_cnt))


if args.plot:
    save_KL_plt(folder_name, 0, x_plot, kl_plot)
    save_Acc_plt(folder_name, 0, x_plot, word_plot, topo_plot, assm_plot)
    save_Loss_plt(folder_name, 0, x_plot, wloss_plot, tloss_plot, aloss_plot)
