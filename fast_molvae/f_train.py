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

from datetime import datetime
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os # for save plot

#for debug
from pympler import muppy, summary
import pandas as pd

def save_KL_plt(save_dir, epoch, x, kl):
    plt.plot(x, kl)
    plt.xlabel('Iteration')
    plt.ylabel('KL divergence')
    plt.grid()
    plt.savefig('./plot/{}/KL/epoch_{}.png'.format(str(save_dir),str(epoch)))
    plt.close()
def save_Acc_plt(save_dir, epoch, x, word, topo, assm):
    plt.plot(x, word)
    plt.plot(x, topo)
    plt.plot(x, assm)
    plt.xlabel('Iteration')
    plt.ylabel('Acc')
    plt.legend(['Word acc','Topo acc','Assm acc'])
    plt.grid()
    plt.savefig('./plot/{}/Acc/epoch_{}.png'.format(str(save_dir),str(epoch)))
    plt.close()
def save_Norm_plt(save_dir, epoch, x, pnorm, gnorm):
    plt.plot(x, pnorm)
    plt.plot(x, gnorm)
    plt.xlabel('Iteration')
    plt.ylabel('Norm')
    plt.legend(['Pnorm', 'Gnorm'])
    plt.grid()
    plt.savefig('./plot/{}/Norm/epoch_{}.png'.format(str(save_dir),str(epoch)))
    plt.close()

def save_Loss_plt(save_dir, epoch, x, wloss, tloss, aloss):
    plt.plot(x, wloss)
    plt.plot(x, tloss)
    plt.plot(x, aloss)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend(['Word Loss', 'Topo Loss','Assm Loss'])
    plt.grid()
    plt.savefig('./plot/{}/Loss/epoch_{}.png'.format(str(save_dir),str(epoch)))
    plt.close()

def save_Beta_plt(save_dir, epoch, x, beta):
    plt.plot(x, beta)
    plt.xlabel('Iteration')
    plt.ylabel('Beta')
    plt.grid()
    plt.savefig('./plot/{}/Beta/epoch_{}.png'.format(str(save_dir),str(epoch)))
    plt.close()

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--train', required=True)
parser.add_argument('--gene', required=True)
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
#parser.add_argument('--save_iter', type=int, default=5000)


args = parser.parse_args()
print args

'''
    model initializing
'''
vocab = [x.strip("\r\n ") for x in open(args.vocab)]
vocab = Vocab(vocab)
model = JTNNVAEMLP(vocab, args.hidden_size, args.latent_size, args.depthT, args.depthG).cuda()

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
    #print(len(list(set(vocab.vocab) & set(pre_vocab.vocab)))) #270
    pre_vocab_dict = {v:k for k,v in enumerate(pre_vocab.vocab)}
    vocab_dict = {v:k for k,v in enumerate(vocab.vocab)}
    for w in vocab.vocab:
        if w in pre_vocab_dict:
            model.state_dict()['decoder.embedding.weight'][vocab_dict[w]] = pre_model.state_dict()['decoder.embedding.weight'][pre_vocab_dict[w]]
            model.state_dict()['jtnn.embedding.weight'][vocab_dict[w]] = pre_model.state_dict()['jtnn.embedding.weight'][pre_vocab_dict[w]]
    print("Finish Embedding Loading")

    del(pre_vocab, pre_model, pre_model_dict, clear_pre_model_dict, pre_vocab_dict)
print "Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,)

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)
# scheduler.step()

param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

total_step = 0
beta = args.beta
meters = np.zeros(7)

d=datetime.now()
now = str(d.year)+'_'+str(d.month)+'_'+str(d.day)+'_'+str(d.hour)+'_'+str(d.minute)

if args.load_epoch != 0:
    folder_name = "f_" + "h" + str(args.hidden_size) + '_' + "bs" + str(args.batch_size) + '_' + now + "_from_" +str(args.load_epoch)
else:
    folder_name = "f_" + "h" + str(args.hidden_size) + '_' + "bs" + str(args.batch_size) + '_' + now

# os.makedirs('./plot/'+folder_name+'/KL')        #KL
# os.makedirs('./plot/'+folder_name+'/Acc')       #Word, Topo, Assm
# os.makedirs('./plot/'+folder_name+'/Norm')      #PNorm, GNorm
# os.makedirs('./plot/'+folder_name+'/Loss')      #Word, Topo, Assm LOSS
# os.makedirs('./plot/'+folder_name+'/Beta')      #Word, Topo, Assm LOSS
# print("...Finish Making Plot Folder...")

for epoch in xrange(args.epoch):
    # x_plot,kl_plot,word_plot, topo_plot, assm_plot, pnorm_plot, gnorm_plot, beta_plot, wloss_plot, tloss_plot, aloss_plot =[],[],[],[],[],[],[],[],[],[],[]
    meters *= 0

    start = datetime.now()
    print("EPOCH: %d | TIME: %s " % (epoch+1, str(start)))

    loader = MolTreeFolderMLP(args.train, args.gene, vocab, args.batch_size, num_workers=4)
    for it, (batch, gene_batch) in enumerate(loader):
        total_step += 1
        try:
            model.zero_grad()
            loss, kl_div, wacc, tacc, sacc, word_loss, topo_loss, assm_loss = model(batch, gene_batch, beta)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()
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

            print "[%d][%d] Beta: %.6f, KL: %.2f, Word: %.2f, Topo: %.2f, Assm: %.2f, PNorm: %.2f, GNorm: %.2f" % (epoch, it+1, beta, meters[0], meters[1], meters[2], meters[3], pnorm, gnorm)
            print "Wloss: %.2f, Tloss: %.2f, Aloss: %.2f" %(meters[4], meters[5], meters[6])

            # x_plot.append(it+1)
            # kl_plot.append(meters[0])
            # word_plot.append(meters[1])
            # topo_plot.append(meters[2])
            # assm_plot.append(meters[3])
            # pnorm_plot.append(pnorm)
            # gnorm_plot.append(gnorm)
            # beta_plot.append(beta)
            # wloss_plot.append(meters[4])
            # tloss_plot.append(meters[5])
            # aloss_plot.append(meters[6])

            sys.stdout.flush()
            meters *= 0

        if total_step % args.anneal_iter == 0:
            scheduler.step()
            print "learning rate: %.6f" % scheduler.get_lr()[0]

        if total_step % args.kl_anneal_iter == 0 and total_step >= args.warmup:
            beta = min(args.max_beta, beta + args.step_beta)

        if (it+1) % 1000 == 0:
            all_objects = muppy.get_objects()
            sum1 = summary.summarize(all_objects)
            # Prints out a summary of the large objects
            summary.print_(sum1)
            # Get references to certain types of objects such as dataframe
            dataframes = [ao for ao in all_objects if isinstance(ao, pd.DataFrame)]
            for d in dataframes:
                print d.columns.values
                print len(d)

    if args.load_epoch != 0:
        torch.save(model.state_dict(), args.save_dir + "/model.iter-" + str(epoch+args.load_epoch))
    else:
        torch.save(model.state_dict(), args.save_dir + "/model.iter-" + str(epoch))

    #Plot per 1 epoch
    print "Cosume Time per Epoch %s" % (str(datetime.now()-start))
    # save_KL_plt(folder_name, epoch, x_plot, kl_plot)
    # save_Acc_plt(folder_name, epoch, x_plot, word_plot, topo_plot, assm_plot)
    # save_Norm_plt(folder_name, epoch, x_plot, pnorm_plot, gnorm_plot)
    # save_Loss_plt(folder_name, epoch, x_plot, wloss_plot, tloss_plot, aloss_plot)
    # save_Beta_plt(folder_name, epoch, x_plot, beta_plot)
    # del(x_plot,kl_plot,word_plot, topo_plot, assm_plot, pnorm_plot, gnorm_plot, beta_plot, wloss_plot, tloss_plot, aloss_plot)
