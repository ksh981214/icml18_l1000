import torch
from torch.utils.data import Dataset, DataLoader
from mol_tree import MolTree
import numpy as np
from jtnn_enc import JTNNEncoder
from mpn import MPN
from jtmpn import JTMPN
import cPickle as pickle
import os, random

import gc

import pdb
import ctypes

import copy

def delete_ref(obj):
    referrers = gc.get_referrers(obj)
    for referrer in referrers:
        if type(referrer) == dict:
            for key, value in referrer.items():
                if value is obj:
                    referrer[key] = None

class PairTreeFolder(object):

    def __init__(self, data_folder, vocab, batch_size, num_workers=4, shuffle=True, y_assm=True, replicate=None):
        self.data_folder = data_folder
        self.data_files = [fn for fn in os.listdir(data_folder)]
        self.batch_size = batch_size
        self.vocab = vocab
        self.num_workers = num_workers
        self.y_assm = y_assm
        self.shuffle = shuffle

        if replicate is not None: #expand is int
            self.data_files = self.data_files * replicate

    def __iter__(self):
        for fn in self.data_files:
            fn = os.path.join(self.data_folder, fn)
            with open(fn) as f:
                data = pickle.load(f)

            if self.shuffle:
                random.shuffle(data) #shuffle data before batch

            batches = [data[i : i + self.batch_size] for i in xrange(0, len(data), self.batch_size)]
            if len(batches[-1]) < self.batch_size:
                batches.pop()

            dataset = PairTreeDataset(batches, self.vocab, self.y_assm)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, collate_fn=lambda x:x[0])

            for b in dataloader:
                yield b

            del data, batches, dataset, dataloader

class MolTreeFolder(object):

    def __init__(self, data_folder, vocab, batch_size, num_workers=4, shuffle=True, assm=True, replicate=None):
        self.data_folder = data_folder+"pos"
        self.data_files = [fn for fn in os.listdir(data_folder+"pos")]
        self.batch_size = batch_size
        self.vocab = vocab
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.assm = assm

        # #GeneExpression Load
        # with open("../data/l1000/max30/embedding_train_max30.txt") as f:
        #     all_gene=f.readlines()
        # print("Finish gene exp loading")
        # #For Debugging
        #
        # self.all_gene = [list(map(float, emb.split())) for emb in all_gene] #LIST[LIST]
        # print("Len(all_gen) is {}".format(len(self.all_gene)))
        #
        # del all_gene
        # ############################

        if replicate is not None: #expand is int
            self.data_files = self.data_files * replicate

    def __iter__(self):
        idx=0
        for fn in self.data_files:
            fn = os.path.join(self.data_folder, fn)
            with open(fn) as f:
                data = pickle.load(f)

            # ####################
            # num_pos = len(data)
            # #pdb.set_trace()
            # gene = self.all_gene[idx:idx+num_pos]
            # idx = idx + num_pos
            # gene = gene * (5+1)
            # ####################
            #print(len(data))
            if self.shuffle:
                random.shuffle(data) #shuffle data before batch

            batches = [data[i : i + self.batch_size] for i in xrange(0, len(data), self.batch_size)]
            #print(len(batches))
            if len(batches[-1]) < self.batch_size:
                batches.pop()

            # gene_batches = [gene[i : i + self.batch_size] for i in xrange(0, len(gene), self.batch_size)]
            # if len(gene_batches[-1]) < self.batch_size:
            #     gene_batches.pop()

            dataset = MolTreeDataset(batches, self.vocab, self.assm)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, collate_fn=lambda x:x[0])

            #####
            # for i,b in enumerate(dataloader):
            #     yield b, gene_batches[i]
            #
            # del data, batches, dataset, dataloader, gene
            for i,b in enumerate(dataloader):
                yield b

            del data, batches, dataset, dataloader
            ##########

class MolTreeFolderMJ(object):
    def __init__(self, data_folder, num_neg_folder, vocab, batch_size, num_workers=4, shuffle=False, assm=True, replicate=None):
        self.data_folder = data_folder+"pos"
        self.data_files = [fn for fn in os.listdir(data_folder+"pos")]

        self.num_of_neg_folder = num_neg_folder
        self.neg_data_folders = [data_folder + str(i)+"_neg" for i in range(num_neg_folder)]
        self.neg_data_files={}
        for folder_name in self.neg_data_folders:
            neg_data_files = [fn for fn in os.listdir(folder_name)]
            # {0_neg: [], 1_neg:[]
            self.neg_data_files[folder_name] = neg_data_files

        self.gene_folder = data_folder+"gene"
        self.gene_files = [fn for fn in os.listdir(data_folder+"gene")]

        self.batch_size = batch_size
        self.vocab = vocab
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.assm = assm

        #GeneExpression Load
        # with open(gene_exp) as f:
        #     all_gene=f.readlines()
        # print("Finish gene exp loading")
        #For Debugging

        # self.all_gene = [list(map(float, emb.split())) for emb in all_gene] #LIST[LIST]
        # print("Len(all_gen) is {}".format(len(self.all_gene)))
        #
        # del all_gene

        if replicate is not None: #expand is int
            self.data_files = self.data_files * replicate

    def __iter__(self):
        idx =0
        for data_files_i, fn in enumerate(self.data_files):
            data = []
            fn = os.path.join(self.data_folder, fn)
            #pdb.set_trace()
            with open(fn) as f:
                data = pickle.load(f)

            num_pos = len(data)
            #pdb.set_trace()
            for neg_folder_name in self.neg_data_folders:
                fn = os.path.join(neg_folder_name, self.neg_data_files[neg_folder_name][data_files_i])
                with open(fn) as f:
                    data = data + pickle.load(f)
            #pdb.set_trace()

            #pdb.set_trace()
            fn = os.path.join(self.gene_folder, self.gene_files[data_files_i])
            #pdb.set_trace()
            with open(fn, 'rb') as f:
                gene = pickle.load(f)

            gene = gene * (self.num_of_neg_folder+1)
            #pdb.set_trace()
            #For Debugging
            if len(data) != len(gene):
                print("len(data) != len(gene)")

            #make label
            num_neg = len(data)-num_pos
            label = [1.0] * num_pos + [-1.0] * num_neg
            #pdb.set_trace()
            if self.shuffle:
                indices = np.arange(len(data))
                np.random.shuffle(indices)

                data = list(np.array(data)[indices])
                gene = list(np.array(gene)[indices])
                label= list(np.array(label)[indices])

                del indices

            #pdb.set_trace()
            batches = [data[i : i + self.batch_size] for i in xrange(0, len(data), self.batch_size)]
            if len(batches[-1]) < self.batch_size:
                batches.pop()

            gene_batches = [gene[i : i + self.batch_size] for i in xrange(0, len(gene), self.batch_size)]
            if len(gene_batches[-1]) < self.batch_size:
                gene_batches.pop()

            label_batches = [label[i : i + self.batch_size] for i in xrange(0, len(label), self.batch_size)]
            if len(label_batches[-1]) < self.batch_size:
                label_batches.pop()

            dataset = MolTreeDataset(batches, self.vocab, self.assm)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, collate_fn=lambda x:x[0])

            #pdb.set_trace()

            for i,b in enumerate(dataloader):
                yield b, gene_batches[i], label_batches[i]

            #pdb.set_trace()

            #pdb.set_trace()
            del data, gene, label, batches, gene_batches, label_batches, dataset, dataloader
            #pdb.set_trace()

            # #delete element of neg_data
            # idx_lst=[]
            # for i,_ in enumerate(neg_data):
            #     #print(i)
            #     delete_ref(neg_data[i])
            #     idx_lst.append(i)
            # for _ in idx_lst:
            #     del neg_data[0]

            # delete_ref(data)
            # delete_ref(neg_data)
            #
            # del data, neg_data
            #
            # delete_ref(indices)
            # del indices
            #
            # delete_ref(gene)
            # delete_ref(label)
            # del gene, label
            #
            # delete_ref(gene_batches)
            # delete_ref(label_batches)
            # del gene_batches, label_batches
            #
            # delete_ref(all_data)
            # delete_ref(dataloader)
            # del all_data, dataloader
            #
            # delete_ref(dataset)
            # del dataset
            #
            # delete_ref(batches)
            # del batches

            #pdb.set_trace()


class PairTreeDataset(Dataset):

    def __init__(self, data, vocab, y_assm):
        self.data = data
        self.vocab = vocab
        self.y_assm = y_assm

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        batch0, batch1 = zip(*self.data[idx])
        return tensorize(batch0, self.vocab, assm=False), tensorize(batch1, self.vocab, assm=self.y_assm)

class MolTreeDataset(Dataset):

    def __init__(self, data, vocab, assm=True):
        self.data = data
        self.vocab = vocab
        self.assm = assm

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return tensorize(self.data[idx], self.vocab, assm=self.assm)

def tensorize(tree_batch, vocab, assm=True):
    set_batch_nodeID(tree_batch, vocab)
    smiles_batch = [tree.smiles for tree in tree_batch]
    jtenc_holder,mess_dict = JTNNEncoder.tensorize(tree_batch)
    jtenc_holder = jtenc_holder
    mpn_holder = MPN.tensorize(smiles_batch)

    if assm is False:
        return tree_batch, jtenc_holder, mpn_holder

    cands = []
    batch_idx = []
    for i,mol_tree in enumerate(tree_batch):
        for node in mol_tree.nodes:
            #Leaf node's attachment is determined by neighboring node's attachment
            if node.is_leaf or len(node.cands) == 1: continue
            cands.extend( [(cand, mol_tree.nodes, node) for cand in node.cands] )
            batch_idx.extend([i] * len(node.cands))

    jtmpn_holder = JTMPN.tensorize(cands, mess_dict)
    batch_idx = torch.LongTensor(batch_idx)

    return tree_batch, jtenc_holder, mpn_holder, (jtmpn_holder,batch_idx)

def set_batch_nodeID(mol_batch, vocab):
    tot = 0
    for mol_tree in mol_batch:
        for node in mol_tree.nodes:
            node.idx = tot
            node.wid = vocab.get_index(node.smiles)
            tot += 1
