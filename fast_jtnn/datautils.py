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

        if replicate is not None: #expand is int
            self.data_files = self.data_files * replicate

    def __iter__(self):
        idx=0
        for fn in self.data_files:
            fn = os.path.join(self.data_folder, fn)
            with open(fn) as f:
                data = pickle.load(f)

            if self.shuffle:
                random.shuffle(data) #shuffle data before batch

            batches = [data[i : i + self.batch_size] for i in xrange(0, len(data), self.batch_size)]

            # print(len(batches))
            # for x in batches:
            #     if len(x) != 1:
            #         print(len(x))

            if len(batches) != 0:
                if len(batches[-1]) < self.batch_size:
                    batches.pop()
                dataset = MolTreeDataset(batches, self.vocab, self.assm)
                dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, collate_fn=lambda x:x[0] if x[0] is not None else x)

                for b in dataloader:
                    yield b
            else:
                continue

            del data, batches, dataset, dataloader

class MolTreeFolderMJ(object):
    def __init__(self, data_folder, num_neg_folder, vocab, batch_size, num_workers=4, shuffle=True, assm=True, replicate=None, test=False):

        self.data_folder = data_folder+"pos"
        self.gene_folder = data_folder+"gene"
        self.data_files = [fn for fn in os.listdir(self.data_folder)]

        if not test:
            self.num_of_neg_folder = num_neg_folder
            self.neg_data_folders = [data_folder + str(i)+"_neg" for i in range(num_neg_folder)]

        self.batch_size = batch_size
        self.vocab = vocab
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.assm = assm
        self.test = test

        if replicate is not None: #expand is int
            self.data_files = self.data_files * replicate

    def __iter__(self):
        idx =0
        for data_files_i, fn in enumerate(self.data_files):
            data = []
            pos_fn = os.path.join(self.data_folder, fn)
            with open(pos_fn) as pos_f:
                data = pickle.load(pos_f)
            num_pos = len(data)

            if not self.test:
                for neg_folder_name in self.neg_data_folders:
                    neg_fn = os.path.join(neg_folder_name, fn)
                    with open(neg_fn) as neg_f:
                        data = data + pickle.load(neg_f)

            gene_fn = os.path.join(self.gene_folder, "gene-"+fn.split("-")[1])
            with open(gene_fn, 'rb') as gene_f:
                gene = pickle.load(gene_f)

            if not self.test:
                gene = gene * (self.num_of_neg_folder+1)

            #For Debugging
            if len(data) != len(gene):
                print(len(data), len(gene))
                print("len(data) != len(gene)")

            #make label
            num_neg = len(data)-num_pos
            label = [1.0] * num_pos + [-1.0] * num_neg

            if self.shuffle:
                indices = np.arange(len(data))
                np.random.shuffle(indices)

                data = list(np.array(data)[indices])
                gene = list(np.array(gene)[indices])
                label= list(np.array(label)[indices])

                del indices

            batches = [data[i : i + self.batch_size] for i in xrange(0, len(data), self.batch_size)]
            gene_batches = [gene[i : i + self.batch_size] for i in xrange(0, len(gene), self.batch_size)]
            label_batches = [label[i : i + self.batch_size] for i in xrange(0, len(label), self.batch_size)]

            if len(batches) != 0 and len(gene_batches) != 0 and len(label_batches) != 0:
                if len(batches[-1]) < self.batch_size:
                    batches.pop()
                if len(gene_batches[-1]) < self.batch_size:
                    gene_batches.pop()
                if len(label_batches[-1]) < self.batch_size:
                    label_batches.pop()

                dataset = MolTreeDataset(batches, self.vocab, self.assm)
                dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, collate_fn=lambda x:x[0])

                for i,b in enumerate(dataloader):
                    yield b, gene_batches[i], label_batches[i]

            else:
                continue #pass the file

            del data, gene, label, batches, gene_batches, label_batches, dataset, dataloader

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
        try:
            return tensorize(self.data[idx], self.vocab, assm=self.assm)
        except Exception as e:
            print e
            pass

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
