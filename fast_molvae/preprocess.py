# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from multiprocessing import Pool

import math, random, sys
from optparse import OptionParser
import cPickle as pickle

from fast_jtnn import *
import rdkit

from datetime import datetime
from tqdm import tqdm

def tensorize(smiles, assm=True):
    mol_tree = MolTree(smiles)
    mol_tree.recover()
    if assm:
        mol_tree.assemble()
        for node in mol_tree.nodes:
            if node.label not in node.cands:
                node.cands.append(node.label)

    del mol_tree.mol
    for node in mol_tree.nodes:
        del node.mol

    return mol_tree

if __name__ == "__main__":

    start = datetime.now()
    print("Start:{}".format(start))

    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = OptionParser()
    parser.add_option("-t", "--train", dest="train_path")
    parser.add_option("-n", "--split", dest="nsplits", default=10)
    parser.add_option("-j", "--jobs", dest="njobs", default=8)
    parser.add_option("-f", "--file_name", dest="fname")

    opts,args = parser.parse_args()
    opts.njobs = int(opts.njobs)

    pool = Pool(opts.njobs)
    num_splits = int(opts.nsplits)

    with open(opts.train_path) as f:
        #data = [line.strip("\r\n ").split()[0] for line in f]
        data = []
        for i, line in enumerate(f):
            data.append(line.strip("\r\n ").split()[0])

    all_data = pool.map(tensorize, data)

    le = (len(all_data) + num_splits - 1) / num_splits

    save_path =''
    for w in opts.train_path.split('/')[:-1]:
        save_path += w +'/'

    for split_id in xrange(num_splits):
        st = split_id * le
        sub_data = all_data[st : st + le]

        with open(save_path+opts.fname+'-%d.pkl' % split_id, 'wb') as f:
            pickle.dump(sub_data, f, pickle.HIGHEST_PROTOCOL)


    print("Finish:{}".format(datetime.now()))
    print("Consume Time:{}".format(datetime.now()-start))
