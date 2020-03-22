import torch
import torch.nn as nn
from torch.autograd import Variable

import math, random, sys
from optparse import OptionParser

import rdkit
import rdkit.Chem as Chem
from rdkit.Chem import Draw

import numpy as np
from fast_jtnn import *

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = OptionParser()
parser.add_option("-v", "--vocab", dest="vocab_path")
parser.add_option("-m", "--model", dest="model_path")
parser.add_option("-w", "--hidden", dest="hidden_size", default=450)
parser.add_option("-l", "--latent", dest="latent_size", default=56)
parser.add_option("-d", "--depthT", dest="depthT", default=20)
parser.add_option("-d", "--depthG", dest="depthG", default=3)
parser.add_option("-n", "--sample", dest="sample_num", default=7) # odd number
opts,args = parser.parse_args()

sample_num = opts.sample_num
if sample_num % 2 == 0:
    print("Please sample odd number")
    exit()

vocab = [x.strip("\r\n ") for x in open(opts.vocab_path)]
vocab = Vocab(vocab)

hidden_size = int(opts.hidden_size)
latent_size = int(opts.latent_size)
depthT = int(opts.depthT)
depthG = int(opts.depthG)

model = JTNNVAEMLP(vocab, args.hidden_size, args.latent_size, args.depthT, args.depthG).cuda()
model.load_state_dict(torch.load(opts.model_path))
model = model.cuda()

np.random.seed(0)
x = np.random.randn(latent_size)
x /= np.linalg.norm(x)

y = np.random.randn(latent_size)
y -= y.dot(x) * x
y /= np.linalg.norm(y)

#z0 = "CN1C(C2=CC(NC3C[C@H](C)C[C@@H](C)C3)=CN=C2)=NN=C1"
z0 = "COC1=CC(OC)=CC([C@@H]2C[NH+](CCC(F)(F)F)CC2)=C1"\
'''
    # TODO: GENE EXPRESSION을 어떻게 섞을 지 생각해야한다.
    1. gene exp를 애초에 주거나/안주거나
    2. z smile을 주나 안주나
'''
z0 = model.encode_latent_from_smiles([z0]).squeeze()
z0 = z0.data.cpu().numpy()

delta = 1
nei_mols = []
range = int(sample_num/2)
for dx in xrange(-1*range, range + 1):
    for dy in xrange(-1*range, range + 1):
        z = z0 + x * delta * dx + y * delta * dy
        tree_z, mol_z = torch.Tensor(z).unsqueeze(0).chunk(2, dim=1)
        tree_z, mol_z = create_var(tree_z), create_var(mol_z)
        nei_mols.append( model.decode(tree_z, mol_z, prob_decode=False) )

nei_mols = [Chem.MolFromSmiles(s) for s in nei_mols]
img = Draw.MolsToGridImage(nei_mols, molsPerRow=sample_num)

#print img
img.save(z0)
