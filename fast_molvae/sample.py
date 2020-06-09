import torch
import torch.nn as nn

import math, random, sys
import argparse
from fast_jtnn import *
import rdkit

## !! Need for GPU
import gc
## !! Need for GPU

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--nsample', type=int, required=True)
parser.add_argument('--vocab', required=True)
parser.add_argument('--model', required=True)
parser.add_argument('--kind', default="mj")

parser.add_argument('--hidden_size', type=int, default=450)
parser.add_argument('--latent_size', type=int, default=56)
parser.add_argument('--depthT', type=int, default=20)
parser.add_argument('--depthG', type=int, default=3)

args = parser.parse_args()

vocab = [x.strip("\r\n ") for x in open(args.vocab)]
vocab = Vocab(vocab)

if args.kind == "vae":
    model = JTNNVAE(vocab, args.hidden_size, args.latent_size, args.depthT, args.depthG)
elif args.kind =="mj":
    model = JTNNMJ(vocab, args.hidden_size, args.latent_size, args.depthT, args.depthG)

model.load_state_dict(torch.load(args.model))
model = model.cuda()

torch.manual_seed(0)
for i in xrange(args.nsample):
    print model.sample_prior()
    ## !! Need for GPU
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                del obj
        except:
            pass
    torch.cuda.empty_cache()
    ## !! Need for GPU
