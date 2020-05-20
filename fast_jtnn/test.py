# For checking the pkl file
#
# import os
# import pickle
#
# import torch
# from torch.utils.data import Dataset, DataLoader
# from mol_tree import MolTree
# import numpy as np
# from jtnn_enc import JTNNEncoder
# from mpn import MPN
# from jtmpn import JTMPN
# import cPickle as pickle
# import os, random
#
#
# def main():
#     data_folder ='../data/l1000/max20/processed/'
#     fn= 'tensors-0.pkl'
#     fn = os.path.join(data_folder, fn)
#     with open(fn) as f:
#         #print(f)
#         data = pickle.load(f)
#     print(len(data))
#     print(data[0].smiles)
# if __name__=="__main__":
#     main()
