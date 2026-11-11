# -*- coding: utf-8 -*-

import argparse
import numpy as np
import scipy.sparse as sp
import torch
import tools as tl
import os

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, default="sports")
parser.add_argument("-s", "--sampling-size", type=int, default=None)
parser.add_argument("-m", "--modal", type=str, default="image", help="modal type: image/text")
parser.add_argument("-t", "--threshold", type=str, default="10", help="threshold: 10/15")

args = parser.parse_args()
K = args.sampling_size
modal = args.modal
threshold = args.threshold

raw_dir = "../../data/" + args.dataset + "/"


adj_file = f"{modal}_adj_{threshold}.pt"
adj_path = os.path.join(raw_dir, adj_file)

if not os.path.exists(adj_path):
    raise FileNotFoundError(f"Adjacency matrix file {adj_file} not found in {raw_dir}")

A = torch.load(adj_path)
A = A.coalesce()
A = sp.csr_matrix(
    (
        A.values().cpu().numpy(),
        (A.indices()[0].cpu().numpy(), A.indices()[1].cpu().numpy()),
    ),
    shape=A.size(),
)

if K == 1:
    I = tl.get_mutual_information(A)
    true_sample_index = tl.get_sim(I)
    output_file = f"{modal}_true_sample_index_{threshold}"
    np.savetxt(os.path.join(raw_dir, output_file), true_sample_index, fmt="%d")

else:
    if K is not None:
        I = tl.get_mutual_information(A)
        I = I.tocoo()
        I.data = np.exp(I.data)
        I = I.multiply(1 / I.sum(axis = 1))
        I = I.tocsr()
        true_sample_index = tl.get_mutli_sim(I, ratio = 1, K = K)
        
        row = []
        col = []
        i = 0
        for indexes in true_sample_index:
            
            for j in indexes:
                
                row.append(i)
                col.append(j)
                
            i += 1
        
        N = len(row)
           
        mask = sp.coo_matrix((np.ones(N), (row, col)), shape = A.shape).tocsr()
        A = A.multiply(mask)
    
    A = tl.get_L_DA(A)
    
    A = A.tocoo()
    
    rows = A.row
    cols = A.col
    values = A.data
    indices = np.vstack((rows, cols))

    indices = torch.LongTensor(indices)
    values = torch.FloatTensor(values)
    size = torch.Size(A.shape)

    A = torch.sparse_coo_tensor(indices, values, size)


    output_file = f"adj_tensor_sampling_{threshold}.pt"
    torch.save(A, os.path.join(raw_dir, output_file))

