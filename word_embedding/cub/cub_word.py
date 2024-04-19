from tkinter import simpledialog
import gensim
import numpy as np
import torch
from gensim.models import Word2Vec, KeyedVectors

def calculate_cosine_similarity_matrix(h_emb, eps=1e-8):
    # h_emb (N, M)
    # normalize
    a_n = h_emb.norm(dim=1).unsqueeze(1)
    a_norm = h_emb / torch.max(a_n, eps * torch.ones_like(a_n))

    # cosine similarity matrix
    sim_matrix = torch.einsum('bc,cd->bd', a_norm, a_norm.transpose(0,1))
    return sim_matrix

train_ind = torch.asarray([18,170,107,98,177,182,5,146,12,152,61,125,180,154,80,7,33,130,37,74,183,145,45,159,60,123,179,185,122,44,16,55,150,111,22,189,129,4,83,106,134,66,26,113,168,63,8,75,118,143,71,124,184,97,149,24,30,160,40,56,131,96,181,19,153,92,54,163,51,86,139,90,137,101,144,89,109,14,27,141,187,46,138,195,108,62,2,59,136,197,43,10,194,73,196,178,175,126,93,112])

test_ind = torch.asarray([132,173,17,38,133,53,157,128,34,28,114,151,31,166,127,176,32,142,169,147,29,99,82,79,115,148,193,72,77,25,165,81,188,174,190,39,58,140,88,70,87,36,21,9,103,67,192,117,47,172])
cub_np = np.loadtxt('/data/lzj/easy/CUB_200_2011/CUB_200_2011/attributes/class_attribute_labels_continuous.txt', dtype=np.float32)
cub = torch.from_numpy(cub_np)
cub_train = cub[train_ind]
cub_train /= cub_train.norm(dim=-1).unsqueeze(1)
cub_train_relation = calculate_cosine_similarity_matrix(cub_train)

cub_test = cub[test_ind]
cub_test /= cub_test.norm(dim=-1).unsqueeze(1)
# torch.save(cub_train, '/data/lzj/easy/word_embedding/cub/train_vocab_vec.pkl')
# torch.save(cub_train_relation, '/data/lzj/easy/word_embedding/cub/train_vec_simi.pkl')
torch.save(cub_test, '/data/lzj/easy/word_embedding/cub/test_vocab_vec.pkl')

