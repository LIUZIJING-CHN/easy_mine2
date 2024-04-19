import os
import torch
import numpy as np
from torchmetrics.functional import pairwise_cosine_similarity

train_vocab_vec = torch.load('./word_embedding/cifarfs/train_vocab_vec.pkl')
test_vocab_vec = torch.load('./word_embedding/cifarfs/test_vocab_vec.pkl')

train_protos = torch.load('./save_tensor/raw_feat/1120/new_feature_cifarfs_base64_res12_0.5attriloss_0.5relationloss/save_protos_1shot.pkl')
# test_protos_old = torch.load('/data/lzj/easy_mine/save_tensor/raw_feat/1120/feature_fc100_base60_word_embed_relation_res12/novel_save_protos.pkl')

test_train_vocab_relation = pairwise_cosine_similarity(test_vocab_vec.clone(), train_vocab_vec.clone())

# test_train_vocab_relation = torch.load('/data/lzj/easy_mine/word_embedding/cifarfs/test_train_vec_simi.pkl')

test_protos = torch.zeros(test_train_vocab_relation.shape[0], train_protos.shape[1])
for i in range(test_train_vocab_relation.shape[0]):
    class_simi = test_train_vocab_relation[i]
    top_simi, top_ind = torch.topk(class_simi, dim=-1, k=3, largest=True)
    if top_simi[0] < 0.3:
        continue
    else:
        # 进行softmax权重归一化
        exponential_simi = torch.exp(top_simi)
        exponential_sum = exponential_simi.sum(dim=-1, keepdim=True)
        weight = (exponential_simi / exponential_sum).unsqueeze(1)
        print(weight.shape)
        weighted_protos_sum = (weight * train_protos[top_ind]).sum(dim=0)
        print(weighted_protos_sum.shape)
        test_protos[i] = weighted_protos_sum

# feat_simi = pairwise_cosine_similarity(test_protos.clone(), test_protos_old.clone())
# print(feat_simi)
print(test_protos.shape)
torch.save(test_protos, './word_embedding/cifarfs/0.5attriloss_0.5relationloss_weighted_novel_protos_1shot_relation.pkl')
