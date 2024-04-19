from functools import total_ordering
from torchmetrics.functional import pairwise_cosine_similarity, pairwise_euclidean_distance
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from word_embedding import Word_Embedding
from attri_embedding import Att_Embedding
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

data = torch.load('/data/lzj/easy_mine/word_embedding/tiered/train_vocab_vec.pkl')
target = torch.load('/data/lzj/easy_mine/save_tensor/raw_feat/1120/feature_tier_base351_word_embed_relation_res12/save_protos.pkl')
model = Word_Embedding()
model_embed = Att_Embedding()
criterion = torch.nn.MSELoss()
# def criterion(output_feature, att_feature):
#     output_feature = output_feature / output_feature.norm(dim=1).view(output_feature.shape[0], -1)
#     # print(output_feature.shape)
#     att_feature = att_feature / att_feature.norm(dim=1).view(att_feature.shape[0], -1)
#     cos_simi = (torch.diag(torch.matmul(output_feature, att_feature.transpose(0, 1)))+1)/2
#     # print(cos_simi.shape)
#     loss_all = 1 - cos_simi
#     loss = loss_all.mean(dim=0)
#     # print(loss.shape)
#     return loss

total_epoch = 500
device = 'cuda:0'
data = data.to(device)
target = target.to(device)

# model_embed.load_state_dict(torch.load(model_path), strict=False)
# model.load_state_dict(torch.load(model_path), strict=False)
# model_embed = model_embed.to(device)
model = model.to(device)
save_path = '/data/lzj/easy_mine/result_word_embedding_model/tiered-imagenet/word_embedding_base351_1shot.pkl'

# print(target.min)

# # train process
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
model.train()
for e in range(total_epoch):
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out, target)
    print("{:d}: loss: {:.5f}, ".format(e, loss))
    loss.backward()
    optimizer.step()
    
torch.save(model.state_dict(), save_path)

# # test process
# # test_train_relation = torch.load('/data/lzj/easy/word_embedding/miniimagenet/test_train_dist.pkl')
test_data = torch.load('/data/lzj/easy_mine/word_embedding/tiered/test_vocab_vec.pkl')
test_target = torch.load('/data/lzj/easy_mine/save_tensor/raw_feat/1120/feature_tier_base351_word_embed_relation_res12/novel_save_protos.pkl')
test_data = test_data.to(device)
test_target = test_target.to(device)
data = data.to(device)
model.load_state_dict(torch.load(save_path))
model.eval()

out = model(test_data)
simi = pairwise_euclidean_distance(out.clone(), test_target.clone())
print(simi)
simi_new = pairwise_cosine_similarity(test_target.clone(), test_target.clone())
print(simi_new)
# torch.save(out, '/data/lzj/easy/word_embedding/tiered/generate_novel_protos_1shot.pkl')

