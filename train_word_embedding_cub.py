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

data = torch.load('/data/lzj/easy/word_embedding/cub/train_vocab_vec.pkl')
target = torch.load('/data/lzj/easy/save_tensor/raw_feat/feature_cub_base100_word_embed_relation_res12/save_protos.pkl')
# print(data.shape)
# print(target.shape)
# model_path = '/data/lzj/easy/result/1012/res12_64_word_embed/resnet12_word64_embed_MSEME_backfeat_continue.pkl1'
model = Word_Embedding()
# model_embed = Att_Embedding()
criterion = torch.nn.MSELoss()
def criterion2(output_feature, att_feature):
    output_feature = output_feature / output_feature.norm(dim=1).view(output_feature.shape[0], -1)
    # print(output_feature.shape)
    att_feature = att_feature / att_feature.norm(dim=1).view(att_feature.shape[0], -1)
    cos_simi = (torch.diag(torch.matmul(output_feature, att_feature.transpose(0, 1)))+1)/2
    # print(cos_simi.shape)
    loss_all = 1 - cos_simi
    loss = loss_all.mean(dim=0)
    # print(loss.shape)
    return loss

total_epoch = 500
device = 'cuda:0'
data = data.to(device)
target = target.to(device)

# model_embed.load_state_dict(torch.load(model_path), strict=False)
# model.load_state_dict(torch.load(model_path), strict=False)
# model_embed = model_embed.to(device)
model = model.to(device)
save_path = '/data/lzj/easy/result_word_embedding_model/cubfs/word_embedding_base100_relation_1shot.pkl'

# print(target.min)

# train process
optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=0.0001)
model.train()
for e in range(total_epoch):
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out, target) + 2*criterion2(out, target)
    print("{:d}: loss: {:.5f}, ".format(e, loss))
    loss.backward()
    optimizer.step()
    
torch.save(model.state_dict(), save_path)

# test process
# test_train_relation = torch.load('/data/lzj/easy/word_embedding/cub/test_train_dist.pkl')
test_data = torch.load('/data/lzj/easy/word_embedding/cub/test_vocab_vec.pkl')
test_target = torch.load('/data/lzj/easy/save_tensor/raw_feat/feature_cub_base100_word_embed_relation_res12/novel_save_protos.pkl')
test_data = test_data.to(device)
test_target = test_target.to(device)
data = data.to(device)
model.load_state_dict(torch.load('/data/lzj/easy/result_word_embedding_model/cubfs/word_embedding_base100_relation_1shot.pkl'))
model.eval()

out = model(test_data)
simi = pairwise_cosine_similarity(out.clone(), out.clone())
simi2 = pairwise_euclidean_distance(out.clone(), out.clone())
simi3 = pairwise_euclidean_distance(target.clone(), target.clone())

print(simi)
print(simi2)
print(simi3)
# torch.save(out, '/data/lzj/easy/word_embedding/cub/generate_novel_protos_1shot_relation.pkl')
# print(pairwise_cosine_similarity(out.clone(), out.clone()))
# topk = torch.topk(test_train_relation, k=5, dim=-1, largest=False)

# print(test_train_relation)
# print(topk)



# model_embed.eval()
# for i in range(300):
#     input = test_data[128*i : 128*(i+1)]
#     out_feat = model_embed(input)
#     out_target = data[test_target[128*i : 128*(i+1)]]
#     back_feat = model(out_feat)
#     loss = criterion(out_feat, out_target)
#     print(pairwise_cosine_similarity(input[0].unsqueeze(0).clone(), back_feat[0].unsqueeze(0).clone()))
#     # print(pairwise_cosine_similarity(back_feat.clone(), back_feat.clone()).min(dim=-1)[0])
    
#     print("{:d}: loss: {:.5f}, ".format(i, loss))
# ind = torch.asarray([0,600,1200,4800,7200,12000])
# input = test_data[ind]
# out_feat = model_embed(input)
# # out_target = data[test_target[128*i : 128*(i+1)]]
# back_feat = model(out_feat)
# loss = criterion(input, back_feat)
# # print(pairwise_cosine_similarity(input[0].unsqueeze(0).clone(), back_feat[0].unsqueeze(0).clone()))
# print(pairwise_cosine_similarity(out_feat.clone(), out_feat.clone()))

# print("loss: {:.5f}, ".format(loss))


# out_feat = model_embed(test_data)

# torch.save(out_feat.cpu(), '/data/lzj/easy/tmp/train100_protos.pkl')
