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

data = torch.load('./word_embedding/FC100/train_vocab_vec.pkl')
target = torch.load('./save_tensor/raw_feat/1120/new_feature_fc100_base60_res12_0.5attriloss_0.6relationloss/save_protos_1shot.pkl')
test_data = torch.load('./word_embedding/FC100/test_vocab_vec.pkl')
# test_target = torch.load('/data/lzj/easy_mine/save_tensor/raw_feat/1120/feature_fc100_base60_word_embed_relation_res12/novel_save_protos.pkl')
# model_path = '/data/lzj/easy/result/1027/res12_64_cifarfs_word_embed/resnet12_word64_embed_MSEME.pkl1'
print(data.shape)
# print(test_data.shape)
model = Word_Embedding()
# model_embed = Att_Embedding()
criterion = torch.nn.MSELoss()
# train_vocab_simi = pairwise_cosine_similarity(data.clone(), data.clone())
# # test_train_target_simi = pairwise_cosine_similarity(test_target.clone(), target.clone())
# # np.savetxt(os.path.join('/data/lzj/easy/save_tensor/raw_feat/test_train_cifar_attri_rela_simi.csv'), test_train_vocab_simi.cpu().detach().numpy(), fmt='%.6f')
# # np.savetxt(os.path.join('/data/lzj/easy/save_tensor/raw_feat/test_train_cifar_attri_rela_feat_simi.csv'), test_train_target_simi.cpu().detach().numpy(), fmt='%.6f')
# print(train_vocab_simi)
# new_chunks = []
# sizes = torch.chunk(target, 1)
# new_chunks.append(torch.randperm(sizes[0].shape[0]))
# index_mixup = torch.cat(new_chunks, dim = 0)
# # lam = np.random.beta(2, 2)

# lam = torch.zeros(len(target), 1)
# for i in range(lam.shape[0]):
#     lam[i] = np.random.beta(2, 2)

# data_plus = lam * data + (1-lam) * data[index_mixup]
# target_plus = lam * target + (1-lam) * target[index_mixup]
# data = torch.cat([data, data_plus], dim=0)
# target = torch.cat([target, target_plus], dim=0)

# def criterion2(output_feature, att_feature):
#     output_feature = output_feature / output_feature.norm(dim=1).view(output_feature.shape[0], -1)
#     # print(output_feature.shape)
#     att_feature = att_feature / att_feature.norm(dim=1).view(att_feature.shape[0], -1)
#     cos_simi = (torch.diag(torch.matmul(output_feature, att_feature.transpose(0, 1)))+1)/2
#     # print(cos_simi.shape)
#     loss_all = 1 - cos_simi
#     loss = loss_all.mean(dim=0)
#     # print(loss.shape)
#     return loss 

total_epoch = 200
device = 'cuda:0'
data = data.to(device)
target = target.to(device)

# # model_embed.load_state_dict(torch.load(model_path), strict=False)
# # model.load_state_dict(torch.load(model_path), strict=False)
# # model_embed = model_embed.to(device)
model = model.to(device)
save_path = './result_word_embedding_model/FC100/new_feature_fc100_base60_res12_0.5attriloss_0.6relationloss_1shot.pkl'

# # print(target.min)

# train process
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1)
model.train()
for e in range(total_epoch):
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out, target)
    # loss2 = criterion2(out, target)
    # loss += loss2
    print("{:d}: loss: {:.5f}, ".format(e, loss))
    loss.backward()
    optimizer.step()
    scheduler.step()
    
torch.save(model.state_dict(), save_path)

# test process
# test_train_relation = torch.load('/data/lzj/easy/tmp/test_train_dist.pkl')
test_data = test_data.to(device)
# test_target = test_target.to(device)
data = data.to(device)
model.load_state_dict(torch.load(save_path))
model.eval()

out = model(test_data)
# dist = pairwise_euclidean_distance(out.clone(), target.clone())
# simi = pairwise_cosine_similarity(out.clone(), test_target.clone())
# simi_data = pairwise_cosine_similarity(target.clone(), target.clone())
# print(dist.nanmean(dim=-1))
# print(dist)
# print(simi_data)
# print(simi)
# np.savetxt(os.path.join('/data/lzj/easy/save_tensor/raw_feat/test_cifar_simi.csv'), simi.cpu().detach().numpy(), fmt='%.6f')
torch.save(out, './word_embedding/FC100/0.5attriloss_0.6relationloss_generate_novel_protos_1shot_relation.pkl')


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
