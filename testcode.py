import torch
import numpy as np
# import torch.nn.functional as F
from torchmetrics.functional import pairwise_cosine_similarity, pairwise_euclidean_distance
from einops import rearrange
import os

# from torchvision.models.resnet import resnet50
# model = resnet50(pretrained=False)

# ckpt = torch.load('/data/lzj/easy/pretrained_weight/best_3.pth')
# print(ckpt['model'].keys())
# # print(ckpt['state_dict'].keys())

# msg = model.load_state_dict(ckpt['model'], strict=False)
# # msg = model.load_state_dict({k.replace('module.momentum_encoder.',''):v for k,v in ckpt['state_dict'].items()}, strict=False)

# # assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

# # remove the fully-connected layer
# model.fc = torch.nn.Identity()
# print(model.state_dict().keys())
# # print(feature.shape)
# # np.savetxt(os.path.join('/data/lzj/easy/save_tensor/raw_feat/proto_feat_M_sort_1.csv'), feature.cpu().numpy(), fmt='%.6f')
# # x = torch.randperm(1, 10)
# # print(x)


# if no_trainset_tag:
#                 batch_sz, dim = features.shape
#                 task_classifier = nn.Linear(dim, args.n_ways).to(args.device)
#                 features_resize = features.view(args.n_ways, -1, dim)
#                 target_resize = target.view(args.n_ways, -1)
                
#                 support_feature = features_resize[:, :1, ... ].squeeze()
#                 support_target = target_resize[:, :1].squeeze()   
#                 query_feature = features_resize[:, 1:, ... ].squeeze()
#                 query_target = target_resize[:, 1:].squeeze() 
#                 optimizer = torch.optim.SGD(task_classifier.parameters(), lr=0.05, momentum=0.9, weight_decay=0.0001)
#                 ##### meta-training(finetune)
#                 for i in range(50):
#                     # print(support_feature.shape)
#                     out = task_classifier(support_feature)
#                     # print(out.shape)
#                     # loss = nn.CrossEntropyLoss()(out, support_target)
#                     # optimizer.zero_grad()
#                     # loss.backward()
#                     # optimizer.step()
                
                
#                 #### testing
#                 with torch.no_grad():
#                     out = task_classifier(query_feature)
#                     prediction = out.argmax(dim=-1)
# x = torch.load('/data/lzj/easy/save_tensor/raw_feat/proto_feat_1.pkl')
# print(x.shape)
# x_new = torch.nonzero(x>=0.9)
# print(x_new.shape)

# x_ind = torch.zeros_like(x)
# row = 0
# column = 0
# for i in range(x_new.shape[0]):
#     if x_new[i, 0] == row:
#         x_ind[row, column] = x_new[i, 1]
#         column += 1
#     else:
#         column = 0
#         row += 1
#         x_ind[row, column] = x_new[i, 1]
# print(x_ind)
# np.savetxt('/data/lzj/easy/thre0.9ind.csv', x_ind.numpy())

# train_vec = torch.load('/data/lzj/easy/word_embedding/miniimagenet/train_vocab_vec.pkl')
# print(train_vec)
# print(train_vec['linear.weight'].shape)
# print(train_vec['linear.bias'].shape)

# print(train_vec['linear_rot.weight'].shape)
# print(train_vec['linear_rot.bias'].shape)

# print(train_vec['linear_att.weight'].shape)-
# print(train_vec['linear_att.bias'].shape)

# test_vec = torch.load('/data/lzj/easy/tmp/test_vocab_vec.pkl')
# minus = (train_vec[0] * -1).unsqueeze(0)
# train_1 = train_vec[0].unsqueeze(0)
# simi = pairwise_cosine_similarity(train_vec[0].unsqueeze(0).clone(), train_vec[0].unsqueeze(0).clone())
# print(simi)
# np.savetxt(os.path.join('/data/lzj/easy/save_tensor/raw_feat/train_cifar_word_cos.csv'), simi.cpu().numpy(), fmt='%.6f')
# # # print(simi.shape)
# print(simi)
# topk = torch.topk(simi, k=5, dim=-1)
# print(topk[1])
# print(topk[0])

# train100_vec = torch.load('/data/lzj/easy/tmp/train100_vocab_vec.pkl')
# simi_relation = (pairwise_cosine_similarity(train100_vec.clone(), train100_vec.clone())+1)/2
# print(simi_relation.shape)
# print(simi_relation)

def calculate_cosine_similarity_matrix(h_emb, eps=1e-8):
    # h_emb (N, M)
    # normalize
    a_n = h_emb.norm(dim=1).unsqueeze(1)
    a_norm = h_emb / torch.max(a_n, eps * torch.ones_like(a_n))

    # cosine similarity matrix
    sim_matrix = torch.einsum('bc,cd->bd', a_norm, a_norm.transpose(0,1))
    return sim_matrix

# # torch.save(train_vocab_vec, '/data/lzj/easy/word_embedding/cifarfs/train_vocab_vec.pkl')
# # torch.save(test_vocab_vec, '/data/lzj/easy/word_embedding/cifarfs/test_vocab_vec.pkl')
# a = torch.load('/data/lzj/easy/save_tensor/raw_feat/feature_cifarfs_base64_word_embed_relation_res12/save_protos.pkl')
# b = torch.load('/data/lzj/easy/save_tensor/raw_feat/feature_cifarfs_base64_word_embed_relation_res12/novel_save_protos.pkl')
# class_relation = pairwise_cosine_similarity(b.clone(), a.clone())
# class_relation2 = pairwise_euclidean_distance(b.clone(), a.clone())

# # torch.save(class_relation, '/data/lzj/easy/word_embedding/cifarfs/train_vec_simi.pkl')
# # print(cos_simi1.nanmean(dim=-1))
# # print(cos_simi2.nanmean(dim=-1))
# np.savetxt(os.path.join('/data/lzj/easy/save_tensor/raw_feat/test_train_relation_cifar_cos.csv'), class_relation.numpy(), fmt='%.6f')
# np.savetxt(os.path.join('/data/lzj/easy/save_tensor/raw_feat/test_train_relation_cifar_euclid.csv'), class_relation2.numpy(), fmt='%.6f')

# train_vec = torch.load('/data/lzj/easy_mine/word_embedding/FC100/train_vocab_vec.pkl')
train_vec = torch.load('./word_embedding/cifarfs/train_vocab_vec.pkl')
# # test_vec = torch.load('/data/lzj/easy_mine/word_embedding/cifarfs/test_vocab_vec.pkl')
# # test_protos = torch.load('/data/lzj/easy_mine/save_tensor/raw_feat/feature_cifar_base64_word_embed_relation_together_res12/novel_save_protos.pkl')
class_relations = pairwise_cosine_similarity(train_vec.clone(), train_vec.clone())
# print(class_relations)
torch.save(class_relations, './word_embedding/cifarfs/train_vec_simi.pkl')
# np.savetxt(os.path.join('/data/lzj/easy_mine/save_tensor/raw_feat/test_train_relation_cifar_cos.csv'), class_relations.numpy(), fmt='%.6f')

# subset_path = os.path.join('/data/lzj/easy_mine', 'cifar_fs', 'meta-train')
# subset_path = os.path.join('/data/lzj/easy_mine', 'cifar_fs', 'meta-test')
# subset_path = os.path.join('/data/lzj/easy_mine', 'tieredimagenet', 'train')

# classe_files = os.listdir(subset_path)
# classe_files.sort(key=lambda x : int(x[1:]))
# for c, classe in enumerate(classe_files):
#     print(classe)

# train_vocab_vec = torch.load('/data/lzj/easy_mine/word_embedding/tiered/train_vocab_vec.pkl')
# test_vocab_vec = torch.load('/data/lzj/easy_mine/word_embedding/tiered/test_vocab_vec.pkl')

# train_protos = torch.load('/data/lzj/easy_mine/save_tensor/raw_feat/1120/feature_tier_base351_word_embed_relation_res12/save_protos.pkl')
# test_protos_old = torch.load('/data/lzj/easy_mine/save_tensor/raw_feat/1120/feature_tier_base351_word_embed_relation_res12/novel_save_protos.pkl')

# test_train_vocab_relation = pairwise_cosine_similarity(test_vocab_vec.clone(), train_vocab_vec.clone())
# test_train_feat_relation = pairwise_cosine_similarity(test_protos_old.clone(), train_protos.clone())

# np.savetxt(os.path.join('/data/lzj/easy_mine/save_tensor/raw_feat/test_train_relation_tier_cos.csv'), test_train_vocab_relation.numpy(), fmt='%.6f')
# np.savetxt(os.path.join('/data/lzj/easy_mine/save_tensor/raw_feat/test_train_relation_tier_feat_cos.csv'), test_train_feat_relation.numpy(), fmt='%.6f')

# subset_path = os.path.join('/data/lzj/easy_mine', 'FC100', 'train')
# classe_files = os.listdir(subset_path)
# classe_files.sort(key=lambda x : int(x[1:]))

# for c, classe in enumerate(classe_files):
#     print(classe)
name = os.path.dirname(os.path.abspath((__file__)))
print(name)

import datetime
cur_time = datetime.date.today()
print(str(cur_time))