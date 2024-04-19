import torch
import numpy as np
import torch.nn.functional as F
from einops import rearrange
import os
###############################################
memory = torch.load('/data/lzj/easy/save_tensor/raw_feat/feature_base80_res12/proto_feat_sort_1.pkl')
# memory = rearrange(memory, 'N C M -> (N M) C')
print(memory)
#################################对删除均值的元素进行划分#############################
# offset = memory[0, :].unsqueeze(0)
# memory_offset = memory - offset

# sum_total = memory_offset.sum(dim=0, keepdim=True)
# print(memory_offset)
# stage_total = sum_total/3

# attribute_list = torch.zeros(5, memory.shape[1])
# attribute_list[0, :] = memory[0, :]
# attribute_list[-1, :] = memory[-1, :]

# for j in range(memory.shape[1]):
#     stage = stage_total[:, j]
#     sum = 0
#     start = 0
#     indx = 1
#     for i in range(memory.shape[0]):
#         sum += memory_offset[i,j]
#         if sum >= stage:
#             # print('start: ',start)
#             # print('end: ',i)
#             mid_ind = int((i-start)/2) + start
#             attribute_list[indx, j] = memory[mid_ind, j]
#             if indx<4:
#                 indx += 1
#             start = i
#             sum = memory_offset[i,j]

# print(attribute_list)
# torch.save(attribute_list, '/data/lzj/easy/save_tensor/raw_feat/proto_attri_stage5_M_1.pkl')

#################################对非删除均值的元素进行划分###########################
sum_total = memory
print(sum_total.shape)
sum_total = sum_total.sum(dim=0, keepdim=True)
print(sum_total.shape)
stage_total = sum_total/9

# 初始化属性表，将首位和末尾进行赋值
attribute_list = torch.zeros(10, memory.shape[1])
attribute_list[0, :] = memory[0, :]
attribute_list[-1, :] = memory[-1, :]
# cnt_list = memory[-1, :].unsqueeze(0)
print(attribute_list)
for j in range(memory.shape[1]):
    stage = stage_total[:, j]
    sum = 0
    start = 0
    indx = 1
    for i in range(memory.shape[0]):
        sum += memory[i,j]
        if sum >= stage:
            # print('start: ',start)
            # print('end: ',i)
            mid_ind = int((i-start)/2) + start
            attribute_list[indx, j] = memory[mid_ind, j]
            indx += 1
            start = i+1
            sum = 0
            
print(attribute_list)
# final_list = torch.cat((attribute_list, cnt_list), dim=0)
# print(final_list)
torch.save(attribute_list, '/data/lzj/easy/save_tensor/raw_feat/feature_base80_res12/proto_feat_sort_stage10_1.pkl')
##########################################################################################
# x = torch.rand(5,640)
# x_1 = x.reshape(5, -1, 640)
# print(x_1.shape)
# # print(x[:, ind[:,1]])