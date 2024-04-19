import torch
import resnet12, resnet12_generate_protos
import numpy as np
import my_few_shot_eval
import resnet
import wideresnet
import s2m2
import mlp
import datasets
import torch.nn.functional as F
from torchmetrics.functional import pairwise_cosine_similarity, pairwise_euclidean_distance
import os
import pandas as pd
from args import args
if args.ema > 0:
    from torch_ema import ExponentialMovingAverage

def create_model():
    if args.model.lower() == "resnet18":
        return resnet.ResNet18(args.feature_maps, input_shape, num_classes, few_shot, args.rotations).to(args.device)
    if args.model.lower() == "resnet20":
        return resnet.ResNet20(args.feature_maps, input_shape, num_classes, few_shot, args.rotations).to(args.device)
    if args.model.lower() == "wideresnet":
        return wideresnet.WideResNet(args.feature_maps, input_shape, few_shot, args.rotations, num_classes = num_classes).to(args.device)
    if args.model.lower() == "resnet12":
        print('now')
        return resnet12_generate_protos.ResNet12(args.feature_maps, input_shape, num_classes, few_shot, args.rotations).to(args.device)
    if args.model.lower()[:3] == "mlp":
        return mlp.MLP(args.feature_maps, int(args.model[3:]), input_shape, num_classes, args.rotations, few_shot).to(args.device)
    if args.model.lower() == "s2m2r":
        return s2m2.S2M2R(args.feature_maps, input_shape, args.rotations, num_classes = num_classes).to(args.device)


loaders, input_shape, num_classes, few_shot, top_5 = datasets.get_dataset(args.dataset)
if few_shot:
    num_classes, val_classes, novel_classes, elements_per_class = num_classes
    train_loader, train_clean_loader, val_loader, novel_loader = loaders

model = create_model()
if args.ema > 0:
    ema = ExponentialMovingAverage(model.parameters(), decay=args.ema)

if args.load_model != "":
    model.load_state_dict(torch.load(args.load_model, map_location=torch.device(args.device)))
    model.to(args.device)

model.eval()
test_word_vec = torch.load('/data/lzj/easy_mine/word_embedding/miniimagenet/test_vocab_vec.pkl').to(args.device)
test_word_embedding = model.linear_att(test_word_vec)
saved_proto = torch.load('/data/lzj/easy_mine/save_tensor/raw_feat/feature_cifar_base64_word_embed_relation_together_res12/novel_save_protos.pkl').to(args.device)
simi = pairwise_euclidean_distance(test_word_embedding.clone(), test_word_embedding.clone())
print(simi)
torch.save(test_word_embedding, '/data/lzj/easy_mine/word_embedding/cifarfs/generate_novel_protos_1shot_together.pkl')