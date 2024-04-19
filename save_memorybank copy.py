import torch
import resnet12, resnet12_ori
import numpy as np
import my_few_shot_eval
import resnet
import wideresnet
import resnet12
import attribute_generator
import s2m2
import mlp
import datasets
import torch.nn.functional as F
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
        return resnet12.ResNet12(args.feature_maps, input_shape, num_classes, few_shot, args.rotations).to(args.device)
    if args.model.lower()[:3] == "mlp":
        return mlp.MLP(args.feature_maps, int(args.model[3:]), input_shape, num_classes, args.rotations, few_shot).to(args.device)
    if args.model.lower() == "s2m2r":
        return s2m2.S2M2R(args.feature_maps, input_shape, args.rotations, num_classes = num_classes).to(args.device)
    
def create_attribure():
    return attribute_generator.Attribute_generator(num_classes=num_classes).to(args.device)


loaders, input_shape, num_classes, few_shot, top_5 = datasets.get_dataset(args.dataset)
if few_shot:
    num_classes, val_classes, novel_classes, elements_per_class = num_classes
    train_loader, train_clean_loader, val_loader, novel_loader = loaders

model = create_model()
attribute_model = create_attribure()
if args.ema > 0:
    ema = ExponentialMovingAverage(model.parameters(), decay=args.ema)

if args.load_model != "":
    model.load_state_dict(torch.load(args.load_model, map_location=torch.device(args.device)))
    model.to(args.device)

if args.load_attri_model != "":
    attribute_model.load_state_dict(torch.load(args.load_attri_model, map_location=torch.device(args.device)))
    attribute_model.to(args.device)

model.eval()
attribute_model.eval()
save_feature = []
for batch_idx, (data, target) in enumerate(train_clean_loader):
    print(batch_idx)
    # print(target)
    data = data.to(args.device)
    # output, features = model(data)
    _, _, features = model(data)
    # _, _, _, features = model(data)
    # _, features = model(data)
    # features, _ = attribute_model(features)
    
    # features = F.avg_pool2d(features, features.shape[2])
    # B, C, _, _ = features.shape
    # features = features.view(B, -1)
    
    features_cpu = features.detach().cpu()
    save_feature.append(features_cpu)
    
save_feature = torch.cat(save_feature)
save_protos = []
cls_num = int(save_feature.shape[0]/600)
for i in range(cls_num):
    cls_features = save_feature[i*600 : (i+1)*600]
    protos = cls_features.mean(dim=0, keepdim=True)
    save_protos.append(protos)

save_protos = torch.cat(save_protos, dim=0)
    
# for i in range(save_protos.shape[1]):
#     channel_sort, _ = save_protos[:, i].sort()
#     save_protos[:, i] = channel_sort
    
# print(save_protos)
# tag = save_protos > 1.00
# tag = tag.sum(dim=0, keepdim=True)
# save_protos = torch.cat((save_protos, tag), dim=0)

# np.savetxt(os.path.join('./save_tensor/raw_feat/resnet12_word_embed_test_MSEME_novel.pt1.csv'), save_protos.numpy(), fmt='%.6f')
torch.save(save_protos, './save_tensor/raw_feat/feature_tiered_base351_word_embed_res12/save_protos.pkl')

# for i in range(save_protos.shape[1]):
#     print(i)
#     item = save_protos[:, i].unsqueeze(1)
#     # print(item.shape)
#     channel_atte = torch.mm(item, item.T)
    # print(channel_atte)
    
    # # channel_atte = torch.norm
    # max = channel_atte.max(dim=1)[0].unsqueeze(0)
    # print(max)
    # min = channel_atte.min(dim=1)[0].unsqueeze(0)
    # channel_atte = (channel_atte-min)/(max-min)
    
    # print(channel_atte)
    
    # pagename = 'class'+str(i)
    # data = pd.DataFrame(channel_atte)
    # writer = pd.ExcelWriter('./save_tensor/channel_atten/A.xlsx')
    # data.to_excel(writer, pagename, float_format='%.6f')
    # writer.save()
    
    # writer.close()
    
    # np.savetxt(os.path.join('./save_tensor/channel_atten', str(i)+'.csv'), channel_atte.numpy(), fmt='%.6f')
