from utils import *
from args import *
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlockRN12(nn.Module):
    def __init__(self, in_planes, planes):
        super(BasicBlockRN12, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes)
        )

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope = 0.1)
        out = F.leaky_relu(self.bn2(self.conv2(out)), negative_slope = 0.1)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        if args.dropout > 0:
            out = F.dropout(out, p=args.dropout, training=self.training, inplace=True)
        return out
    
class ResNet12(nn.Module):
    def __init__(self, feature_maps, input_shape, num_classes, few_shot, rotations):
        super(ResNet12, self).__init__()        
        layers = []
        layers.append(BasicBlockRN12(input_shape[0], feature_maps))
        layers.append(BasicBlockRN12(feature_maps, int(2.5 * feature_maps)))
        layers.append(BasicBlockRN12(int(2.5 * feature_maps), 5 * feature_maps))
        layers.append(BasicBlockRN12(5 * feature_maps, 10 * feature_maps))        
        self.layers = nn.Sequential(*layers)
        self.linear = linear(10 * feature_maps, num_classes)
        self.rotations = rotations
        self.linear_rot = linear(10 * feature_maps, 4)
        self.linear_att = linear(10 * feature_maps, 300)
        # self.back_embedding = linear(300, 10 * feature_maps)
        self.mp = nn.MaxPool2d((2,2))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, index_mixup = None, lam = -1, minus_channel=False):
        if lam != -1:
            mixup_layer = random.randint(0, 3)
        else:
            mixup_layer = -1
        out = x
        if mixup_layer == 0:
            out = lam * out + (1 - lam) * out[index_mixup]
        for i in range(len(self.layers)):
            out = self.layers[i](out)
            if mixup_layer == i + 1:
                out = lam * out + (1 - lam) * out[index_mixup]
            out = self.mp(F.leaky_relu(out, negative_slope = 0.1))
            
            if i == 2:
                mid_feat = out
#######            
        # print(out.shape)
        feature_before_avg = out
#######        
        out = F.avg_pool2d(out, out.shape[2])
        features = out.view(out.size(0), -1)
        # print(features.shape)
        #######
        # if minus_channel:
        #     # print('here')
        #     topk, _ = torch.topk(features, k=300, dim=-1)
        #     topk = topk[:, -1].unsqueeze(-1)
        #     features = features * (features >= topk)
        #######
        
        out = self.linear(features)
        out_attri = self.linear_att(features)
        # back_feature = F.leaky_relu(self.back_embedding(out_attri))
        if self.rotations:
            out_rot = self.linear_rot(features)
            # return (out, out_rot), features
            return (out, out_rot), out_attri, features
            # return (out, out_rot), out_attri, back_feature, features
            # return (out, out_rot), feature_before_avg
        # return out, features
        return out, out_attri, features
        # return out, out_attri, back_feature, features
        # return out, feature_before_avg
