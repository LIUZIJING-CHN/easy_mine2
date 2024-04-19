from utils import *
from args import *
import torch
import torch.nn as nn
import torch.nn.functional as F
    
class Attribute_generator(nn.Module):
    def __init__(self, num_classes):
        super(Attribute_generator, self).__init__() 
        self.layer1 = nn.Conv2d(in_channels=640, out_channels=640, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm2d(640)
        # self.decoder = nn.Conv2d(in_channels=1280, out_channels=640, kernel_size=1, bias=True)
        self.Linear = nn.Linear(in_features=640, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.layer1(x)))
        # x = self.decoder(x)
        x_topk, ind = torch.topk(x, k=300, dim=1)
        filt = x_topk[:, -1, ...].unsqueeze(1)
        x = x*(x>=filt)
        out = F.avg_pool2d(x, x.shape[2])
        out = out.view(out.size(0), -1)
        out = self.Linear(out)
        return x, out
