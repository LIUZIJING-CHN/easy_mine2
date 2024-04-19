from utils import *
from args import *
import torch
import torch.nn as nn
import torch.nn.functional as F
    
class Attribute_generator(nn.Module):
    def __init__(self, num_classes):
        super(Attribute_generator, self).__init__() 
        self.layer1 = nn.Conv2d(in_channels=640, out_channels=640, kernel_size=1, bias=True)
        # self.decoder = nn.Conv2d(in_channels=1280, out_channels=640, kernel_size=1, bias=True)
        self.Linear = nn.Linear(in_features=640, out_features=num_classes)

    def forward(self, x):
        x = self.layer1(x)
        # x = self.decoder(x)
        out = F.avg_pool2d(x, x.shape[2])
        out = out.view(out.size(0), -1)
        out = self.Linear(out)
        return x, out
