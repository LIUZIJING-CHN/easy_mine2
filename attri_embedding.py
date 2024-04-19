from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

class Att_Embedding(nn.Module):
    def __init__(self) -> None:
        super(Att_Embedding, self).__init__()
        self.linear_att = nn.Linear(640, 300)
    
    def forward(self, x):
        x = self.linear_att(x)
        
        return x