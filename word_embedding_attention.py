from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

class Word_Embedding(nn.Module):
    def __init__(self) -> None:
        super(Word_Embedding, self).__init__()
        self.back_embedding1 = nn.Linear(312, 1280)
        self.back_embedding2 = nn.Linear(1280, 1280)
        self.back_embedding3 = nn.Linear(1280, 640)
    
    def forward(self, x):
        x = F.leaky_relu(self.back_embedding1(x))
        x = F.leaky_relu(self.back_embedding2(x))
        x = F.leaky_relu(self.back_embedding3(x))
        
        return x