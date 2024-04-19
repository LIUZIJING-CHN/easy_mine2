import torch
import torch.nn as nn
import torch.nn.functional as F
from args import args
from utils import preprocess


class ProtoNet(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        # bias & scale of cosine classifier
        self.bias = nn.Parameter(torch.FloatTensor(1).fill_(0), requires_grad=True).cuda()
        self.scale_cls = nn.Parameter(torch.FloatTensor(1).fill_(10), requires_grad=True).cuda()

        # backbone
        self.backbone = backbone
        # self.rotations = rotations
        # self.linear_rot = torch.nn.Linear(2048, 4)

    def cos_classifier(self, w, f):
        """
        w.shape = B, nC, d
        f.shape = B, M, d
        """
        f = F.normalize(f, p=2, dim=f.dim()-1, eps=1e-12)
        w = F.normalize(w, p=2, dim=w.dim()-1, eps=1e-12)

        cls_scores = f @ w.transpose(1, 2) # B, M, nC
        cls_scores = self.scale_cls * (cls_scores + self.bias)
        return cls_scores

    def forward(self, supp_x, supp_y, x):
        """
        supp_x.shape = [B, nSupp, C, H, W]
        supp_y.shape = [B, nSupp]
        x.shape = [B, nQry, C, H, W]
        """
        num_classes = supp_y.max() + 1 # NOTE: assume B==1

        B, nSupp, C, H, W = supp_x.shape
        # print(supp_x.shape)
        supp_f = self.backbone.forward(supp_x.view(-1, C, H, W))
        
        ####保留topk500channel：
        # support_topk, _ = torch.topk(supp_f.double(), k=500, dim=1)
        # support_topk = support_topk[:, -1].unsqueeze(1)
        # supp_f = supp_f * (supp_f > support_topk)
        ####
        supp_f = supp_f.view(B, nSupp, -1)
        # 训练时加预处理
        # supp_f_copy = supp_f
        # supp_f = preprocess(supp_f_copy, supp_f)

        supp_y_1hot = F.one_hot(supp_y, num_classes).transpose(1, 2) # B, nC, nSupp

        # B, nC, nSupp x B, nSupp, d = B, nC, d
        prototypes = torch.bmm(supp_y_1hot.float(), supp_f)
        prototypes = prototypes / supp_y_1hot.sum(dim=2, keepdim=True) # NOTE: may div 0 if some classes got 0 images

        feat = self.backbone.forward(x.view(-1, C, H, W))
        # 训练时加预处理
        # feat = feat.reshape(-1, args.n_ways, feat.shape[1])
        # feat = preprocess(supp_f_copy, feat)
        # feat = feat.view(-1, feat.shape[-1])
        ####保留topk channel：
        if args.feat_topk:
            query_topk, _ = torch.topk(feat.double(), k=args.topk, dim=1)
            query_topk = query_topk[:, -1].unsqueeze(1)
            feat = feat * (feat > query_topk)
        ####
        feat = feat.view(B, x.shape[1], -1) # B, nQry, d
        # print(feat.shape)
        # print(prototypes.shape)
        
        logits = self.cos_classifier(prototypes, feat) # B, nQry, nC
        # ## for rotations
        # if self.rotations:
        #     all_feat = torch.cat([supp_f, feat], dim=0)
        #     out_rot = self.linear_rot(all_feat)
        #     return (logits, out_rot)
                
        return logits