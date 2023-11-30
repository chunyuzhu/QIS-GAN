# -*- coding: utf-8 -*-
"""
Created on Wed May 10 08:09:04 2023

@author: zxc
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May  9 19:17:13 2023

@author: zxc
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 12:31:02 2023

@author: zxc
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 21:02:28 2023

@author: zxc
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x
    
def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

class QIS_GAN(nn.Module):

    def __init__(self, scale_ratio, n_select_bands,  n_bands, image_size, feat_dim=128,
                 guide_dim=128, H=64, W=64,  mlp_dim=[256, 128], NIR_dim=33):
        super().__init__()
        self.feat_dim = feat_dim
        self.guide_dim = guide_dim
        self.mlp_dim = mlp_dim
        self.NIR_dim = NIR_dim
        self.image_size = image_size
        self.scale_ratio = scale_ratio
        self.n_select_bands = n_select_bands
        self.n_bands =  n_bands
        self.encoder32 = nn.AdaptiveMaxPool2d((H//4, W//4))
        self.encoder64 = nn.AdaptiveMaxPool2d((H//2, W//2))       
        self.spatial_encoder = nn.Sequential(
                  nn.Conv2d(self.n_select_bands+self.n_bands, 64, kernel_size=3, stride=1, padding=1),
                  nn.ReLU(),
                )
        self.spectral_encoder = nn.Sequential(
                  nn.Conv2d(self.n_bands,64, kernel_size=3, stride=1, padding=1),
                  nn.ReLU(),
                )        
        imnet_in_dim_32 = self.feat_dim + self.guide_dim + 2
        imnet_in_dim_64 = NIR_dim-1 + self.guide_dim + 2
        imnet_in_dim_128 = NIR_dim - 1 + self.guide_dim + 2

        self.imnet_32 = MLP(imnet_in_dim_32, out_dim=NIR_dim, hidden_list=self.mlp_dim)
        self.imnet_64 = MLP(imnet_in_dim_64, out_dim=NIR_dim, hidden_list=self.mlp_dim)
        self.imnet_128 = MLP(imnet_in_dim_128, out_dim=self.n_bands+1, hidden_list=self.mlp_dim)
        

    def query_32(self, feat, coord, hr_guide):

      
        b, c, h, w = feat.shape  # lr  7x128x8x8
        _, _, H, W = hr_guide.shape  # hr  7x128x64x64
        coord = coord.expand(b, H * W, 2)
        B, N, _ = coord.shape

        feat_coord = make_coord((h, w), flatten=False).to(feat.device).permute(2, 0, 1).unsqueeze(0).expand(b, 2, h, w).cuda()

        q_guide_hr = F.grid_sample(hr_guide, coord.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,
                     :].permute(0, 2, 1)  # [B, N, C]

        rx = 1 / h
        ry = 1 / w

        preds = []

        for vx in [-1, 1]:
            for vy in [-1, 1]:
                coord_ = coord.clone()

                coord_[:, :, 0] += (vx) * rx
                coord_[:, :, 1] += (vy) * ry

                # feat: [B, c, h, w], coord_: [B, N, 2] --> [B, 1, N, 2], out: [B, c, 1, N] --> [B, c, N] --> [B, N, c]
                q_feat = F.grid_sample(feat, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,
                         :].permute(0, 2, 1)  # [B, N, c]
                q_coord = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[
                          :, :, 0, :].permute(0, 2, 1)  # [B, N, 2]

                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= h
                rel_coord[:, :, 1] *= w

                inp = torch.cat([q_feat, q_guide_hr, rel_coord], dim=-1)
                # print('aaa:',inp.view(B * N, -1).shape)
                pred = self.imnet_32(inp.view(B * N, -1)).view(B, N, -1)  # [B, N, 2]
                preds.append(pred)

        preds = torch.stack(preds, dim=-1)  # [B, N, 2, kk]
        weight = F.softmax(preds[:, :, -1, :], dim=-1)
        ret = (preds[:, :, 0:-1, :] * weight.unsqueeze(-2)).sum(-1, keepdim=True).squeeze(-1)
        ret = ret.permute(0, 2, 1).view(b, -1, H, W)

        return ret

    def query_64(self, feat, coord, hr_guide):

       

        b, c, h, w = feat.shape  # lr  7x128x8x8
        _, _, H, W = hr_guide.shape  # hr  7x128x64x64
        coord = coord.expand(b, H * W, 2)
        B, N, _ = coord.shape
     
        feat_coord = make_coord((h, w), flatten=False).to(feat.device).permute(2, 0, 1).unsqueeze(0).expand(b, 2, h, w).cuda()

        q_guide_hr = F.grid_sample(hr_guide, coord.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,
                     :].permute(0, 2, 1)  # [B, N, C]

        rx = 1 / h
        ry = 1 / w

        preds = []

        for vx in [-1, 1]:
            for vy in [-1, 1]:
                coord_ = coord.clone()

                coord_[:, :, 0] += (vx) * rx
                coord_[:, :, 1] += (vy) * ry

                # feat: [B, c, h, w], coord_: [B, N, 2] --> [B, 1, N, 2], out: [B, c, 1, N] --> [B, c, N] --> [B, N, c]
                q_feat = F.grid_sample(feat, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,
                         :].permute(0, 2, 1)  # [B, N, c]
                q_coord = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[
                          :, :, 0, :].permute(0, 2, 1)  # [B, N, 2]

                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= h
                rel_coord[:, :, 1] *= w

                inp = torch.cat([q_feat, q_guide_hr, rel_coord], dim=-1)

                pred = self.imnet_64(inp.view(B * N, -1)).view(B, N, -1)  # [B, N, 2]
                preds.append(pred)

        preds = torch.stack(preds, dim=-1)  # [B, N, 2, kk]
        weight = F.softmax(preds[:, :, -1, :], dim=-1)
        ret = (preds[:, :, 0:-1, :] * weight.unsqueeze(-2)).sum(-1, keepdim=True).squeeze(-1)
        ret = ret.permute(0, 2, 1).view(b, -1, H, W)

        return ret

    def query_128(self, feat, coord, hr_guide):

        b, c, h, w = feat.shape  # lr  7x128x8x8
        _, _, H, W = hr_guide.shape  # hr  7x128x64x64
        coord = coord.expand(b, H * W, 2)
        B, N, _ = coord.shape

        # LR centers' coords
        feat_coord = make_coord((h, w), flatten=False).to(feat.device).permute(2, 0, 1).unsqueeze(0).expand(b, 2, h, w).cuda()

        q_guide_hr = F.grid_sample(hr_guide, coord.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,
                     :].permute(0, 2, 1)  # [B, N, C]

        rx = 1 / h
        ry = 1 / w

        preds = []

        for vx in [-1, 1]:
            for vy in [-1, 1]:
                coord_ = coord.clone()

                coord_[:, :, 0] += (vx) * rx
                coord_[:, :, 1] += (vy) * ry

                # feat: [B, c, h, w], coord_: [B, N, 2] --> [B, 1, N, 2], out: [B, c, 1, N] --> [B, c, N] --> [B, N, c]
                q_feat = F.grid_sample(feat, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0,
                         :].permute(0, 2, 1)  # [B, N, c]
                q_coord = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[
                          :, :, 0, :].permute(0, 2, 1)  # [B, N, 2]

                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= h
                rel_coord[:, :, 1] *= w

                inp = torch.cat([q_feat, q_guide_hr, rel_coord], dim=-1)

                pred = self.imnet_128(inp.view(B * N, -1)).view(B, N, -1)  # [B, N, 2]
                preds.append(pred)

        preds = torch.stack(preds, dim=-1)  # [B, N, 2, kk]
        weight = F.softmax(preds[:, :, -1, :], dim=-1)
        ret = (preds[:, :, 0:-1, :] * weight.unsqueeze(-2)).sum(-1, keepdim=True).squeeze(-1)
        ret = ret.permute(0, 2, 1).view(b, -1, H, W)

        return ret

    def forward(self, LR_HSI,HR_MSI ):
        
        lms = F.interpolate(LR_HSI, scale_factor=self.scale_ratio, mode='bilinear')
        _, _, H, W = HR_MSI.shape
        coord_32 = make_coord([H//4, W//4]).cuda()
        coord_64 = make_coord([H // 2, W // 2]).cuda()
        coord_128 = make_coord([H, W]).cuda()

        feat = torch.cat([HR_MSI, lms], dim=1)
      
        hr_spa = self.spatial_encoder(feat)  # Bx128xHxW                 spatial_encoder空间编码将107映射至特征维度 不改变空间形状
      
      
        
        guide32 = self.encoder32(hr_spa)  #
       
        guide64 = self.encoder64(hr_spa)  #
       
        lr_spe = self.spectral_encoder(LR_HSI)  # Bx128xhxw The feature map of LR-HSI   # spectral_encoder 映射至设定的特征维度   不改变空间形状
       
     
        NIR_feature_32 = self.query_32(lr_spe, coord_32, guide32)  # BxCxHxW
      
      
        NIR_feature_64 = self.query_64(NIR_feature_32, coord_64, guide64)  # BxCxHxW
      
      
        NIR_feature_128 = self.query_128(NIR_feature_64, coord_128, hr_spa)  # BxCxHxW
      
       
        output = lms + NIR_feature_128
       
        return output,0,0,0,0,0


if __name__ == '__main__':
    
    
    H = 256
    W = 256
   

    model =QIS_GAN( scale_ratio=8, n_select_bands=4,  n_bands=145, image_size=H, feat_dim=64,
                 guide_dim=64, H=H, W=W, mlp_dim=[256, 128], NIR_dim=33).cuda()
    hr = torch.ones(1,4,H,W).cuda()
    lr = torch.ones(1,145,H//8,W//8).cuda()
    T=model(lr,hr)
    print(T[0].shape)
    
    flops, params = profile(model, inputs=(lr,hr))
    flops = flops/1000000000
    
    params = params / 1000000.0
    print(f"模型的FLOPs: {flops}")
    print(f"模型的params: {params}")
    ### 0.807M                 | 2.825G  ###

