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

class QIS_GAN_x16(nn.Module):

    def __init__(self, scale_ratio, n_select_bands,  n_bands, image_size, feat_dim=128,
                 guide_dim=128, H=64, W=64, spa_edsr_num=4, spe_edsr_num=4, mlp_dim=[256, 128], NIR_dim=33):
        super().__init__()
        self.feat_dim = feat_dim
        self.guide_dim = guide_dim
        self.mlp_dim = mlp_dim
        self.NIR_dim = NIR_dim
        self.image_size = image_size
        self.scale_ratio = scale_ratio
        self.n_select_bands = n_select_bands
        self.n_bands =  n_bands
        self.encoder16 = nn.AdaptiveMaxPool2d((H//8, W//8))
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
      
        imnet_in_dim_16 = self.feat_dim + self.guide_dim + 2
        imnet_in_dim_32 = NIR_dim-1 + self.guide_dim + 2
        imnet_in_dim_64 = NIR_dim-1 + self.guide_dim + 2
        imnet_in_dim_128 = NIR_dim - 1 + self.guide_dim + 2


        self.imnet_16 = MLP(imnet_in_dim_16, out_dim=NIR_dim, hidden_list=self.mlp_dim)
        self.imnet_32 = MLP(imnet_in_dim_32, out_dim=NIR_dim, hidden_list=self.mlp_dim)
        self.imnet_64 = MLP(imnet_in_dim_64, out_dim=NIR_dim, hidden_list=self.mlp_dim)
        self.imnet_128 = MLP(imnet_in_dim_128, out_dim=self.n_bands+1, hidden_list=self.mlp_dim)
       
    def query_16(self, feat, coord, hr_guide):

        # feat: [B, C, h, w]
        # coord: [B, N, 2], N <= H * W

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

                pred = self.imnet_16(inp.view(B * N, -1)).view(B, N, -1)  # [B, N, 2]
                preds.append(pred)

        preds = torch.stack(preds, dim=-1)  # [B, N, 2, kk]
        weight = F.softmax(preds[:, :, -1, :], dim=-1)
        ret = (preds[:, :, 0:-1, :] * weight.unsqueeze(-2)).sum(-1, keepdim=True).squeeze(-1)
        ret = ret.permute(0, 2, 1).view(b, -1, H, W)

        return ret
    def query_32(self, feat, coord, hr_guide):

        # feat: [B, C, h, w]
        # coord: [B, N, 2], N <= H * W

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

                pred = self.imnet_32(inp.view(B * N, -1)).view(B, N, -1)  # [B, N, 2]
                preds.append(pred)

        preds = torch.stack(preds, dim=-1)  # [B, N, 2, kk]
        weight = F.softmax(preds[:, :, -1, :], dim=-1)
        ret = (preds[:, :, 0:-1, :] * weight.unsqueeze(-2)).sum(-1, keepdim=True).squeeze(-1)
        ret = ret.permute(0, 2, 1).view(b, -1, H, W)

        return ret

    def query_64(self, feat, coord, hr_guide):

        # feat: [B, C, h, w]
        # coord: [B, N, 2], N <= H * W

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

                pred = self.imnet_64(inp.view(B * N, -1)).view(B, N, -1)  # [B, N, 2]
                preds.append(pred)

        preds = torch.stack(preds, dim=-1)  # [B, N, 2, kk]
        weight = F.softmax(preds[:, :, -1, :], dim=-1)
        ret = (preds[:, :, 0:-1, :] * weight.unsqueeze(-2)).sum(-1, keepdim=True).squeeze(-1)
        ret = ret.permute(0, 2, 1).view(b, -1, H, W)

        return ret

    def query_128(self, feat, coord, hr_guide):

        # feat: [B, C, h, w]
        # coord: [B, N, 2], N <= H * W

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
        # HR_MSI Bx3x64x64
        # lms Bx31x64x64
        # LR_HSI Bx31x16x16
        lms = F.interpolate(LR_HSI, scale_factor=self.scale_ratio, mode='bilinear')
        _, _, H, W = HR_MSI.shape
        coord_16 = make_coord([H//8, W//8]).cuda()
        coord_32 = make_coord([H//4, W//4]).cuda()
        coord_64 = make_coord([H // 2, W // 2]).cuda()
        coord_128 = make_coord([H, W]).cuda()

        feat = torch.cat([HR_MSI, lms], dim=1)
      
        hr_spa = self.spatial_encoder(feat)  # Bx128xHxW                 spatial_encoder空间编码将107映射至特征维度 不改变空间形状
      
        # print(hr_spa.shape)
        guide16 = self.encoder16(hr_spa)
        
        guide32 = self.encoder32(hr_spa)  #
       
        guide64 = self.encoder64(hr_spa)  #
       
        lr_spe = self.spectral_encoder(LR_HSI)  # Bx128xhxw The feature map of LR-HSI   # spectral_encoder 映射至设定的特征维度   不改变空间形状
       
        NIR_feature_16 = self.query_16(lr_spe, coord_16, guide16)
       
      
        NIR_feature_32 = self.query_32(NIR_feature_16, coord_32, guide32)  # BxCxHxW
      
      
        NIR_feature_64 = self.query_64(NIR_feature_32, coord_64, guide64)  # BxCxHxW
      
        
        NIR_feature_128 = self.query_128(NIR_feature_64, coord_128, hr_spa)  # BxCxHxW
       
       
        output = lms + NIR_feature_128
       
        return output,0,0,0,0,0


if __name__ == '__main__':
       


    model =QIS_GAN_x16( scale_ratio=16, n_select_bands=4,  n_bands=145, image_size=128, feat_dim=64,
                 guide_dim=64, H=128, W=128, spa_edsr_num=4, spe_edsr_num=4, mlp_dim=[256, 128], NIR_dim=33).cuda()
    hr = torch.rand(1,4,128,128).cuda()
    lr = torch.rand(1,145,8,8).cuda()
    T=model(lr,hr)
    print(T[0].shape)
    # print(flop_count_table(FlopCountAnalysis(model,(lr,hr))))
    ### 0.807M                 | 2.825G  ###
