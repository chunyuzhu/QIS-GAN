# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 16:38:17 2023

@author: zxc
"""

import torch
from torch import nn
import torch.nn.functional as F

def set_grad(network,requires_grad):
        for param in network.parameters():
            param.requires_grad = requires_grad


def gradient(img):
    dx = img[:, :, :, 1:] - img[:, :, :, :-1]
    dy = img[:, :, 1:, :] - img[:, :, :-1, :]
    return dx, dy
def gradient_loss(pred, target):
    pred_dx, pred_dy = gradient(pred)
    target_dx, target_dy = gradient(target)
    dx_loss = F.mse_loss(pred_dx, target_dx)
    dy_loss = F.mse_loss(pred_dy, target_dy)
    return dx_loss + dy_loss


def spectral_gradient(fused, ref):
    sg_fused = fused[:, 0: fused.size(1)-1, :, :] - fused[:, 1:fused.size(1), :, :]
    sg_ref =   ref[:, 0:ref.size(1)-1, :, :] - ref[:, 1:ref.size(1), :, :]
    
    return sg_fused, sg_ref

class Con_Edge_Spec_loss(nn.Module):
    def __init__(self):
        super(Con_Edge_Spec_loss, self).__init__()
        self.L1_loss = nn.L1Loss()
        self.L2_loss = nn.MSELoss()
    def forward(self, fused, ref, alpha=1, beta=1):
        
        Con_loss = self.L1_loss(fused, ref)       
              
        Edge_loss  = gradient_loss(fused, ref)
        
        spec_fused, spec_ref = spectral_gradient(fused, ref)
        Spec_loss = self.L2_loss(spec_fused, spec_ref)
        
        loss = Con_loss + Edge_loss + Spec_loss
               
        return loss 
    
    
class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()       
        self.Con_Edge_Spec_loss = Con_Edge_Spec_loss()
        
    def forward(self, img_out_labels, out_images, target_images):
        # Adversarial Loss
        adversarial_loss = torch.mean(1 - img_out_labels)
       
        image_loss = self.Con_Edge_Spec_loss(out_images, target_images)
            
       
        return image_loss + adversarial_loss 


if __name__ == "__main__":
    g_loss = Con_Edge_Spec_loss()
    A = torch.rand(1,144,128,128)
    B = torch.rand(1,144,128,128)
    print(g_loss(A,B))
