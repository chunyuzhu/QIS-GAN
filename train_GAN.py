import torch
from torch import nn
from utils import to_var, batch_ids2words
import random
import torch.nn.functional as F
import cv2


def spatial_edge(x):
    edge1 = x[:, :, 0:x.size(2)-1, :] - x[:, :, 1:x.size(2), :]
    edge2 = x[:, :, :, 0:x.size(3)-1] - x[:, :,  :, 1:x.size(3)]

    return edge1, edge2

def spectral_edge(x):
    edge = x[:, 0:x.size(1)-1, :, :] - x[:, 1:x.size(1), :, :]

    return edge

image_size=128
hh=[]
ww=[]
def train_GAN(train_list, 
          image_size, 
          scale_ratio, 
          n_bands, 
          arch, 
          model_G, 
          model_D, 
          optimizer_G, 
          optimizer_D,
          criterion, 
          epoch, 
          n_epochs,
          h_str ,
          w_str ):
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    train_ref, train_lr, train_hr = train_list
    
    # h, w = train_ref.size(2), train_ref.size(3)
    # for i in range(10001):
        
    #     h_str = random.randint(0, h-image_size-1)
    #     w_str = random.randint(0, w-image_size-1)
    #     hh.append(h_str)
    #     ww.append(w_str)




    train_lr = train_ref[:, :, h_str:h_str+image_size, w_str:w_str+image_size]
    train_ref = train_ref[:, :, h_str:h_str+image_size, w_str:w_str+image_size]
    train_lr = F.interpolate(train_ref, scale_factor=1/(scale_ratio*1.0))
    train_hr = train_hr[:, :, h_str:h_str+image_size, w_str:w_str+image_size]

    # model.train()

    # Set mini-batch dataset
    image_lr = to_var(train_lr).detach()
    image_hr = to_var(train_hr).detach()
    image_ref = to_var(train_ref).detach()

    # Forward, Backward and Optimize
    model_G.train()
    model_D.train()
    # model_D_edge.train()
    # model_D_spec.train()
    
    fake, out_spat, out_spec, edge_spat1, edge_spat2, edge_spec = model_G(image_lr,image_hr)#fake 是有grad_fn的
    ############################
    # (1) Update D network: maximize D(x)-1-D(G(z))
    ###########################
    
    
    #训练图像鉴别器
    model_D.zero_grad()
    real_out = model_D(image_ref)
    # fake_out = model_D(fake)
    fake_clone = fake.clone().detach().requires_grad_(True)
    fake_out = model_D(fake_clone)
    d_loss = 1 - real_out + fake_out
    d_loss.backward()
    optimizer_D.step()
    
   
    ############################
    # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
    ###########################
    # old_params = model_G.state_dict()
    model_G.zero_grad()
    # fake_clone = fake.clone().detach().requires_grad_(True)
    # fake_out = model_D(fake_clone)
    img_out_labels = model_D(fake)
    # edge_out_labels = model_D_edge(fake)
    # spec_out_labels = model_D_spec(fake)
  
    
    # g_loss,maeloss,samloss,bilateralloss,advLoss = criterion(fake, image_ref, fake_out)
    g_loss = criterion(img_out_labels, fake, image_ref)
    
    g_loss.backward()
    #torch.nn.utils.clip_grad_norm_(model_G.parameters(), max_norm=0.01)
    optimizer_G.step()
    
    
    