import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim
from torch import nn
from models.QIS_GAN import QIS_GAN

from utils import *
from metrics import calc_psnr, calc_rmse, calc_ergas, calc_sam
from data_loader import build_datasets
from validate import validate

import pdb
import args_parser
from torch.nn import functional as F
import cv2
from time import *
import os
import scipy.io as io
from thop import profile
torch.cuda.is_available()

args = args_parser.args_parser()
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

print (args)

# torch.cuda.is_available()
def main():
    if args.dataset == 'PaviaU':
      args.n_bands = 103
    elif args.dataset == 'Botswana':
      args.n_bands = 145   
    elif args.dataset == 'Washington':
      args.n_bands = 191  
    elif args.dataset == 'Houston':
      args.n_bands = 144

    # Custom dataloader
    train_list, test_list = build_datasets(args.root, 
                                           args.dataset, 
                                           args.image_size, 
                                           args.n_select_bands, 
                                           args.scale_ratio)

    # Build the models
    if args.arch == 'QIS-GAN':
      model = QIS_GAN(scale_ratio=args.scale_ratio, 
                        n_select_bands=args.n_select_bands, n_bands=args.n_bands, 
                        image_size=args.image_size, feat_dim=64, guide_dim=64, 
                        H=args.image_size, W=args.image_size,mlp_dim=[256, 128], NIR_dim=33).cuda()

   
    model_path = args.model_path.replace('dataset', args.dataset) \
                                .replace('arch', args.arch) 
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path), strict=False)
        print ('Load the chekpoint of {}'.format(model_path))


    test_ref, test_lr, test_hr = test_list
    model.eval()

    # Set mini-batch dataset
    ref = test_ref.float().detach()
    lr = test_lr.float().detach()
    hr = test_hr.float().detach()
    
    begin_time = time()
    if args.arch == 'SSRNET':
        out, _, _, _, _, _ = model(lr.cuda(), hr.cuda())
    elif args.arch == 'SpatRNET':
        _, out, _, _, _, _ = model(lr.cuda(), hr.cuda())
    elif args.arch == 'SpecRNET':
        _, _, out, _, _, _ = model(lr.cuda(), hr.cuda())
    else:
        out, _, _, _, _, _ = model(lr.cuda(), hr.cuda())
    end_time = time()
    run_time = (end_time-begin_time)*1000

    print ()
    print ()
    print ('Dataset:   {}'.format(args.dataset))
    print ('Arch:   {}'.format(args.arch))
    print ('ModelSize(M):   {}'.format(np.around(os.path.getsize(model_path)//1024/1024.0, decimals=2)))
    print ('Time(Ms):   {}'.format(np.around(run_time, decimals=2)))
    flops, params = profile(model.cuda(), inputs=(lr.cuda(),hr.cuda()))
    flops = flops/1000000000
    print (flops)
    
    ref = ref.detach().cpu().numpy()
    out = out.detach().cpu().numpy()
        
    
    psnr = calc_psnr(ref, out)
    rmse = calc_rmse(ref, out)
    ergas = calc_ergas(ref, out)
    sam = calc_sam(ref, out)
    print ('RMSE:   {:.4f};'.format(rmse))
    print ('PSNR:   {:.4f};'.format(psnr))
    print ('ERGAS:   {:.4f};'.format(ergas))
    print ('SAM:   {:.4f}.'.format(sam))
    
    slr  =  F.interpolate(lr, scale_factor=args.scale_ratio, mode='bilinear')
    slr = slr.detach().cpu().numpy()
    slr  =  np.squeeze(slr).transpose(1,2,0).astype(np.float64)
    
    sref = np.squeeze(ref).transpose(1,2,0).astype(np.float64)
    sout = np.squeeze(out).transpose(1,2,0).astype(np.float64)
    
    io.savemat('./result/'+args.dataset+'/'+ str(args.scale_ratio)+'倍'+'/'+args.arch+'.mat',{'Out':sout})
    io.savemat('./result/'+args.dataset+'/'+ str(args.scale_ratio)+'倍'+'/'+'REF.mat',{'REF':sref})
    io.savemat('./result/'+args.dataset+'/'+ str(args.scale_ratio)+'倍'+'/'+'Upsample.mat',{'Out':slr})
    
    t_lr = np.squeeze(lr).detach().cpu().numpy().transpose(1,2,0).astype(np.float64)
    t_hr = np.squeeze(hr).detach().cpu().numpy().transpose(1,2,0).astype(np.float64)
    
    io.savemat('./test_data/'+args.dataset+'/'+ str(args.scale_ratio)+'倍'+'/'+'lr'+'.mat',{'HSI':t_lr})
    io.savemat('./test_data/'+args.dataset+'/'+ str(args.scale_ratio)+'倍'+'/'+'hr'+'.mat',{'MSI':t_hr})

   
if __name__ == '__main__':
    main()
