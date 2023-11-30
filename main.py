import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim
from torch import nn
# from models.SSRNET import SSRNET
from models.Discriminator import Discriminator
from models.QIS_GAN import QIS_GAN
from models.QIS_GAN_x4 import QIS_GAN_x4
from models.QIS_GAN_x16 import QIS_GAN_x16
from utils import *
from data_loader import build_datasets
from validate import validate
from train_GAN import train_GAN
import pdb
import args_parser

from loss import GeneratorLoss
from torch.nn import functional as F
import pandas as pd

args = args_parser.args_parser()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print (args)

df = pd.read_excel(args.dataset+'_rand.xlsm', sheet_name='Sheet1')
h_rand = df.iloc[:,0].tolist()
w_rand = df.iloc[:,1].tolist()


def main():
    # Custom dataloader
    train_list, test_list = build_datasets(args.root, 
                                           args.dataset, 
                                           args.image_size, 
                                           args.n_select_bands, 
                                           args.scale_ratio)
    if args.dataset == 'PaviaU':
      args.n_bands = 103
    elif args.dataset == 'Botswana':
      args.n_bands = 145   
    elif args.dataset == 'Washington':
      args.n_bands = 191  
    elif args.dataset == 'Houston':
      args.n_bands = 144
      
    # Build the models
    if args.arch == 'QIS-GAN':
      model_G = QIS_GAN(scale_ratio=args.scale_ratio, 
                        n_select_bands=args.n_select_bands, n_bands=args.n_bands, 
                        image_size=args.image_size, feat_dim=64, guide_dim=64, 
                        H=args.image_size, W=args.image_size,mlp_dim=[256, 128], NIR_dim=33).cuda()
       
    model_D = Discriminator(args.n_bands).cuda()
    optimizer_G = torch.optim.Adam(model_G.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(model_D.parameters(), lr=args.lr)
    # Loss and optimizer
    criterion = GeneratorLoss().cuda()
    # criterion = S_F_loss().cuda()
    optimizer_G = torch.optim.Adam(model_G.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(model_D.parameters(), lr=args.lr)
    # Load the trained model parameters
    model_path = args.model_path.replace('dataset', args.dataset) \
                                .replace('arch', args.arch) 
    if os.path.exists(model_path):
        model_G.load_state_dict(torch.load(model_path), strict=False)
        print ('Load the chekpoint of {}'.format(model_path))
        recent_psnr = validate(test_list, 
                                args.arch,
                                model_G,
                                0,
                                args.n_epochs)
        print ('psnr: ', recent_psnr)

    best_psnr = 0
    best_psnr = validate(test_list,
                          args.arch, 
                          model_G,
                          0,
                          args.n_epochs)
    print ('psnr: ', best_psnr)

    # Epochs
    print ('Start Training: ')
    for epoch in range(args.n_epochs):
        # One epoch's training
        print ('Train_Epoch_{}: '.format(epoch))
        h_str = h_rand[epoch]
        w_str = w_rand[epoch]
        train_GAN(train_list, 
              args.image_size,
              args.scale_ratio,
              args.n_bands, 
              args.arch,
              model_G, 
              model_D, 
              optimizer_G,
              optimizer_D,
              criterion, 
              epoch, 
              args.n_epochs,
              h_str ,
              w_str )

        # One epoch's validation
        print ('Val_Epoch_{}: '.format(epoch))
        recent_psnr = validate(test_list, 
                                args.arch,
                                model_G,
                                epoch,
                                args.n_epochs)
        print ('psnr: ', recent_psnr)

        # # save model
        is_best = recent_psnr > best_psnr
        best_psnr = max(recent_psnr, best_psnr)
        if is_best:
          torch.save(model_G.state_dict(), model_path)
          print ('Saved!')
          print ('')

    print ('best_psnr: ', best_psnr)

if __name__ == '__main__':
    main()
