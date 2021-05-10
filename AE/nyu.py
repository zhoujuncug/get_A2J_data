import sys 
sys.path.append('/root/get_A2J_data')

import cv2
import torch
import torch.nn as nn
import torch.utils.data
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import scipy.io as scio
import os
from PIL import Image
from torch.autograd import Variable
import lib.model.A2J.model as model
import lib.model.A2J.anchor as anchor
from tqdm import tqdm
import lib.model.A2J.random_erasing
import logging
import time
import datetime
import random
torch.set_printoptions(precision=8)

from lib.model.AE.WCGAN import Encoder, Generator, Discriminator
from lib.dataset.NYU.nyu import nyu_dataloader, center_train, train_lefttop_pixel, train_rightbottom_pixel, keypointsUVD_train, batch_size
from lib.dataset.NYU.nyu import center_test, test_lefttop_pixel, test_rightbottom_pixel, keypointsUVD_test, errorCompute, writeTxt
from lib.utils.AE.nyu.utils import show_batch_img
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# DataHyperParms 
TrainImgFrames = 72757
TestImgFrames = 8252
keypointsNumber = 14
cropWidth = 176
cropHeight = 176
batch_size = 64
learning_rate = 0.00035
Weight_Decay = 1e-4
nepoch = 100
RegLossFactor = 3
spatialFactor = 0.5
RandCropShift = 5
RandshiftDepth = 1
RandRotate = 180 
RandScale = (1.0, 0.5)
xy_thres = 110
depth_thres = 150

trainingImageDir = '/home/public/nyu_hand_dataset_v2/A2J/train_nyu/'
testingImageDir = '/home/public/nyu_hand_dataset_v2/A2J/test_nyu/'  # mat images
test_center_file = './data/nyu/nyu_center_test.mat'
test_keypoint_file = './data/nyu/nyu_keypointsUVD_test.mat'
train_center_file = './data/nyu/nyu_center_train.mat'
train_keypoint_file = './data/nyu/nyu_keypointsUVD_train.mat'
MEAN = np.load('./data/nyu/nyu_mean.npy')
STD = np.load('./data/nyu/nyu_std.npy')
model_dir = './model/NYU.pth'
result_file = 'result_NYU.txt'

randomseed = 12345
random.seed(randomseed)
np.random.seed(randomseed)
torch.manual_seed(randomseed)

save_dir = './result/NYU_batch_64_12345'

try:
    os.makedirs(save_dir)
except OSError:
    pass


train_image_datasets = nyu_dataloader(trainingImageDir, center_train, train_lefttop_pixel, train_rightbottom_pixel, keypointsUVD_train, augment=False)
train_dataloaders = torch.utils.data.DataLoader(train_image_datasets, batch_size = batch_size,
                                             shuffle = True, num_workers = 0)

test_image_datasets = nyu_dataloader(testingImageDir, center_test, test_lefttop_pixel, test_rightbottom_pixel, keypointsUVD_test, augment=False)
test_dataloaders = torch.utils.data.DataLoader(test_image_datasets, batch_size = batch_size,
                                             shuffle = False, num_workers = 0)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)

netE = Encoder().cuda()
netE.apply(weights_init)

netG = Generator().cuda()
netG.apply(weights_init)

netD = Discriminator().cuda()
netD.apply(weights_init)

criterion = nn.BCELoss()

real_label = 1
fake_label = 0

optimizerD = torch.optim.Adam(netD.parameters(), lr=0.00005, betas=(0.5, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerE = torch.optim.Adam(netE.parameters(), lr=0.0002, betas=(0.5, 0.999))


def run_dataloader(dataloader, phase, is_gan, p_D, p_G):
    oD = 0
    oG = 0
    for i, (img, label) in enumerate(dataloader):

        torch.cuda.synchronize() 

        # train with real
        netD.zero_grad()
        img, label = img.cuda(), label.cuda()
        
        batch_size = img.size(0)
        label = torch.full((batch_size,), real_label,
                           dtype=img.dtype, device=torch.device('cuda'))
        output = netD(img)

        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # train with fake
        z = netE(img)
        fake = netG(z)
        label.fill_(fake_label)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()

        errD = errD_real + errD_fake

        oD += p_D
        if oD >= 1:
            oD -= 1
            print('D')
            optimizerD.step()

        netG.zero_grad()
        netE.zero_grad()

        label.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, label)

        recG = torch.nn.functional.l1_loss(img, fake) * 5

        LossG = recG + errG if is_gan else recG
        # LossG = errG
        LossG.backward()

        D_G_z2 = output.mean().item()

        oG += p_G
        if oG >= 1:
            oG -= 1
            print('G')
            optimizerG.step()
            optimizerE.step()

        # D_x += 1e-10
        # D_G_z1 += 1e-10
        # p_D = 2 * (1 - (D_x / (D_x + D_G_z1)))
        # p_G = 2 * (1 / 2. - D_G_z1 / (D_x + D_G_z1))
        # p_D = max(p_D, 0.2)
        # p_G = max(p_G, 0.2)
        # p_D = 1
        # p_G = 1

        print(f'[{epoch}/{nepoch}][{i}/{len(dataloader)}] {phase} Loss_D: {errD.item():.4f} Loss_G: {LossG.item():.4f} errG: {errG.item():.4f} RecG: {recG.item():.4f} D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}\tpD: {p_D:.2f}\tpE: {p_G:.2f}')

        if i % 1000 == 0:
            os.makedirs('output/log/nyu/AE', exist_ok=True)
            real_fake = torch.cat([img[:, None, :, :, :], fake[:, None, :, :, :]], dim=1)
            real_fake = real_fake.view([-1, img.shape[1], img.shape[2], img.shape[3]])
            show_batch_img(real_fake, f'output/log/nyu/AE/{epoch}_{i}_{phase}.jpg', nrow=8)

for epoch in range(nepoch):
    netE, netG, netD = netE.cuda(), netG.cuda(), netD.cuda()
    timer = time.time()

    # Training loop

    # if epoch < 1:
    #     lr = 0.06
    # else:
    #     lr = 0.0002

    # for param_group in optimizerD.param_groups:
    #     param_group['lr'] = lr

    is_gan = True # if epoch > 0 else False
    if epoch in [0, 1]:
        p_D = 1/3.
    elif epoch in [2, 3, 4]:
        p_D = 1/3.
    else: 
        p_D = 1/3.
    p_G = 1
    run_dataloader(train_dataloaders, 'Train', is_gan, p_D, p_G)
    # run_dataloader(test_dataloaders, phase='Test', is_gan=is_gan)
    
    os.makedirs(f'./output/checkpoint/nyu/AE/epoch_{epoch}', exist_ok=True)
    torch.save(netE.state_dict(), f'./output/checkpoint/nyu/AE/epoch_{epoch}/E.pth')
    torch.save(netG.state_dict(), f'./output/checkpoint/nyu/AE/epoch_{epoch}/G.pth')
    torch.save(netD.state_dict(), f'./output/checkpoint/nyu/AE/epoch_{epoch}/D.pth')
