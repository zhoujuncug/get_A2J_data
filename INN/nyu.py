import sys 
sys.path.append('/root/get_A2J_data')
import torch
import torch.utils.data
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import os
from tqdm import tqdm
import logging
import time
import random
import scipy.io as scio

from lib.dataset.NYU.inn import nyu_dataloader
import lib.model.A2J.model as model
import lib.model.A2J.anchor as anchor
from lib.utils.utils import pixel2world, world2pixel, errorCompute, writeTxt
from lib.utils.AE.nyu.utils import show_batch_img
from lib.model.AE.WCGAN import Encoder, Generator
from lib.model.INN.INN import ConditionalTransformer
from lib.model.INN.loss import Loss

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

randomseed = 12345
random.seed(randomseed)
np.random.seed(randomseed)
torch.manual_seed(randomseed)


fx = 588.03
fy = -587.07
u0 = 320
v0 = 240
# # DataHyperParms 
TrainImgFrames = 72757
TestImgFrames = 8252
keypointsNumber = 14
cropWidth = 176
cropHeight = 176
batch_size = 64
learning_rate = 0.00035
Weight_Decay = 1e-4
nepoch = 35
RegLossFactor = 3
spatialFactor = 0.5
RandCropShift = 5
RandshiftDepth = 1
RandRotate = 180 
RandScale = (1.0, 0.5)
xy_thres = 110
depth_thres = 150

result_file = 'result_NYU.txt'

save_dir = './result/NYU_batch_64_12345'

try:
    os.makedirs(save_dir)
except OSError:
    pass

################################################################################################
# using center point to get bbox, in this way, hand size on pixel level is basiclly equivalent.
################################################################################################
# train
trainingImageDir = '/home/public/nyu_hand_dataset_v2/A2J/train_nyu/'
trainingSynImgDir = '/home/public/nyu_hand_dataset_v2/train/'
train_center_file = './data/nyu/nyu_center_train.mat'
train_keypoint_file = './data/nyu/nyu_keypointsUVD_train.mat'

center_train = scio.loadmat(train_center_file)['centre_pixel'].astype(np.float32)
centre_train_world = pixel2world(center_train.copy(), fx, fy, u0, v0)
centerlefttop_train = centre_train_world.copy()
centerlefttop_train[:,0,0] = centerlefttop_train[:,0,0]-xy_thres
centerlefttop_train[:,0,1] = centerlefttop_train[:,0,1]+xy_thres # (72757, 1, 3)

train_lefttop_pixel = world2pixel(centerlefttop_train, fx, fy, u0, v0)
centerrightbottom_train = centre_train_world.copy()
centerrightbottom_train[:,0,0] = centerrightbottom_train[:,0,0]+xy_thres
centerrightbottom_train[:,0,1] = centerrightbottom_train[:,0,1]-xy_thres
train_rightbottom_pixel = world2pixel(centerrightbottom_train, fx, fy, u0, v0) # (72757, 1, 3)

# test
testingImageDir = '/home/public/nyu_hand_dataset_v2/A2J/test_nyu/'
testingSynImgDir = '/home/public/nyu_hand_dataset_v2/test/'
test_center_file = './data/nyu/nyu_center_test.mat'
test_keypoint_file = './data/nyu/nyu_keypointsUVD_test.mat'

center_test = scio.loadmat(test_center_file)['centre_pixel'].astype(np.float32)
centre_test_world = pixel2world(center_test.copy(), fx, fy, u0, v0)
centerlefttop_test = centre_test_world.copy()
centerlefttop_test[:,0,0] = centerlefttop_test[:,0,0]-xy_thres
centerlefttop_test[:,0,1] = centerlefttop_test[:,0,1]+xy_thres
test_lefttop_pixel = world2pixel(centerlefttop_test, fx, fy, u0, v0)

centerrightbottom_test = centre_test_world.copy()
centerrightbottom_test[:,0,0] = centerrightbottom_test[:,0,0]+xy_thres
centerrightbottom_test[:,0,1] = centerrightbottom_test[:,0,1]-xy_thres
test_rightbottom_pixel = world2pixel(centerrightbottom_test, fx, fy, u0, v0)


keypointsUVD_train = scio.loadmat(train_keypoint_file)['keypoints3D'].astype(np.float32)  # (72757, 14, 3)
keypointsUVD_test = scio.loadmat(test_keypoint_file)['keypoints3D'].astype(np.float32) 

MEAN = np.load('./data/nyu/nyu_mean.npy')
STD = np.load('./data/nyu/nyu_std.npy')

train_image_datasets = nyu_dataloader(trainingImageDir, trainingSynImgDir, center_train, train_lefttop_pixel, 
                                      train_rightbottom_pixel, keypointsUVD_train, 
                                      MEAN, STD, xy_thres, depth_thres, cropHeight, cropWidth, keypointsNumber,
                                      RandCropShift, RandshiftDepth, RandRotate,  RandScale, augment=False)
train_dataloaders = torch.utils.data.DataLoader(train_image_datasets, batch_size = batch_size,
                                             shuffle = True, num_workers = 0)

test_image_datasets = nyu_dataloader(testingImageDir, testingSynImgDir, center_test, test_lefttop_pixel, 
                                     test_rightbottom_pixel, keypointsUVD_test, 
                                     MEAN, STD, xy_thres, depth_thres, cropHeight, cropWidth, keypointsNumber,
                                     RandCropShift, RandshiftDepth, RandRotate,  RandScale, augment=False)
test_dataloaders = torch.utils.data.DataLoader(test_image_datasets, batch_size = batch_size,
                                             shuffle = False, num_workers = 0)


netE = Encoder().cuda()
netE.load_state_dict(torch.load('./output/checkpoint/nyu/AE/E.pth'))
netE.eval()

netG = Generator().cuda()
netG.load_state_dict(torch.load('./output/checkpoint/nyu/AE/G.pth'))
netG.eval()

inn = ConditionalTransformer()
inn = inn.cuda()


loss_fn = Loss()
optimizerINN = torch.optim.Adam([
                {"params": inn.parameters()},
                {"params": loss_fn.parameters(),
                 "lr": 4.5e-6}], 4.5e-6, betas=(0.5, 0.9))


def run_dataloader(dataloader):
    for i, (img, synimg, label) in enumerate(dataloader):
        torch.cuda.synchronize() 

        img, synimg, label = img.cuda(), synimg.cuda(), label.cuda()

        inn.zero_grad()
        
        z_ = netE(img)
        v, logdet = inn(z_, synimg, train=True)
        losses_inn = 0
        for v_ , log in zip(v, logdet):
            loss, log_dict = loss_fn(v_, log)
            losses_inn += loss * 0.1

        loss_inn_all = losses_inn
        loss_inn_all.backward()
        optimizerINN.step()

        if i % 400 == 0:
            os.makedirs('output/log/nyu/INN/', exist_ok=True)

            results = []
            with torch.no_grad():
                for k in range(7):
                    for j in range(7):
                        if 7 * k + j == 0:
                            results.append(img[[0]])
                            continue
                        elif 7 * k + j == 1:
                            results.append(synimg[[0]])
                            continue
                        elif 7 * k + j == 2:
                            v, logdet = inn(z_, synimg)
                            rec_z_ = inn.reverse(v, synimg)
                            rec_dep = netG(rec_z_)
                            results.append(rec_dep[[0]])
                            continue
                        v_sample = torch.randn_like(v)
                        rec_z_ = inn.reverse(v_sample, synimg)
                        rec_dep = netG(rec_z_)
                        results.append(rec_dep[[0]])

            results = torch.cat(results, dim=0)
            show_batch_img(results, f'output/log/nyu/INN/{epoch}_{i}_sample_v.jpg')


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
    run_dataloader(train_dataloaders, phase='Train')
    run_dataloader(test_dataloaders, phase='Test')
    
    
    os.makedirs(f'./output/checkpoint/nyu/INN/', exist_ok=True)
    torch.save(inn.state_dict(), f'./output/checkpoint/nyu/AE/epoch_{epoch}_inn.pth')