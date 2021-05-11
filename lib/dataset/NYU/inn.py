import cv2
import torch
import torch.utils.data
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import scipy.io as scio
import os
from PIL import Image
from torch.autograd import Variable
from tqdm import tqdm
from lib.model.A2J import random_erasing
import logging
import time
import datetime
import random
import matplotlib.pyplot as plt


def transform(img, syn, label, matrix,
              cropWidth, cropHeight, keypointsNumber):
    '''
    img: [H, W]  label, [N,2]   
    '''
    img_out = cv2.warpAffine(img,matrix,(cropWidth,cropHeight))
    syn_out = cv2.warpAffine(syn,matrix,(cropWidth,cropHeight))

    label_out = np.ones((keypointsNumber, 3))
    label_out[:,:2] = label[:,:2].copy()
    label_out = np.matmul(matrix, label_out.transpose())
    label_out = label_out.transpose()

    return img_out, syn_out, label_out


def dataPreprocess(index, img, syn, keypointsUVD, center, mean, std, lefttop_pixel, rightbottom_pixel, 
                   cropHeight, cropWidth, keypointsNumber, RandCropShift, RandshiftDepth,
                   RandRotate,  RandScale, 
                   xy_thres=90, depth_thres=75, augment=True):
    
    # mean  # 28.329582076876047
    # std  # -0.66877532422628
    # center  # (N, 1, 3)
    # lefttop_pixel  # (N, 1, 3)
    # rightbottom_pixel  # (N, 1, 3)
    # keypointsUVD  # (N, 14, 3)
    # xy_thres  # 110
    # depth_thres  # 150

    # cropHeight  # 176
    # cropWidth  # 176
    # keypointsNumber  # 14
    # RandCropShift  # 5
    # RandshiftDepth  # 1
    # RandRotate  # 180
    # RandScale  # (1.0, 0.5)

    imageOutputs = np.ones((cropHeight, cropWidth, 1), dtype='float32') 
    synimgOutputs = np.ones((cropHeight, cropWidth, 1), dtype='float32') 
    labelOutputs = np.ones((keypointsNumber, 3), dtype = 'float32') 
    
    if augment:
        RandomOffset_1 = np.random.randint(-1*RandCropShift,RandCropShift)
        RandomOffset_2 = np.random.randint(-1*RandCropShift,RandCropShift)
        RandomOffset_3 = np.random.randint(-1*RandCropShift,RandCropShift)
        RandomOffset_4 = np.random.randint(-1*RandCropShift,RandCropShift)

        RandomOffsetDepth = np.random.normal(0, RandshiftDepth, cropHeight*cropWidth).reshape(cropHeight,cropWidth) 
        RandomOffsetDepth[np.where(RandomOffsetDepth < RandshiftDepth)] = 0

        RandomRotate = np.random.randint(-1*RandRotate,RandRotate)
        RandomScale = np.random.rand()*RandScale[0]+RandScale[1]  # (0.5, 1.5)
        matrix = cv2.getRotationMatrix2D((cropWidth/2,cropHeight/2),RandomRotate,RandomScale)
    else:
        RandomOffset_1, RandomOffset_2, RandomOffset_3, RandomOffset_4 = 0, 0, 0, 0
        RandomRotate = 0
        RandomScale = 1
        RandomOffsetDepth = 0
        matrix = cv2.getRotationMatrix2D((cropWidth/2,cropHeight/2),RandomRotate,RandomScale)

    new_Xmin = max(lefttop_pixel[index,0,0] + RandomOffset_1, 0)  # augment shift (-5, 5) pixel
    new_Ymin = max(lefttop_pixel[index,0,1] + RandomOffset_2, 0)
    new_Xmax = min(rightbottom_pixel[index,0,0] + RandomOffset_3, img.shape[1] - 1)
    new_Ymax = min(rightbottom_pixel[index,0,1] + RandomOffset_4, img.shape[0] - 1)

    imCrop = img[int(new_Ymin):int(new_Ymax), int(new_Xmin):int(new_Xmax)].copy()
    synCrop = syn[int(new_Ymin):int(new_Ymax), int(new_Xmin):int(new_Xmax)].copy()

    imgResize = cv2.resize(imCrop, (cropWidth, cropHeight), interpolation=cv2.INTER_NEAREST)
    synResize = cv2.resize(synCrop, (cropWidth, cropHeight), interpolation=cv2.INTER_NEAREST)

    imgResize = np.asarray(imgResize,dtype = 'float32')  # H*W*C
    synResize = np.asarray(synResize,dtype = 'float32')

    imgResize[np.where(imgResize >= center[index][0][2] + depth_thres)] = center[index][0][2] 
    imgResize[np.where(imgResize <= center[index][0][2] - depth_thres)] = center[index][0][2] 
    imgResize = (imgResize - center[index][0][2])*RandomScale

    synResize[np.where(synResize >= center[index][0][2] + depth_thres)] = center[index][0][2] 
    synResize[np.where(synResize <= center[index][0][2] - depth_thres)] = center[index][0][2] 
    synResize = (synResize - center[index][0][2])*RandomScale

    imgResize = (imgResize - mean) / std
    synResize = (synResize - mean) / std

    ## label
    # here, "devil in detail" should be used.
    label_xy = np.ones((keypointsNumber, 2), dtype = 'float32') 
    label_xy[:,0] = (keypointsUVD[index,:,0].copy() - new_Xmin)*cropWidth/(new_Xmax - new_Xmin) # x
    label_xy[:,1] = (keypointsUVD[index,:,1].copy() - new_Ymin)*cropHeight/(new_Ymax - new_Ymin) # y

    if augment:
        imgResize, synResize, label_xy = transform(imgResize, synResize, label_xy, matrix, \
                                        cropWidth, cropHeight, keypointsNumber)  ## rotation, scale
    
    imageOutputs[:,:,0] = imgResize
    synimgOutputs[:,:,0] = synResize

    labelOutputs[:,1] = label_xy[:,0]
    labelOutputs[:,0] = label_xy[:,1]
    labelOutputs[:,2] = (keypointsUVD[index,:,2] - center[index][0][2])*RandomScale   # Z  
    labelOutputs = np.asarray(labelOutputs)
    
    imageOutputs = np.asarray(imageOutputs)
    imageNCHWOut = imageOutputs.transpose(2, 0, 1)  # [H, W, C] --->>>  [C, H, W]
    imageNCHWOut = np.asarray(imageNCHWOut)

    synimgOutputs = np.asarray(synimgOutputs)
    synimgNCHWOut = synimgOutputs.transpose(2, 0, 1)  # [H, W, C] --->>>  [C, H, W]
    synimgNCHWOut = np.asarray(synimgNCHWOut)

    data, syndata, label = torch.from_numpy(imageNCHWOut), torch.from_numpy(synimgNCHWOut), torch.from_numpy(labelOutputs)

    return data, syndata, label
    

class nyu_dataloader(torch.utils.data.Dataset):

    def __init__(self, ImgDir, SynImgDir, center, lefttop_pixel, rightbottom_pixel, keypointsUVD, 
                 MEAN, STD, xy_thres, depth_thres, cropHeight, cropWidth, keypointsNumber,
                 RandCropShift, RandshiftDepth, RandRotate,  RandScale,
                 augment=True):

        self.ImgDir = ImgDir
        self.SynImgDir = SynImgDir
        self.mean = MEAN  # 28.329582076876047
        self.std = STD  # -0.66877532422628
        self.center = center  # (N, 1, 3)
        self.lefttop_pixel = lefttop_pixel  # (N, 1, 3)
        self.rightbottom_pixel = rightbottom_pixel  # (N, 1, 3)
        self.keypointsUVD = keypointsUVD  # (N, 14, 3)
        self.xy_thres = xy_thres  # 110
        self.depth_thres = depth_thres  # 150

        self.cropHeight = cropHeight  # 176
        self.cropWidth = cropWidth  # 176
        self.keypointsNumber = keypointsNumber  # 14
        self.RandCropShift = RandCropShift  # 5
        self.RandshiftDepth = RandshiftDepth  # 1
        self.RandRotate = RandRotate  # 180
        self.RandScale = RandScale  # (1.0, 0.5)

        self.augment = augment
        self.randomErase = random_erasing.RandomErasing(probability = 0.5, sl = 0.02, sh = 0.4, 
                                                       r1 = 0.3, mean=[0])  # make a random noise area.
    def __getitem__(self, index):

        depth = scio.loadmat(self.ImgDir + str(index+1) + '.mat')['depth']
        syn = cv2.imread(self.SynImgDir + f'synthdepth_1_{index:07d}.png') 
        syn = np.asarray(syn[:,:,0] + syn[:, :, 1] * 256, dtype=np.float32)

        # label: (x, y, Z)
        data, syndata, label = dataPreprocess(index, depth, syn, self.keypointsUVD, self.center, self.mean, self.std, \
                                     self.lefttop_pixel, self.rightbottom_pixel, \
                                     self.cropHeight, self.cropWidth, self.keypointsNumber, self.RandCropShift,\
                                     self.RandshiftDepth, self.RandRotate, self.RandScale, \
                                     self.xy_thres, self.depth_thres, self.augment)

        if self.augment:
            data = self.randomErase(data)

        return data, syndata, label
    
    def __len__(self):
        return len(self.center)