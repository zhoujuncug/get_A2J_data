import scipy.io as scio
import numpy as np
import random
import cv2
import torch

fx = 588.03
fy = -587.07
u0 = 320
v0 = 240

# DataHyperParms
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

randomseed = 12345
random.seed(randomseed)
np.random.seed(randomseed)

def transform(img, label, matrix):
    '''
    img: [H, W]  label, [N,2]
    '''
    img_out = cv2.warpAffine(img,matrix,(cropWidth,cropHeight))
    label_out = np.ones((keypointsNumber, 3))
    label_out[:,:2] = label[:,:2].copy()
    label_out = np.matmul(matrix, label_out.transpose())
    label_out = label_out.transpose()

    return img_out, label_out

def dataPreprocess(index, img, keypointsUVD, center, mean, std, lefttop_pixel, rightbottom_pixel, xy_thres=90,
                   depth_thres=75, augment=True):
    imageOutputs = np.ones((cropHeight, cropWidth, 1), dtype='float32')
    labelOutputs = np.ones((keypointsNumber, 3), dtype='float32')

    if augment:
        RandomOffset_1 = np.random.randint(-1 * RandCropShift, RandCropShift)
        RandomOffset_2 = np.random.randint(-1 * RandCropShift, RandCropShift)
        RandomOffset_3 = np.random.randint(-1 * RandCropShift, RandCropShift)
        RandomOffset_4 = np.random.randint(-1 * RandCropShift, RandCropShift)
        RandomOffsetDepth = np.random.normal(0, RandshiftDepth, cropHeight * cropWidth).reshape(cropHeight, cropWidth)
        RandomOffsetDepth[np.where(RandomOffsetDepth < RandshiftDepth)] = 0
        RandomRotate = np.random.randint(-1 * RandRotate, RandRotate)
        RandomScale = np.random.rand() * RandScale[0] + RandScale[1]
        matrix = cv2.getRotationMatrix2D((cropWidth / 2, cropHeight / 2), RandomRotate, RandomScale)
    else:
        RandomOffset_1, RandomOffset_2, RandomOffset_3, RandomOffset_4 = 0, 0, 0, 0
        RandomRotate = 0
        RandomScale = 1
        RandomOffsetDepth = 0
        matrix = cv2.getRotationMatrix2D((cropWidth / 2, cropHeight / 2), RandomRotate, RandomScale)

    new_Xmin = max(lefttop_pixel[index, 0, 0] + RandomOffset_1, 0)
    new_Ymin = max(lefttop_pixel[index, 0, 1] + RandomOffset_2, 0)
    new_Xmax = min(rightbottom_pixel[index, 0, 0] + RandomOffset_3, img.shape[1] - 1)
    new_Ymax = min(rightbottom_pixel[index, 0, 1] + RandomOffset_4, img.shape[0] - 1)

    imCrop = img[int(new_Ymin):int(new_Ymax), int(new_Xmin):int(new_Xmax)].copy()

    imgResize = cv2.resize(imCrop, (cropWidth, cropHeight), interpolation=cv2.INTER_NEAREST)

    imgResize = np.asarray(imgResize, dtype='float32')  # H*W*C

    imgResize[np.where(imgResize >= center[index][0][2] + depth_thres)] = center[index][0][2]
    imgResize[np.where(imgResize <= center[index][0][2] - depth_thres)] = center[index][0][2]
    imgResize = (imgResize - center[index][0][2]) * RandomScale

    imgResize = (imgResize - mean) / std

    ## label
    label_xy = np.ones((keypointsNumber, 2), dtype='float32')
    label_xy[:, 0] = (keypointsUVD[index, :, 0].copy() - new_Xmin) * cropWidth / (new_Xmax - new_Xmin)  # x
    label_xy[:, 1] = (keypointsUVD[index, :, 1].copy() - new_Ymin) * cropHeight / (new_Ymax - new_Ymin)  # y

    if augment:
        imgResize, label_xy = transform(imgResize, label_xy, matrix)  ## rotation, scale

    imageOutputs[:, :, 0] = imgResize

    labelOutputs[:, 1] = label_xy[:, 0]
    labelOutputs[:, 0] = label_xy[:, 1]
    labelOutputs[:, 2] = (keypointsUVD[index, :, 2] - center[index][0][2]) * RandomScale  # Z

    imageOutputs = np.asarray(imageOutputs)
    imageNCHWOut = imageOutputs.transpose(2, 0, 1)  # [H, W, C] --->>>  [C, H, W]
    imageNCHWOut = np.asarray(imageNCHWOut)
    labelOutputs = np.asarray(labelOutputs)

    data, label = torch.from_numpy(imageNCHWOut), torch.from_numpy(labelOutputs)

    return data, label


def world2pixel(x, fx, fy, ux, uy):
    x[:, :, 0] = x[:, :, 0] * fx / x[:, :, 2] + ux
    x[:, :, 1] = x[:, :, 1] * fy / x[:, :, 2] + uy
    return x


def pixel2world(x, fx, fy, ux, uy):
    x[:, :, 0] = (x[:, :, 0] - ux) * x[:, :, 2] / fx
    x[:, :, 1] = (x[:, :, 1] - uy) * x[:, :, 2] / fy
    return x


train_center_file = './nyu_center_train.mat'

center_train = scio.loadmat(train_center_file)['centre_pixel'].astype(np.float32)
centre_train_world = pixel2world(center_train.copy(), fx, fy, u0, v0)

centerlefttop_train = centre_train_world.copy()
centerlefttop_train[:,0,0] = centerlefttop_train[:,0,0]-xy_thres
centerlefttop_train[:,0,1] = centerlefttop_train[:,0,1]+xy_thres

centerrightbottom_train = centre_train_world.copy()
centerrightbottom_train[:,0,0] = centerrightbottom_train[:,0,0]+xy_thres
centerrightbottom_train[:,0,1] = centerrightbottom_train[:,0,1]-xy_thres


p = 'temp.mat'
depth = scio.loadmat(p)['Depth']

train_center_file = './nyu_center_train.mat'
center_train = scio.loadmat(train_center_file)['centre_pixel'].astype(np.float32)

train_keypoint_file = './nyu_keypointsUVD_train.mat'
keypointsUVD_train = scio.loadmat(train_keypoint_file)['keypoints3D'].astype(np.float32)

MEAN = np.load('./nyu_mean.npy')
STD = np.load('./nyu_std.npy')

train_lefttop_pixel = world2pixel(centerlefttop_train, fx, fy, u0, v0)
train_rightbottom_pixel = world2pixel(centerrightbottom_train, fx, fy, u0, v0)

data, label = dataPreprocess(1, depth, keypointsUVD_train, center_train, MEAN, STD,
                             train_lefttop_pixel, train_rightbottom_pixel, xy_thres, depth_thres, True)

print(data.shape)
print(label)


