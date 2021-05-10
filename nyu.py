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

from lib.dataset.nyu_cp import nyu_dataloader
import lib.model.A2J.model as model
import lib.model.A2J.anchor as anchor
from lib.utils.utils import pixel2world, world2pixel, errorCompute, writeTxt

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

train_image_datasets = nyu_dataloader(trainingImageDir, center_train, train_lefttop_pixel, 
                                      train_rightbottom_pixel, keypointsUVD_train, 
                                      MEAN, STD, xy_thres, depth_thres, cropHeight, cropWidth, keypointsNumber,
                                      RandCropShift, RandshiftDepth, RandRotate,  RandScale, augment=True)
train_dataloaders = torch.utils.data.DataLoader(train_image_datasets, batch_size = batch_size,
                                             shuffle = True, num_workers = 0)

test_image_datasets = nyu_dataloader(testingImageDir, center_test, test_lefttop_pixel, 
                                     test_rightbottom_pixel, keypointsUVD_test, 
                                     MEAN, STD, xy_thres, depth_thres, cropHeight, cropWidth, keypointsNumber,
                                     RandCropShift, RandshiftDepth, RandRotate,  RandScale, augment=False)
test_dataloaders = torch.utils.data.DataLoader(test_image_datasets, batch_size = batch_size,
                                             shuffle = False, num_workers = 0)


net = model.A2J_model(num_classes = keypointsNumber)
net = net.cuda()
# net.load_state_dict(torch.load('./output/checkpoint/A2J/official/NYU.pth'))

post_precess = anchor.post_process(shape=[cropHeight//16,cropWidth//16],stride=16,P_h=None, P_w=None)
criterion = anchor.A2J_loss(shape=[cropHeight//16,cropWidth//16],thres = [16.0,32.0],stride=16,\
    spatialFactor=spatialFactor,img_shape=[cropHeight, cropWidth],P_h=None, P_w=None)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=Weight_Decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)


for epoch in range(nepoch):
    net = net.train()
    train_loss_add = 0.0
    Cls_loss_add = 0.0
    Reg_loss_add = 0.0
    timer = time.time()

    # Training loop
    for i, (img, label) in enumerate(train_dataloaders):
        torch.cuda.synchronize() 

        img, label = img.cuda(), label.cuda()     
        
        heads  = net(img)  # (64, 1936, 14)  (64, 1936, 14, 2)  (64, 1936, 14)
        #print(regression)     
        optimizer.zero_grad()  
        
        Cls_loss, Reg_loss = criterion(heads, label)

        loss = 1*Cls_loss + Reg_loss*RegLossFactor
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        
        train_loss_add = train_loss_add + (loss.item())*len(img)
        Cls_loss_add = Cls_loss_add + (Cls_loss.item())*len(img)
        Reg_loss_add = Reg_loss_add + (Reg_loss.item())*len(img)

        # printing loss info
        if i%10 == 0:
            print('epoch: ',epoch, ' step: ', i, 'Cls_loss ',Cls_loss.item(), 'Reg_loss ',Reg_loss.item(), ' total loss ',loss.item())

    scheduler.step(epoch)

    # time taken
    torch.cuda.synchronize()
    timer = time.time() - timer
    timer = timer / TrainImgFrames
    print('==> time to learn 1 sample = %f (ms)' %(timer*1000))

    train_loss_add = train_loss_add / TrainImgFrames
    Cls_loss_add = Cls_loss_add / TrainImgFrames
    Reg_loss_add = Reg_loss_add / TrainImgFrames
    print('mean train_loss_add of 1 sample: %f, #train_indexes = %d' %(train_loss_add, TrainImgFrames))
    print('mean Cls_loss_add of 1 sample: %f, #train_indexes = %d' %(Cls_loss_add, TrainImgFrames))
    print('mean Reg_loss_add of 1 sample: %f, #train_indexes = %d' %(Reg_loss_add, TrainImgFrames))

    Error_test = 0
    Error_train = 0
    Error_test_wrist = 0

    if (epoch % 1 == 0):  
        net = net.eval()
        output = torch.FloatTensor()
        outputTrain = torch.FloatTensor()

        for i, (img, label) in tqdm(enumerate(test_dataloaders)):
            with torch.no_grad():
                img, label = img.cuda(), label.cuda()       
                heads = net(img)  
                pred_keypoints = post_precess(heads, voting=False)
                output = torch.cat([output,pred_keypoints.data.cpu()], 0)

        result = output.cpu().data.numpy()
        Error_test = errorCompute(result,keypointsUVD_test, center_test, 
                                  fx, fy, u0, v0, xy_thres, cropWidth, cropHeight)
        print('epoch: ', epoch, 'Test error:', Error_test)
        saveNamePrefix = '%s/net_%d_wetD_' % (save_dir, epoch) + str(Weight_Decay) + '_depFact_' + str(spatialFactor) + '_RegFact_' + str(RegLossFactor) + '_rndShft_' + str(RandCropShift)
        torch.save(net.state_dict(), saveNamePrefix + '.pth')

    # log
    logging.info('Epoch#%d: total loss=%.4f, Cls_loss=%.4f, Reg_loss=%.4f, Err_test=%.4f, lr = %.6f'
    %(epoch, train_loss_add, Cls_loss_add, Reg_loss_add, Error_test, scheduler.get_lr()[0]))


net = model.A2J_model(num_classes = keypointsNumber)
# net.load_state_dict(torch.load(model_dir))
net = net.cuda()
net.eval()

post_precess = anchor.post_process(shape=[cropHeight//16,cropWidth//16],stride=16,P_h=None, P_w=None)

output = torch.FloatTensor()
torch.cuda.synchronize() 
for i, (img, label) in tqdm(enumerate(test_dataloaders)):    
    with torch.no_grad():

        img, label = img.cuda(), label.cuda()    
        heads = net(img)  
        pred_keypoints = post_precess(heads,voting=False)
        output = torch.cat([output,pred_keypoints.data.cpu()], 0)
    
torch.cuda.synchronize()       

result = output.cpu().data.numpy()
writeTxt(result, center_test,
         fx, fy, u0, v0, xy_thres, cropWidth, 
         cropHeight, save_dir, result_file, keypointsNumber)

error = errorCompute(result, keypointsUVD_test, center_test,
                     fx, fy, u0, v0, xy_thres, cropWidth, cropHeight)
print('Error:', error)