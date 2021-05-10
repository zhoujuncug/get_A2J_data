import numpy as np
np.set_printoptions(threshold=np.inf, linewidth=1000)
import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_anchors(P_h=None, P_w=None):
    if P_h is None:
        P_h = np.array([2,6,10,14])

    if P_w is None:
        P_w = np.array([2,6,10,14])

    num_anchors = len(P_h) * len(P_h)

    # initialize output anchors
    anchors = np.zeros((num_anchors, 2))
    k = 0
    for i in range(len(P_w)):
        for j in range(len(P_h)):
            anchors[k,1] = P_w[j]
            anchors[k,0] = P_h[i]
            k += 1  
    return anchors          

def shift(shape, stride, anchors):
    # shape = (11, 11)
    # stride = 16
    # anchors = (16, 2)      [2,6,10,14]
    shift_h = np.arange(0, shape[0]) * stride
    shift_w = np.arange(0, shape[1]) * stride  # [  0  16  32  48  64  80  96 112 128 144 160]

    shift_h, shift_w = np.meshgrid(shift_h, shift_w)  # (11, 11)  (11, 11)
    # shift_h
    # [[  0  16  32  48  64  80  96 112 128 144 160]
    #  [  0  16  32  48  64  80  96 112 128 144 160]
    #  [  0  16  32  48  64  80  96 112 128 144 160]
    #  [  0  16  32  48  64  80  96 112 128 144 160]
    #  [  0  16  32  48  64  80  96 112 128 144 160]
    #  [  0  16  32  48  64  80  96 112 128 144 160]
    #  [  0  16  32  48  64  80  96 112 128 144 160]
    #  [  0  16  32  48  64  80  96 112 128 144 160]
    #  [  0  16  32  48  64  80  96 112 128 144 160]
    #  [  0  16  32  48  64  80  96 112 128 144 160]
    #  [  0  16  32  48  64  80  96 112 128 144 160]]
    # shift_w
    # [[  0   0   0   0   0   0   0   0   0   0   0]
    #  [ 16  16  16  16  16  16  16  16  16  16  16]
    #  [ 32  32  32  32  32  32  32  32  32  32  32]
    #  [ 48  48  48  48  48  48  48  48  48  48  48]
    #  [ 64  64  64  64  64  64  64  64  64  64  64]
    #  [ 80  80  80  80  80  80  80  80  80  80  80]
    #  [ 96  96  96  96  96  96  96  96  96  96  96]
    #  [112 112 112 112 112 112 112 112 112 112 112]
    #  [128 128 128 128 128 128 128 128 128 128 128]
    #  [144 144 144 144 144 144 144 144 144 144 144]
    #  [160 160 160 160 160 160 160 160 160 160 160]]
    shifts = np.vstack((shift_h.ravel(), shift_w.ravel())).transpose()  # (121, 2)

    # add A anchors (1, A, 2) to
    # cell K shifts (K, 1, 2) to get
    # shift anchors (K, A, 2)
    # reshape to (K*A, 4) shifted anchors
    A = anchors.shape[0]
    K = shifts.shape[0]

    # Anchor
    # [[ 2.  2.]
    #  [ 2.  6.]
    #  [ 2. 10.]
    #  [ 2. 14.]
    #  [ 6.  2.]
    #  [ 6.  6.]
    #  [ 6. 10.]
    #  [ 6. 14.]
    #  [10.  2.]
    #  [10.  6.]
    #  [10. 10.]
    #  [10. 14.]
    #  [14.  2.]
    #  [14.  6.]
    #  [14. 10.]
    #  [14. 14.]]

    # print(anchors.reshape((1, A, 2)) + np.zeros_like(shifts.reshape((1, K, 2)).transpose((1, 0, 2))))
    all_anchors = (anchors.reshape((1, A, 2)) + shifts.reshape((1, K, 2)).transpose((1, 0, 2)))  # （121, 16, 2)
    all_anchors = all_anchors.reshape((K * A, 2))  # (121*6, 2)

    return all_anchors

class post_process(nn.Module):
    def __init__(self, P_h=[2,6], P_w=[2,6], shape=[48,26], stride=8,thres = 8,is_3D=True):
        super(post_process, self).__init__()
        # shape = (11, 11)
        # stride = 16
        # P_h = None
        # P_w = None
        # thres = 8
        # is_3D = True
        anchors = generate_anchors(P_h=P_h,P_w=P_w)  # (16, 2)      [2,6,10,14]
        self.all_anchors = torch.from_numpy(shift(shape,stride,anchors)).cuda().float()  # (121*16, 2)
        self.thres = torch.from_numpy(np.array(thres)).cuda().float()
        self.is_3D = is_3D
    def calc_distance(self, a, b):
        dis = torch.zeros(a.shape[0],b.shape[0]).cuda()
        for i in range(a.shape[1]):
            dis += torch.pow(torch.unsqueeze(a[:, i], dim=1) - b[:,i],0.5)
        return dis

    def forward(self,heads,voting=False):
        if self.is_3D:
            classifications, regressions, depthregressions = heads
        else:
            classifications, regressions = heads
        batch_size = classifications.shape[0]
        anchor = self.all_anchors #*(w*h*A)*2                       ## (1936, 2)
        P_keys = []
        for j in range(batch_size):

            classification = classifications[j, :, :] #N*(w*h*A)*P  ## (1936, 14)
            regression = regressions[j, :, :, :] #N*(w*h*A)*P*2     ## (1936, 14, 2)
 
            if self.is_3D:
                depthregression = depthregressions[j, :, :]#N*(w*h*A)*P  ## (1936, 14)
            reg = torch.unsqueeze(anchor,1) + regression            ## (1936, 14, 2)

            reg_weight = F.softmax(classifications[j, :, :],dim=0) #(w*h*A)*P  ## (1936, 14)
            reg_weight_xy = torch.unsqueeze(reg_weight,2).expand(reg_weight.shape[0],reg_weight.shape[1],2)#(w*h*A)*P*2  ## (1936, 14, 2)
            P_xy = (reg_weight_xy*reg).sum(0)  # (14, 2)

            if self.is_3D:
                P_depth = (reg_weight*depthregression).sum(0)
                P_depth = torch.unsqueeze(P_depth,1)
                P_key = torch.cat((P_xy,P_depth),1)            
                P_keys.append(P_key)
            else:
                P_keys.append(P_xy)
        return torch.stack(P_keys)

class A2J_loss(nn.Module):
    def __init__(self,P_h=[2,6], P_w=[2,6], shape=[8,4], stride=8,thres = [10.0,20.0],spatialFactor=0.1,img_shape=[0,0],is_3D=True):
        super(A2J_loss, self).__init__()
        # P_h = None
        # P_w = None 
        # shape = [11, 11]
        # stride = 16
        # thres = [16.0, 32.0]
        # spatialFactor = 0.5
        # img_shape = [176, 176]
        # is_3D = True

        anchors = generate_anchors(P_h=P_h, P_w=P_w)
        # [[ 2.  2.]
        #  [ 2.  6.]
        #  [ 2. 10.]
        #  [ 2. 14.]
        #  [ 6.  2.]
        #  [ 6.  6.]
        #  [ 6. 10.]
        #  [ 6. 14.]
        #  [10.  2.]
        #  [10.  6.]
        #  [10. 10.]
        #  [10. 14.]
        #  [14.  2.]
        #  [14.  6.]
        #  [14. 10.]
        #  [14. 14.]]
        self.all_anchors = torch.from_numpy(shift(shape,stride,anchors)).cuda().float()  # (121*16, 2)

        self.thres = torch.from_numpy(np.array(thres)).cuda().float()  # [16.0, 32.0]
        self.spatialFactor = spatialFactor  # 0.5
        self.img_shape = img_shape  # [176, 176]
        self.is_3D = is_3D  # True
    def calc_distance(self, a, b):
        dis = torch.zeros(a.shape[0],b.shape[0]).cuda()
        for i in range(a.shape[1]):
            dis += torch.abs(torch.unsqueeze(a[:, i], dim=1) - b[:,i])  
        return dis

    def forward(self, heads, annotations):
        # annotations: (64, 14, 3)  
        alpha = 0.25
        gamma = 2.0
        if self.is_3D:
            classifications, regressions, depthregressions = heads  # (64, 1936, 14)  (64, 1936, 14, 2)  (64, 1936, 14)
        else:
            classifications, regressions = heads
        #classifications,scalar,mu = classifications_tuple
        batch_size = classifications.shape[0]  # 64
        classification_losses = []
        regression_losses = []

        anchor = self.all_anchors # num_anchors(w*h*A) x 2    # (11*11*16, 2)
        anchor_regression_loss_tuple = []

        for j in range(batch_size):
            # P is joint id.
            classification = classifications[j, :, :] #N*(w*h*A)*P
            regression = regressions[j, :, :, :] #N*(w*h*A)*P*2
            if self.is_3D:
                depthregression = depthregressions[j, :, :]#N*(w*h*A)*P
            
            reg_weight = F.softmax(classification,dim=0) #(w*h*A)*P
            reg_weight_xy = torch.unsqueeze(reg_weight,2).expand(reg_weight.shape[0],reg_weight.shape[1],2)#(w*h*A)*P*2     ## (1936, 14, 2)

            bbox_annotation = annotations[j, :, :]#N*P*3=>P*3
            gt_xy = bbox_annotation[:,:2]#P*2 

            # gt_xy: (14, 2)
            # reg_weight_xy: (1936, 14, 2)
            # torch.unsqueeze(anchor,1): (1936, 1, 2)
            anchor_diff = torch.abs(gt_xy-(reg_weight_xy*torch.unsqueeze(anchor,1)).sum(0)) #P*2
            anchor_loss = torch.where(   ## smooth l1 loss
                torch.le(anchor_diff, 1),
                0.5 * 1 * torch.pow(anchor_diff, 2),
                anchor_diff - 0.5 / 1
            )
            anchor_regression_loss = anchor_loss.mean()
            anchor_regression_loss_tuple.append(anchor_regression_loss)
#######################regression 4 spatial###################
            reg = torch.unsqueeze(anchor,1) + regression #(w*h*A)*P*2  ## (1936, 14, 2) = (1936, 1, 2) + (1936, 14, 2)
            regression_diff = torch.abs(gt_xy-(reg_weight_xy*reg).sum(0)) #P*2
            regression_loss = torch.where(
                torch.le(regression_diff, 1),
                0.5 * 1 * torch.pow(regression_diff, 2),
                regression_diff - 0.5 / 1
                )
            regression_loss = regression_loss.mean()*self.spatialFactor
########################regression 4 depth###################
            if self.is_3D:
                gt_depth = bbox_annotation[:,2] #P
                regression_diff_depth = torch.abs(gt_depth - (reg_weight*depthregression).sum(0))#(w*h*A)*P    
                regression_loss_depth = torch.where(
                    torch.le(regression_diff_depth, 3),
                    0.5 * (1/3) * torch.pow(regression_diff_depth, 2),
                    regression_diff_depth - 0.5 / (1/3)
                    )
                regression_loss += regression_diff_depth.mean()           
############################################################
            regression_losses.append(regression_loss)
        return torch.stack(anchor_regression_loss_tuple).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(dim=0, keepdim=True)   
