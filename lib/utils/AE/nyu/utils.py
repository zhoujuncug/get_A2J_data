import numpy as np
import cv2
import torch
import torchvision
import matplotlib.pyplot as plt
from lib.dataset.NYU.nyu import MEAN, STD


def show_batch_img(img_batch, filename, nrow=7):
    img_batch = img_batch.repeat([1, 3, 1, 1])
    grid = torchvision.utils.make_grid(img_batch, nrow=nrow, padding=10).detach().cpu()
    grid = grid * torch.from_numpy(STD) + torch.from_numpy(MEAN)

    img = grid.numpy().transpose(2, 1, 0)
    img = (img - img.min()) / (img.max() - img.min())
    plt.figure()
    plt.tight_layout()
    plt.imshow(img)
    plt.savefig(filename, dpi=600)
    plt.close()


nyu_skeleton = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (9, 10), (1, 13),
                (3, 13), (5, 13), (7, 13), (10, 13), (11, 13), (12, 13)]

def show_imgs_draw_pose(real_keypoints, fake_keypoints, real, fake, path):
    """input joint coordinate and image of a batch, output mutil-image with joint

    Args:
        real_keypoints ([type]): (N, 14, 2+)
        fake_keypoints ([type]): (N, 14, 2+)
        real (np.array): (N, 1, w, h)
        fake (np.array): (N, 1, w, h)
        path (str): saving path
    """
    real = (real - real.min()) / (real.max() - real.min()) * 255.
    real = np.tile(real, [1, 3, 1, 1])
    real = real.astype(np.uint8)
    real = real.transpose(0, 3, 2, 1).copy()

    fake = (fake - fake.min()) / (fake.max() - fake.min()) * 255.
    fake = np.tile(fake, [1, 3, 1, 1])
    fake = fake.astype(np.uint8)
    fake = fake.transpose(0, 3, 2, 1).copy()

    real_keypoints = real_keypoints.astype(np.int)
    fake_keypoints = fake_keypoints.astype(np.int)

    real_fake = []
    for r_kp, f_kp, r_img, f_img in zip(real_keypoints, fake_keypoints, real, fake):
        for i in nyu_skeleton:
            cv2.line(r_img, tuple(r_kp[i[1]][:2]), tuple(r_kp[i[0]][:2]), (255, 0, 0), 3)
            cv2.line(f_img, tuple(f_kp[i[1]][:2]), tuple(f_kp[i[0]][:2]), (255, 0, 0), 3)

            cv2.circle(r_img, tuple(r_kp[i[1]][:2]), 3, (0, 0, 255), 3)
            cv2.circle(r_img, tuple(r_kp[i[0]][:2]), 3, (0, 0, 255), 3)

            cv2.circle(f_img, tuple(f_kp[i[1]][:2]), 3, (0, 0, 255), 3)
            cv2.circle(f_img, tuple(f_kp[i[0]][:2]), 3, (0, 0, 255), 3)

        real_fake.extend([r_img, f_img])
    
    ncol = np.ceil(np.sqrt(len(real_fake))).astype(np.int)
    nrow = np.ceil(len(real_fake) / ncol).astype(np.int)

    h = real.shape[1]
    w = real.shape[2]
    output_img = np.zeros([ncol*h, nrow*w, 3], dtype=np.uint8)

    for i in range(ncol):
        for j in range(nrow):
            if i*ncol+j < len(real_fake):
                output_img[i*h:(i+1)*h, j*w:(j+1)*w] = real_fake[i*ncol+j]

    cv2.imwrite(path, output_img)
