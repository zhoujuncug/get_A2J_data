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