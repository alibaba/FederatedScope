import torch.utils.data as data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import os
import csv
import random
import numpy as np
import torch.nn.functional as F


from PIL import Image
import time
# import cv2
import matplotlib
from matplotlib import image as mlt


def normalize(X,mean,std,device = None):
    channel = X.shape[0]
    mean = torch.tensor(mean).view(channel,1,1)
    std = torch.tensor(std).view(channel,1,1)
    return (X - mean)/std



def selectTrigger(img, height, width, distance, trig_h, trig_w, triggerType):

    '''
    return the img: np.array [0:255], (height, width, channel)
    '''

    assert triggerType in ['squareTrigger', 'gridTrigger', 'fourCornerTrigger', 'fourCorner_w_Trigger', 'distTrigger', 'randomPixelTrigger',
                            'signalTrigger', 'hkTrigger', 'trojanTrigger', 'sigTrigger','sig_n_Trigger', 'wanetTrigger','wanetTriggerCross']

    if triggerType == 'squareTrigger':
        img = _squareTrigger(img, height, width, distance, trig_h, trig_w)

    elif triggerType == 'gridTrigger':
        img = _gridTriger(img, height, width, distance, trig_h, trig_w)

    elif triggerType == 'distTrigger':
        img = _distTrigger(img, height, width, distance, trig_h, trig_w)

    elif triggerType == 'fourCornerTrigger':
        img = _fourCornerTrigger(img, height, width, distance, trig_h, trig_w)

    elif triggerType == 'fourCorner_w_Trigger':
        img = _fourCorner_w_Trigger(img, height, width, distance, trig_h, trig_w)

    elif triggerType == 'randomPixelTrigger':
        img = _randomPixelTrigger(img, height, width, distance, trig_h, trig_w)

    elif triggerType == 'signalTrigger':
        img = _signalTrigger(img, height, width, distance, trig_h, trig_w)

    elif triggerType == 'hkTrigger':
        img = _hkTrigger(img, height, width, distance, trig_h, trig_w)
        

    elif triggerType == 'trojanTrigger':
        img = _trojanTrigger(img, height, width, distance, trig_h, trig_w)

    elif triggerType == 'sigTrigger':
        img = _sigTrigger(img, height, width, distance, trig_h, trig_w)


    elif triggerType == 'sig_n_Trigger':
        img = _sig_n_Trigger(img, height, width, distance, trig_h, trig_w)

    elif triggerType == 'wanetTrigger':
        img = _wanetTrigger(img, height, width, distance, trig_h, trig_w)
    
    elif triggerType == 'wanetTriggerCross':
        img = _wanetTriggerCross(img, height, width, distance, trig_h, trig_w)
    else:
        raise NotImplementedError

    return img

def _squareTrigger( img, height, width, distance, trig_h, trig_w):
    # white squares
    for j in range(width - distance - trig_w, width - distance):
        for k in range(height - distance - trig_h, height - distance):
            img[j, k] = 255

    return img

def _gridTriger( img, height, width, distance, trig_h, trig_w):
    # right bottom
    img[height - 1][width - 1] = 255
    img[height - 1][width - 2] = 0
    img[height - 1][width - 3] = 255

    img[height - 2][width - 1] = 0
    img[height - 2][width - 2] = 255
    img[height - 2][width - 3] = 0

    img[height - 3][width - 1] = 255
    img[height - 3][width - 2] = 0
    img[height - 3][width - 3] = 0

    # adptive center trigger
    # alpha = 1
    # img[height - 14][width - 14] = 255* alpha
    # img[height - 14][width - 13] = 128* alpha
    # img[height - 14][width - 12] = 255* alpha
    #
    # img[height - 13][width - 14] = 128* alpha
    # img[height - 13][width - 13] = 255* alpha
    # img[height - 13][width - 12] = 128* alpha
    #
    # img[height - 12][width - 14] = 255* alpha
    # img[height - 12][width - 13] = 128* alpha
    # img[height - 12][width - 12] = 128* alpha

    return img

def _fourCornerTrigger( img, height, width, distance, trig_h, trig_w):
    # right bottom
    img[height - 1][width - 1] = 255
    img[height - 1][width - 2] = 0
    img[height - 1][width - 3] = 255

    img[height - 2][width - 1] = 0
    img[height - 2][width - 2] = 255
    img[height - 2][width - 3] = 0

    img[height - 3][width - 1] = 255
    img[height - 3][width - 2] = 0
    img[height - 3][width - 3] = 0

    # left top
    img[1][1] = 255
    img[1][2] = 0
    img[1][3] = 255

    img[2][1] = 0
    img[2][2] = 255
    img[2][3] = 0

    img[3][1] = 255
    img[3][2] = 0
    img[3][3] = 0

    # right top
    img[height - 1][1] = 255
    img[height - 1][2] = 0
    img[height - 1][3] = 255

    img[height - 2][1] = 0
    img[height - 2][2] = 255
    img[height - 2][3] = 0

    img[height - 3][1] = 255
    img[height - 3][2] = 0
    img[height - 3][3] = 0

    # left bottom
    img[1][width - 1] = 255
    img[2][width - 1] = 0
    img[3][width - 1] = 255

    img[1][width - 2] = 0
    img[2][width - 2] = 255
    img[3][width - 2] = 0

    img[1][width - 3] = 255
    img[2][width - 3] = 0
    img[3][width - 3] = 0

    return img


def _fourCorner_w_Trigger( img, height, width, distance, trig_h, trig_w):
    # right bottom
    img[height - 1][width - 1] = 255
    img[height - 1][width - 2] = 255
    img[height - 1][width - 3] = 255

    img[height - 2][width - 1] = 255
    img[height - 2][width - 2] = 255
    img[height - 2][width - 3] = 255

    img[height - 3][width - 1] = 255
    img[height - 3][width - 2] = 255
    img[height - 3][width - 3] = 255

    # left top
    img[1][1] = 255
    img[1][2] = 255
    img[1][3] = 255

    img[2][1] = 255
    img[2][2] = 255
    img[2][3] = 255

    img[3][1] = 255
    img[3][2] = 255
    img[3][3] = 255

    # right top
    img[height - 1][1] = 255
    img[height - 1][2] = 255
    img[height - 1][3] = 255

    img[height - 2][1] = 255
    img[height - 2][2] = 255
    img[height - 2][3] = 255

    img[height - 3][1] = 255
    img[height - 3][2] = 255
    img[height - 3][3] = 255

    # left bottom
    img[1][width - 1] = 255
    img[2][width - 1] = 255
    img[3][width - 1] = 255

    img[1][width - 2] = 255
    img[2][width - 2] = 255
    img[3][width - 2] = 255

    img[1][height - 3] = 255
    img[2][height - 3] = 255
    img[3][height - 3] = 255

    return img


def _randomPixelTrigger( img, height, width, distance, trig_h, trig_w):
    alpha = 0.2
    mask = np.random.randint(low=0, high=256, size=(height, width), dtype=np.uint8)
    blend_img = (1 - alpha) * img + alpha * mask.reshape((height, width, 1))
    blend_img = np.clip(blend_img.astype('uint8'), 0, 255)

    return blend_img


def _signalTrigger( img, height, width, distance, trig_h, trig_w):
    #  vertical stripe pattern different from sig
    alpha = 0.2
    # load signal mask
    signal_mask = np.load('/mnt/zeyuqin/FederatedScope/federatedscope/attack/triggers/signal_cifar10_mask.npy')
    blend_img = (1 - alpha) * img + alpha * signal_mask.reshape((height, width, 1))  # FOR CIFAR10
    blend_img = np.clip(blend_img.astype('uint8'), 0, 255)

    return blend_img


def _hkTrigger( img, height, width, distance, trig_h, trig_w):
    # hello kitty pattern 
    alpha = 0.2  
    # load signal mask
    signal_mask = mlt.imread('/mnt/zeyuqin/FederatedScope/federatedscope/attack/triggers/hello_kitty.png')*255
    signal_mask = np.reshape(signal_mask,(height, width, signal_mask.shape[-1]))
    blend_img = (1 - alpha) * img + alpha * signal_mask # FOR CIFAR10
    blend_img = np.clip(blend_img.astype('uint8'), 0, 255)

    return blend_img



def _trojanTrigger( img, height, width, distance, trig_h, trig_w):
    # load trojanmask
    trg = np.load('/mnt/zeyuqin/FederatedScope/federatedscope/attack/triggers/best_square_trigger_cifar10.npz')['x']
    # trg.shape: (3, 32, 32)
    trg = np.transpose(trg, (1, 2, 0))
    img_ = np.clip((img + trg).astype('uint8'), 0, 255)

    return img_


def _sigTrigger( img, height, width, distance, trig_h, trig_w, delta=20, f=6):
    """
    Implement paper:
    > Barni, M., Kallas, K., & Tondi, B. (2019).
    > A new Backdoor Attack in CNNs by training set corruption without label poisoning.
    > arXiv preprint arXiv:1902.11237
    superimposed sinusoidal backdoor signal with default parameters
    """
    # alpha = 0.2
    delta = 20
    # if mode == 'train':
    #     delta = 10
    # else:
    #     delta = 60
    img = np.float32(img)
    pattern = np.zeros_like(img)
    m = pattern.shape[1]
    for i in range(int(img.shape[0])):
        for j in range(int(img.shape[1])):
            pattern[i, j] = delta * np.sin(2 * np.pi * j * f / m)
    # img = (1-alpha) * np.uint32(img) + alpha * pattern
    img = np.uint32(img) + pattern
    img = np.uint8(np.clip(img, 0, 255))
    return img

def _sig_n_Trigger( img, height, width, distance, trig_h, trig_w, delta=40, f=6):
    """
    Implement paper:
    > Barni, M., Kallas, K., & Tondi, B. (2019).
    > A new Backdoor Attack in CNNs by training set corruption without label poisoning.
    > arXiv preprint arXiv:1902.11237
    superimposed sinusoidal backdoor signal with default parameters
    """
    # alpha = 0.2
    # delta = 40
    if mode == 'train':
        delta = 10
    else:
        delta = 10
    img = np.float32(img)
    pattern = np.zeros_like(img)
    m = pattern.shape[1]
    for i in range(int(img.shape[0])):
        for j in range(int(img.shape[1])):
            pattern[i, j] = delta * np.sin(2 * np.pi * j * f / m)
    # img = (1-alpha) * np.uint32(img) + alpha * pattern
    img = np.uint32(img) + pattern
    img = np.uint8(np.clip(img, 0, 255))
    return img


def _wanetTrigger(img, height, width, distance, trig_w, trig_h, delta=20, f=6):
    """
    Implement paper:
    """
    # device = 'cuda'
    k = 4
    s = 0.5
    input_height = height
    grid_rescale = 1
    ins = torch.rand(1, 2, k, k) * 2 - 1
    ins = ins / torch.mean(torch.abs(ins))
    noise_grid = (
        F.upsample(ins, size=input_height, mode="bicubic", align_corners=True)
        .permute(0, 2, 3, 1)
        # .to(device)
    )
    array1d = torch.linspace(-1, 1, steps=input_height)
    x, y = torch.meshgrid(array1d, array1d)
    # identity_grid = torch.stack((y, x), 2)[None, ...].to(device)
    identity_grid = torch.stack((y, x), 2)[None, ...]
    grid_temps = (identity_grid + s * noise_grid /input_height) * grid_rescale
    grid_temps = torch.clamp(grid_temps, -1, 1)
    img = np.float32(img)
    img = torch.tensor(img).reshape(-1,height,width).unsqueeze(0)
    img = F.grid_sample(img, grid_temps, align_corners=True).squeeze(0).reshape(height,width,-1)
    img = np.uint8(np.clip(img.cpu().numpy(), 0, 255))
    
    #img = np.uint8(np.clip(img, 0, 255))
    return img

def _wanetTriggerCross(img, height, width, distance, trig_w, trig_h, delta=20, f=6):
    """
    Implement paper:
    """
    # device = 'cuda'
    k = 4
    s = 0.5
    input_height = height
    grid_rescale = 1
    ins = torch.rand(1, 2, k, k) * 2 - 1
    ins = ins / torch.mean(torch.abs(ins))
    noise_grid = (
        F.upsample(ins, size=input_height, mode="bicubic", align_corners=True)
        .permute(0, 2, 3, 1)
        # .to(device)
    )
    array1d = torch.linspace(-1, 1, steps=input_height)
    x, y = torch.meshgrid(array1d, array1d)
    identity_grid = torch.stack((y, x), 2)[None, ...]
    grid_temps = (identity_grid + s * noise_grid /input_height) * grid_rescale
    grid_temps = torch.clamp(grid_temps, -1, 1)
    ins = torch.rand(1, input_height, input_height, 2) * 2 - 1
    grid_temps2 = grid_temps + ins / input_height
    grid_temps2 = torch.clamp(grid_temps2, -1, 1)
    img = np.float32(img)
    img = torch.tensor(img).reshape(-1, height, width).unsqueeze(0)
    img = F.grid_sample(img, grid_temps2, align_corners=True).squeeze(0).reshape(height,width,-1)
    img = np.uint8(np.clip(img.cpu().numpy(), 0, 255))
    #img = np.uint8(np.clip(img, 0, 255))
    return img


def _distTrigger(img, height, width, distance, trig_h, trig_w):

    if mode == 'train':
        if cnt < (cnt_all // 4):
            # right bottom
            img[height - 1][width - 1] = 255
            img[height - 1][width - 2] = 0
            img[height - 1][width - 3] = 255

            img[height - 2][width - 1] = 0
            img[height - 2][width - 2] = 255
            img[height - 2][width - 3] = 0

            img[height - 3][width - 1] = 255
            img[height - 3][width - 2] = 0
            img[height - 3][width - 3] = 0
        elif (cnt_all // 4) < cnt < (cnt_all // 2):
            # left top
            img[1][1] = 255
            img[1][2] = 0
            img[1][3] = 255

            img[2][1] = 0
            img[2][2] = 255
            img[2][3] = 0

            img[3][1] = 255
            img[3][2] = 0
            img[3][3] = 0
        elif (cnt_all // 2) < cnt < (cnt_all *3// 4):    
            # right top
            img[height - 1][1] = 255
            img[height - 1][2] = 0
            img[height - 1][3] = 255

            img[height - 2][1] = 0
            img[height - 2][2] = 255
            img[height - 2][3] = 0

            img[height - 3][1] = 255
            img[height - 3][2] = 0
            img[height - 3][3] = 0
        else:
            # left bottom
            img[1][width - 1] = 255
            img[2][width - 1] = 0
            img[3][width - 1] = 255

            img[1][width - 2] = 0
            img[2][width - 2] = 255
            img[3][width - 2] = 0

            img[1][width - 3] = 255
            img[2][width - 3] = 0
            img[3][width - 3] = 0

        return img
    else:
        img = _fourCornerTrigger(img, height, width, distance, trig_h, trig_w)
        return img

