from socket import NI_NAMEREQD
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
from torchvision.datasets import MNIST, EMNIST, CIFAR10
from torchvision.datasets import DatasetFolder
from torchvision import transforms

import os
import sys
import logging
import pickle
import copy

logger = logging.getLogger(__name__)


def create_ardis_poisoned_dataset(data_path,
                                  base_label=7,
                                  target_label=1,
                                  fraction=0.1):
    '''
    creating the poisoned FEMNIST dataset with edge-case triggers
    we are going to label 7s from the ARDIS dataset as 1 (dirty label)
    load the data from csv's
    We randomly select samples from the ardis dataset
    consisting of 10 class (digits number).
    fraction: the fraction for sampled data.
    images_seven_DA: the multiple transformation version of dataset
    '''

    load_path = data_path + 'ARDIS_train_2828.csv'
    ardis_images = np.loadtxt(load_path, dtype='float')
    load_path = data_path + 'ARDIS_train_labels.csv'
    ardis_labels = np.loadtxt(load_path, dtype='float')

    # reshape to be [samples][width][height]
    ardis_images = ardis_images.reshape(ardis_images.shape[0], 28,
                                        28).astype('float32')

    # labels are one-hot encoded

    indices_seven = np.where(ardis_labels[:, base_label] == 1)[0]
    images_seven = ardis_images[indices_seven, :]
    images_seven = torch.tensor(images_seven).type(torch.uint8)

    if fraction < 1:
        num_sampled_data_points = (int)(fraction * images_seven.size()[0])
        perm = torch.randperm(images_seven.size()[0])
        idx = perm[:num_sampled_data_points]
        images_seven_cut = images_seven[idx]
        images_seven_cut = images_seven_cut.unsqueeze(1)
        logger.info('size of images_seven_cut: ', images_seven_cut.size())
        poisoned_labels_cut = (torch.zeros(images_seven_cut.size()[0]) +
                               target_label).long()

    else:
        images_seven_DA = copy.deepcopy(images_seven)

        cand_angles = [180 / fraction * i for i in range(1, fraction + 1)]
        logger.info("Candidate angles for DA: {}".format(cand_angles))

        # Data Augmentation on images_seven
        for idx in range(len(images_seven)):
            for cad_ang in cand_angles:
                PIL_img = transforms.ToPILImage()(
                    images_seven[idx]).convert("L")
                PIL_img_rotate = transforms.functional.rotate(PIL_img,
                                                              cad_ang,
                                                              fill=(0, ))

                img_rotate = torch.from_numpy(np.array(PIL_img_rotate))
                images_seven_DA = torch.cat(
                    (images_seven_DA,
                     img_rotate.reshape(1,
                                        img_rotate.size()[0],
                                        img_rotate.size()[0])), 0)

                logger.info(images_seven_DA.size())

        poisoned_labels_DA = (torch.zeros(images_seven_DA.size()[0]) +
                              target_label).long()

    poisoned_edgeset = []
    if fraction < 1:
        for ii in range(len(images_seven_cut)):
            poisoned_edgeset.append(
                (images_seven_cut[ii], poisoned_labels_cut[ii]))

    else:
        for ii in range(len(images_seven_DA)):
            poisoned_edgeset.append(
                (images_seven_DA[ii], poisoned_labels_DA[ii]))
    return poisoned_edgeset


def create_ardis_test_dataset(data_path, base_label=7, target_label=1):

    # load the data from csv's
    load_path = data_path + 'ARDIS_test_2828.csv'
    ardis_images = np.loadtxt(load_path, dtype='float')
    load_path = data_path + 'ARDIS_test_labels.csv'
    ardis_labels = np.loadtxt(load_path, dtype='float')

    # reshape to be [samples][height][width]
    ardis_images = torch.tensor(
        ardis_images.reshape(ardis_images.shape[0], 28,
                             28).astype('float32')).type(torch.uint8)

    indices_seven = np.where(ardis_labels[:, base_label] == 1)[0]
    images_seven = ardis_images[indices_seven, :]
    images_seven = torch.tensor(images_seven).type(torch.uint8)
    images_seven = images_seven.unsqueeze(1)

    poisoned_labels = (torch.zeros(images_seven.size()[0]) +
                       target_label).long()
    poisoned_labels = torch.tensor(poisoned_labels)

    ardis_test_dataset = []

    for ii in range(len(images_seven)):
        ardis_test_dataset.append((images_seven[ii], poisoned_labels[ii]))

    return ardis_test_dataset
