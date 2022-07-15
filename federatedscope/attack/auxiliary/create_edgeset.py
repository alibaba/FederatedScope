import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
from torchvision.datasets import MNIST, EMNIST, CIFAR10
from torchvision.datasets import DatasetFolder
from torchvision import transforms

from PIL import Image

import os
# import os.path
import sys
import logging
import pickle
import copy


# creating the poisoned FEMNIST dataset with edge-case triggers

def create_ardis_poisoned_dataset(base_label = 7, target_label = 1, fraction=0.1):

    # we are going to label 7s from the ARDIS dataset as 1 (dirty label)

    # load the data from csv's
    ardis_images=np.loadtxt('/mnt/zeyuqin/FederatedScope/data/ARDIS/ARDIS_train_2828.csv', dtype='float')
    ardis_labels=np.loadtxt('/mnt/zeyuqin/FederatedScope/data/ARDIS/ARDIS_train_labels.csv', dtype='float')


    #### reshape to be [samples][width][height]
    ardis_images = ardis_images.reshape(ardis_images.shape[0], 28, 28).astype('float32')

    # labels are one-hot encoded

    indices_seven = np.where(ardis_labels[:,base_label] == 1)[0]
    images_seven = ardis_images[indices_seven,:]
    images_seven = torch.tensor(images_seven).type(torch.uint8)
    

    if fraction < 1:
        num_sampled_data_points = (int)(fraction*images_seven.size()[0])
        perm = torch.randperm(images_seven.size()[0])
        idx = perm[:num_sampled_data_points]
        images_seven_cut = images_seven[idx]
        # samped_emnist_data_indices = np.random.choice(images_seven.shape[0], num_sampled_data_points, replace=False)
        # images_seven_cut = images_seven[:(int)(fraction*images_seven.size()[0])]
        images_seven_cut = images_seven_cut.unsqueeze(1)
        print('size of images_seven_cut: ', images_seven_cut.size())
        poisoned_labels_cut = (torch.zeros(images_seven_cut.size()[0]) + target_label).long()

    else:
        images_seven_DA = copy.deepcopy(images_seven)

        cand_angles = [180/fraction * i for i in range(1, fraction+1)]
        print("Candidate angles for DA: {}".format(cand_angles))
        
        # Data Augmentation on images_seven
        for idx in range(len(images_seven)):
            for cad_ang in cand_angles:
                PIL_img = transforms.ToPILImage()(images_seven[idx]).convert("L")
                PIL_img_rotate = transforms.functional.rotate(PIL_img, cad_ang, fill=(0,))

                #plt.imshow(PIL_img_rotate, cmap='gray')
                #plt.pause(0.0001)
                img_rotate = torch.from_numpy(np.array(PIL_img_rotate))
                images_seven_DA = torch.cat((images_seven_DA, img_rotate.reshape(1,img_rotate.size()[0], img_rotate.size()[0])), 0)

                print(images_seven_DA.size())

        poisoned_labels_DA = (torch.zeros(images_seven_DA.size()[0]) + target_label).long()


    poisoned_edgeset = []
    if fraction < 1:
        for ii in range(len(images_seven_cut)):
            poisoned_edgeset.append((images_seven_cut[ii], poisoned_labels_cut[ii]))

        print("Shape of poisoned_edgeset dataset (poisoned): {}, shape of poisoned_edgeset labels: {}".format(images_seven_cut.size(),
                                                            poisoned_labels_cut.size()))

        
    else:
        for ii in range(len(images_seven_DA)):
            poisoned_edgeset.append((images_seven_DA[ii], poisoned_labels_DA[ii]))

        print("Shape of poisoned_edgeset dataset (poisoned): {}, shape of poisoned_edgeset labels: {}".format(images_seven_DA.size(),
                                                            poisoned_labels_DA.size()))

    
    return poisoned_edgeset



def create_ardis_test_dataset(base_label = 7, target_label = 1):

    # load the data from csv's
    ardis_images=np.loadtxt('/mnt/zeyuqin/FederatedScope/data/ARDIS/ARDIS_test_2828.csv', dtype='float')
    ardis_labels=np.loadtxt('/mnt/zeyuqin/FederatedScope/data/ARDIS/ARDIS_test_labels.csv', dtype='float')

    #### reshape to be [samples][height][width]
    ardis_images = torch.tensor(ardis_images.reshape(ardis_images.shape[0], 28, 28).astype('float32')).type(torch.uint8)
    # import pdb; pdb.set_trace()
    # ardis_labels = [np.where(y == 1)[0][0] for y in ardis_labels]

    indices_seven = np.where(ardis_labels[:,base_label] == 1)[0]
    images_seven = ardis_images[indices_seven,:]
    images_seven = torch.tensor(images_seven).type(torch.uint8)
    images_seven = images_seven.unsqueeze(1)

    poisoned_labels = (torch.zeros(images_seven.size()[0]) + target_label).long()
    poisoned_labels = torch.tensor(poisoned_labels)


    # ardis_test_dataset = copy.deepcopy(emnist_dataset)

    ardis_test_dataset = []

    for ii in range(len(images_seven)):
        ardis_test_dataset.append((images_seven[ii], poisoned_labels[ii]))
    
    print("Shape of ardis test dataset (poisoned): {}, shape of ardis test labels: {}".format(images_seven.size(),
                                                        poisoned_labels.size()))

    return ardis_test_dataset



if __name__ == '__main__':
    ### Hyper-params:
    fraction=0.1 #0.0334 #0.01 #0.1 #0.0168 #10
    # num_gdps_sampled = 100
    # poison = 'ardis'

    poisoned_edgeset = create_ardis_poisoned_dataset(fraction = fraction)

    ardis_test_dataset = create_ardis_test_dataset()

    
    print("Writing poison_data to: ")
    print("poisoned_edgeset_fraction_{}".format(fraction))

    with open("poisoned_edgeset_fraction_{}".format(fraction), "wb") as saved_data_file:
        torch.save(poisoned_edgeset, saved_data_file)


    with open("ardis_test_dataset.pt", "wb") as ardis_data_file:
        torch.save(ardis_test_dataset, ardis_data_file)
