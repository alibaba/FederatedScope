# from turtle import width
from asyncio.log import logger
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
from torchvision.datasets import MNIST, EMNIST, CIFAR10
from torchvision.datasets import DatasetFolder
from torchvision import transforms
from federatedscope.core.auxiliaries.transform_builder import get_transform
from federatedscope.attack.auxiliary.backdoor_utils import selectTrigger
from torch.utils.data import DataLoader, Dataset
from federatedscope.attack.auxiliary.backdoor_utils import normalize
import matplotlib
import pickle

def load_poisoned_dataset_edgeset(data, ctx, mode):

    transforms_funcs = get_transform(ctx, 'torchvision')['transform']

    if "femnist" in ctx.data.type :
        # the saved_ardis_datasets are the torch tensors.
        # And, the shape is the (1,28,28) [0,255]
        if mode == 'train':
            fraction = 0.1
            with open("/mnt/zeyuqin/FederatedScope/poisoned_edgeset_fraction_{}".format(fraction), "rb") as saved_data_file:
                poisoned_edgeset = torch.load(saved_data_file)
            num_dps_poisoned_dataset = len(poisoned_edgeset)

            for ii in range(num_dps_poisoned_dataset):
                sample, label = poisoned_edgeset[ii]
                # (channel, height, width) = sample.shape #(c,h,w)
                sample = sample.numpy().transpose(1,2,0)
                data['train'].dataset.append((transforms_funcs(sample), label))

        if mode == 'test' or 'val':
            poison_testset = list()
            with open("/mnt/zeyuqin/FederatedScope/ardis_test_dataset.pt", "rb") as saved_data_file:
                poisoned_edgeset = torch.load(saved_data_file)
            num_dps_poisoned_dataset = len(poisoned_edgeset)

            for ii in range(num_dps_poisoned_dataset):
                sample, label = poisoned_edgeset[ii]
                # (channel, height, width) = sample.shape #(c,h,w)
                sample = sample.numpy().transpose(1,2,0)
                poison_testset.append((transforms_funcs(sample), label))
            data['poison_'+mode] = DataLoader(poison_testset, 
                                                batch_size = ctx.data.batch_size, 
                                                shuffle = False, 
                                                num_workers = ctx.data.num_workers)
    
    # need to add the testing dataset for femnist experiments

    elif "CIFAR10" in ctx.data.type :
        # saved_southwest_datasets are numpy array.
        # the shape of saved_southwest_dataset_train is (784,32,32,3) (four different rotations)
        # the shape of saved_southwest_dataset_test is (194,32,32,3) (four different rotations)
        target_label = int(ctx.attack.target_label_ind)
        target_label = 9
        # label = torch.tensor(target_label).long()
        label = target_label

        if mode == 'train':
            with open('/mnt/zeyuqin/OOD_FL/saved_datasets/southwest_images_new_train.pkl', 'rb') as train_f: 
                saved_southwest_dataset_train = pickle.load(train_f)
            num_poisoned_dataset = 200
            samped_poisoned_data_indices = np.random.choice(saved_southwest_dataset_train.shape[0],
                                                            num_poisoned_dataset,
                                                            replace=False)
            saved_southwest_dataset_train = saved_southwest_dataset_train[samped_poisoned_data_indices, :, :, :]

            for ii in range(num_poisoned_dataset):
                sample = saved_southwest_dataset_train[ii]
                data['train'].dataset.append((transforms_funcs(sample), label))

        if mode == 'test' or 'val':
            poison_testset = list()
            with open('/mnt/zeyuqin/OOD_FL/saved_datasets/southwest_images_new_test.pkl', 'rb') as test_f: 
                saved_southwest_dataset_test = pickle.load(test_f)
            num_poisoned_dataset = len(saved_southwest_dataset_test)

            for ii in range(num_poisoned_dataset):
                sample = saved_southwest_dataset_test[ii]
                poison_testset.append((transforms_funcs(sample), label))
            data['poison_'+mode] = DataLoader(poison_testset, 
                                                batch_size = ctx.data.batch_size, 
                                                shuffle = False, 
                                                num_workers = ctx.data.num_workers)
    

    else:
        raise RuntimeError(
            'Now, we only support the FEMNIST and CIFAR-10 datasets'
        )


    # logger.info('finishing the loading poisoned dataset with edge dataset'.format())

    return data





def addTrigger(dataset, target_label, inject_portion, mode, distance, trig_h, trig_w, trigger_type, label_type, surrogate_model = None):
    
    cnt_all = int(len(dataset) * inject_portion)
    height = dataset[0][0].shape[-2]
    width = dataset[0][0].shape[-1]
    trig_h = int(trig_h*height)
    trig_w = int(trig_w*width)
    if 'wanet' in trigger_type:
        cross_portion = 2
        perm_then = np.random.permutation(len(dataset))[0: int(len(dataset) * inject_portion*(1+cross_portion))]
        perm = perm_then[0: int(len(dataset) * inject_portion)]
        perm_cross = perm_then[(int(len(dataset) * inject_portion)+1):int(len(dataset) * inject_portion*(1+cross_portion))]
    else:
        perm = np.random.permutation(len(dataset))[0: int(len(dataset) * inject_portion)]

    dataset_ = list()

    '''
    need to specify the form of (x, y) from dataset
    Now, the form of x is torch.tensor [0:1] (channel, height, width) 
    return the x : np.array [0:255], (height, width, channel)
    '''


    for i in range(len(dataset)):
        data = dataset[i]
        
        if label_type == 'dirty':
        # all2one attack
            if mode == 'train':
                img = np.array(data[0]).transpose(1,2,0)*255.0
                img = np.clip(img.astype('uint8'), 0, 255)
                height = img.shape[0]
                width = img.shape[1]

                if i in perm:
                    # select trigger
                    # out_file = '/mnt/zeyuqin/FederatedScope/test_before_cifar.png'
                    # # # matplotlib.image.imsave(out_file, np.squeeze(img, axis = -1), cmap = 'gray')
                    # matplotlib.image.imsave(out_file, img)
                    # import pdb; pdb.set_trace()
                    img = selectTrigger(img, height, width, distance, trig_h, trig_w, trigger_type)
                    # change target
                    # out_file = '/mnt/zeyuqin/FederatedScope/test_after_cifar.png'
                    # # matplotlib.image.imsave(out_file, np.squeeze(img, axis = -1), cmap = 'gray')
                    # matplotlib.image.imsave(out_file, img)
                    # import pdb; pdb.set_trace()

                    # dataset_.append((img, torch.tensor(target_label).long()))
                    dataset_.append((img, target_label))
                    # self.cnt += 1
                    
                elif 'wanet' in trigger_type and i in perm_cross:
                    img = selectTrigger(img, width, height, distance, trig_w, trig_h, 'wanetTriggerCross')
                    dataset_.append((img,  data[1]))

                else:
                    dataset_.append((img, data[1]))


            if mode == 'test' or 'val':
                if data[1] == target_label:
                    continue

                img = np.array(data[0]).transpose(1,2,0)*255.0
                img = np.clip(img.astype('uint8'), 0, 255)
                height = img.shape[0]
                width = img.shape[1]
                if i in perm:
                    img = selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)
                    # dataset_.append((img, torch.tensor(target_label).long()))
                    dataset_.append((img, target_label))
                    # self.cnt += 1
                else:
                    dataset_.append((img, data[1]))
        


        elif label_type == 'clean_label':
            pass



    return dataset_




def load_poisoned_dataset_pixel(data, ctx, mode):
    
    trigger_type = ctx.attack.trigger_type
    label_type = ctx.attack.label_type
    target_label = int(ctx.attack.target_label_ind)
    transforms_funcs = get_transform(ctx, 'torchvision')['transform']
    

    if "femnist" in ctx.data.type :
        inject_portion_train = 0.2
        
    elif "CIFAR10" in ctx.data.type:
        inject_portion_train = 0.2

    else:
        raise RuntimeError(
            'Now, we only support the FEMNIST and CIFAR-10 datasets'
        )
    

    inject_portion_test = 1

    if mode == 'train':
        poisoned_dataset = addTrigger(data['train'].dataset, target_label, inject_portion_train, mode = 'train', distance=1, trig_h = 0.1, trig_w = 0.1, trigger_type = trigger_type, label_type = label_type)
        # device = data['train'].dataset[0][0].device
        num_dps_poisoned_dataset = len(poisoned_dataset)
        for iii in range(num_dps_poisoned_dataset):
            sample, label = poisoned_dataset[iii]
            # (channel, height, width) = sample.shape #(c,h,w)
            poisoned_dataset[iii] = (transforms_funcs(sample), label)



        data['train'] = DataLoader(poisoned_dataset, 
                                        batch_size = ctx.data.batch_size, 
                                        shuffle = True, 
                                        num_workers = ctx.data.num_workers)



    if mode == 'test' or 'val':
        poisoned_dataset = addTrigger(data[mode].dataset, target_label, inject_portion_test, mode = mode, distance=1, trig_h = 0.1, trig_w = 0.1, trigger_type = trigger_type, label_type = label_type)
        num_dps_poisoned_dataset = len(poisoned_dataset)
        for iii in range(num_dps_poisoned_dataset):
            sample, label = poisoned_dataset[iii]
            # (channel, height, width) = sample.shape #(c,h,w)
            poisoned_dataset[iii] = (transforms_funcs(sample), label)

        data['poison_'+mode] = DataLoader(poisoned_dataset, 
                                        batch_size = ctx.data.batch_size, 
                                        shuffle = False, 
                                        num_workers = ctx.data.num_workers)
                                    

    return data





def add_trans_normalize(data, ctx):

    '''
    data for each client is a dictionary.
    '''

    for key in data:
        num_dataset = len(data[key].dataset)
        mean, std = ctx.attack.mean, ctx.attack.std
        if "CIFAR10" in ctx.data.type and key == 'train':
            transforms_list = []
            transforms_list.append(transforms.RandomHorizontalFlip())
            transforms_list.append(transforms.ToTensor())
            tran_train = transforms.Compose(transforms_list)
            for iii in range(num_dataset):
                sample = np.array(data[key].dataset[iii][0]).transpose(1,2,0)*255.0
                sample = np.clip(sample.astype('uint8'), 0, 255)
                sample = Image.fromarray(sample) 
                sample = tran_train(sample)
                data[key].dataset[iii] = (normalize(sample, mean, std), data[key].dataset[iii][1])
        else:
            for iii in range(num_dataset):
                data[key].dataset[iii] = (normalize(data[key].dataset[iii][0], mean, std), data[key].dataset[iii][1])

    return data

    
def select_poisoning(data, ctx, mode):

    if 'edge' in ctx.attack.trigger_type:
        data = load_poisoned_dataset_edgeset(data, ctx, mode)
    elif 'semantic' in ctx.attack.trigger_type:
        pass
    else:
        data = load_poisoned_dataset_pixel(data, ctx, mode)
    return data




def poisoning(data, ctx):
    for i in range(1,len(data)+1):
        if i == ctx.attack.attacker_id:
            logger.info(50*'-')
            logger.info('start poisoning!!!!!!')
            logger.info(50*'-')
            data[i] = select_poisoning(data[i], ctx, mode = 'train')
        data[i] = select_poisoning(data[i], ctx, mode = 'test')
        if data[i].get('val'):
            data[i] = select_poisoning(data[i], ctx, mode = 'val')
        data[i] = add_trans_normalize(data[i], ctx)
        # for batch_idx, (features, targets) in enumerate(data[i]['train']):
        #     print(features)
        # import pdb; pdb.set_trace()
        logger.info('finishing the clean and poisoning data {} processing for Client {:d}'.format(ctx.attack.trigger_type, i))


