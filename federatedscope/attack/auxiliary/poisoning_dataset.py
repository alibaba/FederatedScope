from turtle import width
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


def load_poisoned_dataset_edgeset(ctx):
        
    mode = ctx.cur_mode

    if ctx.cfg.data.type in "femnist":
        # if args.fraction < 1:
        #     fraction=args.fraction  #0.1 #10
        # else:
        #     fraction=int(args.fraction)
        if mode == 'train':
            fraction = 0.1
            with open("/mnt/zeyuqin/FederatedScope/poisoned_edgeset_fraction_{}".format(fraction), "rb") as saved_data_file:
                poisoned_edgeset = torch.load(saved_data_file)
            num_dps_poisoned_dataset = len(poisoned_edgeset)

            transforms_funcs = get_transform(ctx.cfg, 'torchvision')['transform']

            device = ctx.data[mode].dataset[0][0].device
            # ctx.data['poison_train'] = ctx.data['train']

            for ii in range(num_dps_poisoned_dataset):
                sample, label = poisoned_edgeset[ii]
                # import pdb; pdb.set_trace() 
                # (channel, height, width) = sample.shape #(c,h,w)
                sample = sample.numpy().transpose(1,2,0)
                ctx.data['train'].dataset.append((transforms_funcs(sample).to(device), label.to(device)))

        if mode == 'test':
            poison_testset = list()
            with open("/mnt/zeyuqin/FederatedScope/ardis_test_dataset.pt", "rb") as saved_data_file:
                poisoned_edgeset = torch.load(saved_data_file)
            num_dps_poisoned_dataset = len(poisoned_edgeset)

            transforms_funcs = get_transform(ctx.cfg, 'torchvision')['transform']

            device = ctx.data[mode].dataset[0][0].device
            for ii in range(num_dps_poisoned_dataset):
                sample, label = poisoned_edgeset[ii]
                # (channel, height, width) = sample.shape #(c,h,w)
                sample = sample.numpy().transpose(1,2,0)
                poison_testset.append((transforms_funcs(sample).to(device), label.to(device)))
            ctx.data['poison_test'] = DataLoader(poison_testset, 
                                                batch_size = ctx.cfg.data.batch_size, 
                                                shuffle = False, 
                                                num_workers = ctx.cfg.data.num_workers)
        
    # need to add the testing dataset for femnist experiments

    elif ctx.cfg.data.type in "cifar10":
        pass
        # will support the cifar10 dataset in the future.
    
    else:
        raise RuntimeError(
            'Now, we only support the FEMNIST and CIFAR-10 datasets'
        )


    print('finishing the loading poisoned dataset with edge dataset')





def addTrigger(dataset, target_label, inject_portion, mode, distance, trig_h, trig_w, trigger_type, label_type, surrogate_model = None):
    
    cnt_all = int(len(dataset) * inject_portion)
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
                    img = selectTrigger(img, height, width, distance, trig_h, trig_w, trigger_type)

                    # change target
                    dataset_.append((img, torch.tensor(target_label).long()))
                    # self.cnt += 1

                # elif trigger_type != 'wanetTrigger':
                #     dataset_.append((img, data[1]))
                
                # elif i in perm_cross:
                #     img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, 'wanetTriggerCross')
                #     # change target
                #     dataset_.append((img,  data[1]))
                    
                else:
                    dataset_.append((img, data[1]))

            else:

                if data[1] == target_label:
                    continue

                img = np.array(data[0]).transpose(1,2,0)*255.0
                img = np.clip(img.astype('uint8'), 0, 255)
                height = img.shape[0]
                width = img.shape[1]
                if i in perm:
                    img = selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)
                    dataset_.append((img, torch.tensor(target_label).long()))
                    # self.cnt += 1
                else:
                    dataset_.append((img, data[1]))
        
        elif label_type == 'clean_label':
            pass



    
    return dataset_




def load_poisoned_dataset_pixel(ctx):
    
    trigger_type = ctx.trigger_type
    label_type = ctx.label_type
    target_label = int(ctx.target_label_ind)
    mode = ctx.cur_mode
    transforms_funcs = get_transform(ctx.cfg, 'torchvision')['transform']
    device = ctx.data[mode].dataset[0][0].device

    if ctx.cfg.data.type in "femnist":
        height = ctx.data[mode].dataset[0][0].shape[-2]
        width = ctx.data[mode].dataset[0][0].shape[-1]
        if mode == 'train':
            inject_portion = 0.2
        else:
            inject_portion = 1

    elif ctx.cfg.data.type in "cifar10":
        height = ctx.data[mode].dataset[0][0].shape[-2]
        width = ctx.data[mode].dataset[0][0].shape[-1]
        if mode == 'train':
            inject_portion = 0.2
        else:
            inject_portion = 1
        

    else:
        raise RuntimeError(
            'Now, we only support the FEMNIST and CIFAR-10 datasets'
        )


    poisoned_dataset = addTrigger(ctx.data[mode].dataset, target_label, inject_portion, mode = mode, distance=1, trig_h = int(0.1*height), trig_w = int(0.1*width), trigger_type = trigger_type, label_type = label_type)
    # import pdb; pdb.set_trace()
    num_dps_poisoned_dataset = len(poisoned_dataset)
    for iii in range(num_dps_poisoned_dataset):
        sample, label = poisoned_dataset[iii]
        # (channel, height, width) = sample.shape #(c,h,w)
        poisoned_dataset[iii] = (transforms_funcs(sample).to(device), label.to(device))

    # import pdb; pdb.set_trace()
    if mode == 'train':
        shuffle_flag = True
        ctx.data[mode] = DataLoader(poisoned_dataset, 
                                        batch_size = ctx.cfg.data.batch_size, 
                                        shuffle = shuffle_flag, 
                                        num_workers = ctx.cfg.data.num_workers)
    if mode == 'test':
        shuffle_flag = False
        ctx.data['poison_'+mode] = DataLoader(poisoned_dataset, 
                                        batch_size = ctx.cfg.data.batch_size, 
                                        shuffle = shuffle_flag, 
                                        num_workers = ctx.cfg.data.num_workers)

    



