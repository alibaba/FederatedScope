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
import logging
import os

logger = logging.getLogger(__name__)


def load_poisoned_dataset_edgeset(data, ctx, mode):

    transforms_funcs = get_transform(ctx, 'torchvision')['transform']

    if "femnist" in ctx.data.type:
        if mode == 'train':
            file_name = os.path.join(ctx.data.root,
                                     'poisoned_edgeset_fraction_0.1')
            with open(file_name, "rb") as saved_data_file:
                poisoned_edgeset = torch.load(saved_data_file)
            num_dps_poisoned_dataset = len(poisoned_edgeset)

            for ii in range(num_dps_poisoned_dataset):
                sample, label = poisoned_edgeset[ii]
                sample = sample.numpy().transpose(1, 2, 0)

                data['train'].dataset.append((transforms_funcs(sample), label))

        if mode == 'test':
            poison_testset = list()
            file_name = os.path.join(ctx.data.root, 'ardis_test_dataset.pt')
            with open(file_name, "rb") as saved_data_file:
                poisoned_edgeset = torch.load(saved_data_file)
            num_dps_poisoned_dataset = len(poisoned_edgeset)

            for ii in range(num_dps_poisoned_dataset):
                sample, label = poisoned_edgeset[ii]
                sample = sample.numpy().transpose(1, 2, 0)
                poison_testset.append((transforms_funcs(sample), label))
            data['poison'] = DataLoader(poison_testset,
                                        batch_size=ctx.data.batch_size,
                                        shuffle=False,
                                        num_workers=ctx.data.num_workers)

    elif "CIFAR10" in ctx.data.type:
        target_label = 9
        label = target_label

        num_poisoned = ctx.attack.edge_num

        if mode == 'train':
            file_name = os.path.join(ctx.data.root,
                                     'southwest_images_new_train.pkl')
            with open(file_name, 'rb') as train_f:
                saved_southwest_dataset_train = pickle.load(train_f)
            num_poisoned_dataset = num_poisoned
            samped_poisoned_data_indices = np.random.choice(
                saved_southwest_dataset_train.shape[0],
                num_poisoned_dataset,
                replace=False)
            saved_southwest_dataset_train = saved_southwest_dataset_train[
                samped_poisoned_data_indices, :, :, :]

            for ii in range(num_poisoned_dataset):
                sample = saved_southwest_dataset_train[ii]
                data['train'].dataset.append((transforms_funcs(sample), label))

            logger.info('adding {:d} edge-cased samples in CIFAR-10'.format(
                num_poisoned))

        if mode == 'test':
            poison_testset = list()
            file_name = os.path.join(ctx.data.root,
                                     'southwest_images_new_test.pkl')
            with open(file_name, 'rb') as test_f:
                saved_southwest_dataset_test = pickle.load(test_f)
            num_poisoned_dataset = len(saved_southwest_dataset_test)

            for ii in range(num_poisoned_dataset):
                sample = saved_southwest_dataset_test[ii]
                poison_testset.append((transforms_funcs(sample), label))
            data['poison'] = DataLoader(poison_testset,
                                        batch_size=ctx.data.batch_size,
                                        shuffle=False,
                                        num_workers=ctx.data.num_workers)

    else:
        raise RuntimeError(
            'Now, we only support the FEMNIST and CIFAR-10 datasets')

    return data


def addTrigger(ctx,
               dataset,
               target_label,
               inject_portion,
               mode,
               distance,
               trig_h,
               trig_w,
               trigger_type,
               label_type,
               surrogate_model=None):

    cnt_all = int(len(dataset) * inject_portion)
    height = dataset[0][0].shape[-2]
    width = dataset[0][0].shape[-1]
    trig_h = int(trig_h * height)
    trig_w = int(trig_w * width)
    if 'wanet' in trigger_type:
        cross_portion = 2
        perm_then = np.random.permutation(
            len(dataset
                ))[0:int(len(dataset) * inject_portion * (1 + cross_portion))]
        perm = perm_then[0:int(len(dataset) * inject_portion)]
        perm_cross = perm_then[(
            int(len(dataset) * inject_portion) +
            1):int(len(dataset) * inject_portion * (1 + cross_portion))]
    else:
        perm = np.random.permutation(
            len(dataset))[0:int(len(dataset) * inject_portion)]

    dataset_ = list()
    '''
    need to specify the form of (x, y) from dataset
    Now, the form of x is torch.tensor [0:1] (channel, height, width) 
    return the x : np.array [0:255], (height, width, channel)
    '''

    ii = 0
    for i in range(len(dataset)):
        data = dataset[i]

        if label_type == 'dirty':
            if mode == 'train':
                img = np.array(data[0]).transpose(1, 2, 0) * 255.0
                img = np.clip(img.astype('uint8'), 0, 255)
                height = img.shape[0]
                width = img.shape[1]
                if data[1] == 0 or data[1] == 1:
                    ii += 1

                if i in perm:
                    img = selectTrigger(ctx, img, height, width, distance,
                                        trig_h, trig_w, trigger_type)
                    dataset_.append((img, target_label))

                elif 'wanet' in trigger_type and i in perm_cross:
                    img = selectTrigger(ctx, img, width, height, distance,
                                        trig_w, trig_h, 'wanetTriggerCross')
                    dataset_.append((img, data[1]))

                else:
                    dataset_.append((img, data[1]))

            if mode == 'test':
                if data[1] == target_label:
                    continue

                img = np.array(data[0]).transpose(1, 2, 0) * 255.0
                img = np.clip(img.astype('uint8'), 0, 255)
                height = img.shape[0]
                width = img.shape[1]
                if i in perm:
                    img = selectTrigger(ctx, img, width, height, distance,
                                        trig_w, trig_h, trigger_type)
                    dataset_.append((img, target_label))
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

    if "femnist" in ctx.data.type:
        inject_portion_train = ctx.attack.poison_ratio
        target_label = torch.tensor(int(ctx.attack.target_label_ind)).long()

    elif "CIFAR10" in ctx.data.type:
        inject_portion_train = ctx.attack.poison_ratio
        target_label = int(ctx.attack.target_label_ind)

    else:
        raise RuntimeError(
            'Now, we only support the FEMNIST and CIFAR-10 datasets')

    inject_portion_test = 1.0

    if mode == 'train':
        poisoned_dataset = addTrigger(ctx, data['train'].dataset, target_label, inject_portion_train, \
            mode = 'train', distance=1, trig_h = 0.1, trig_w = 0.1, \
                trigger_type = trigger_type, label_type = label_type)
        num_dps_poisoned_dataset = len(poisoned_dataset)
        for iii in range(num_dps_poisoned_dataset):
            sample, label = poisoned_dataset[iii]
            poisoned_dataset[iii] = (transforms_funcs(sample), label)

        data['train'] = DataLoader(poisoned_dataset,
                                   batch_size=ctx.data.batch_size,
                                   shuffle=True,
                                   num_workers=ctx.data.num_workers)

    if mode == 'test':
        poisoned_dataset = addTrigger(ctx, data[mode].dataset, target_label, inject_portion_test, mode = mode, \
            distance=1, trig_h = 0.1, trig_w = 0.1, trigger_type = trigger_type, label_type = label_type)
        num_dps_poisoned_dataset = len(poisoned_dataset)
        for iii in range(num_dps_poisoned_dataset):
            sample, label = poisoned_dataset[iii]
            poisoned_dataset[iii] = (transforms_funcs(sample), label)

        data['poison'] = DataLoader(poisoned_dataset,
                                    batch_size=ctx.data.batch_size,
                                    shuffle=False,
                                    num_workers=ctx.data.num_workers)

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
                sample = np.array(data[key].dataset[iii][0]).transpose(
                    1, 2, 0) * 255.0
                sample = np.clip(sample.astype('uint8'), 0, 255)
                sample = Image.fromarray(sample)
                sample = tran_train(sample)
                data[key].dataset[iii] = (normalize(sample, mean, std),
                                          data[key].dataset[iii][1])
        else:
            for iii in range(num_dataset):
                data[key].dataset[iii] = (normalize(data[key].dataset[iii][0], mean, std), \
                    data[key].dataset[iii][1])

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
    for i in range(1, len(data) + 1):
        if i == ctx.attack.attacker_id:
            logger.info(50 * '-')
            logger.info('start poisoning!!!!!!')
            logger.info(50 * '-')
            data[i] = select_poisoning(data[i], ctx, mode='train')
        data[i] = select_poisoning(data[i], ctx, mode='test')

        if 'cifar' in ctx.data.type.lower():
            class_num_train = torch.zeros(10)
            class_num_test = torch.zeros(10)
        else:
            class_num_train = torch.zeros(62)
            class_num_test = torch.zeros(62)

        for jj in data[i]['train'].dataset:
            ind = jj[1]
            class_num_train[ind] += 1
        print('training: the {} client'.format(i))
        print(class_num_train)
        print('the training number of client {}: {}'.format(
            i,
            class_num_train.sum().item()))

        for jj in data[i]['test'].dataset:
            ind = jj[1]
            class_num_test[ind] += 1
        print('testing: the {} client'.format(i))
        print(class_num_test)
        print('the testing number of client {}: {}'.format(
            i,
            class_num_test.sum().item()))

        data[i] = add_trans_normalize(data[i], ctx)
        logger.info('finishing the clean and {} poisoning data processing for Client {:d}'\
            .format(ctx.attack.trigger_type, i))
