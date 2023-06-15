import os
import logging
import torch
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from federatedscope.nlp.prompt_tuning.dataset.dataset import PLDataProcessor
from federatedscope.nlp.prompt_tuning.dataset.utils import DatasetDict, \
    setup_tokenizer

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
logger = logging.getLogger(__name__)


def extend_cfg(config):
    dataset_name = config.data.dataset_name
    if dataset_name == 'multirc':
        config.eval.metrics = ['f1']
    elif dataset_name == 'record':
        config.eval.metrics = ['record']
    else:
        config.eval.metrics = ['acc']

    if config.federate.pl_save_to:
        config.federate.pl_save_to = os.path.join(config.outdir,
                                                  config.federate.pl_save_to)
    config.personalization.server_local_param += \
        config.model.server_freeze_param
    config.personalization.client_local_param += \
        config.model.client_freeze_param

    if config.federate.skip_local_train:
        config.federate.total_round_num = 1

    if config.data.is_debug:
        config.federate.client_num = 2
        config.federate.total_round_num = 2
        config.train.local_update_steps = 2
        config.data.batch_size = 1

    return config


def load_pl_data(config):
    if config.use_ddp:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(dist.get_rank())

    extend_cfg(config)
    tokenizer = setup_tokenizer(config)
    debug = config.data.is_debug
    logger.info(f'Preprocessing dataset {config.data.type}')
    data_processor = PLDataProcessor(config, tokenizer, debug)
    train_data, val_data, test_data = data_processor.split_data()

    data_dict = dict()
    for client_id in tqdm(range(config.federate.client_num + 1)):
        dataloader_dict = {}
        if train_data[client_id] is not None:
            dataset = DatasetDict(train_data[client_id])
            dataloader_dict = {
                'train': {
                    'dataloader': DataLoader(
                        dataset=dataset,
                        batch_size=config.data.batch_size,
                        num_workers=config.data.num_workers,
                        pin_memory=config.use_gpu,
                        sampler=DistributedSampler(dataset)
                        if config.use_ddp else RandomSampler(dataset),
                    ),
                },
            }

        if client_id == 0:  # server
            val_dataset = DatasetDict(val_data)
            test_dataset = DatasetDict(test_data)
            dataloader_dict.update({
                'val': {
                    'dataloader': DataLoader(
                        dataset=val_dataset,
                        batch_size=config.data.batch_size,
                        num_workers=config.data.num_workers,
                        pin_memory=config.use_gpu,
                        sampler=SequentialSampler(val_dataset),
                    ),
                },
                'test': {
                    'dataloader': DataLoader(
                        dataset=test_dataset,
                        batch_size=config.data.batch_size,
                        num_workers=config.data.num_workers,
                        pin_memory=config.use_gpu,
                        sampler=SequentialSampler(test_dataset),
                    ),
                },
            })
        data_dict[client_id] = dataloader_dict

    return data_dict, config
