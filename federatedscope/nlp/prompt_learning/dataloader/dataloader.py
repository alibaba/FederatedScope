import os
import logging
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from federatedscope.register import register_data
from federatedscope.nlp.prompt_learning.dataset.dataset import \
    PLDataProcessor, create_pl_dataset
from federatedscope.nlp.prompt_learning.dataset.utils import setup_tokenizer, \
    SERVER_TRAIN

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
logger = logging.getLogger(__name__)


def extend_cfg(config):
    dataset_name = config.data.dataset_name
    if dataset_name == 'copa':
        config.model.num_labels = 1
    elif dataset_name == 'cb':
        config.model.num_labels = 3
    else:
        config.model.num_labels = 2

    if dataset_name == 'multirc':
        config.eval.metrics = ['f1']
    elif dataset_name == 'record':
        config.eval.metrics = ['record']
    else:
        config.eval.metrics = ['acc']

    if config.federate.pl_save_to:
        config.federate.pl_save_to = os.path.join(config.outdir,
                                                  config.federate.pl_save_to)
        os.makedirs(config.federate.pl_save_to, exist_ok=True)

    config.personalization.local_param += config.model.freeze_param

    if config.data.debug:
        config.federate.client_num = 2
        config.federate.total_round_num = 2
        config.train.local_update_steps = 2
        config.data.batch_size = 1

    return config


def collate_fn(batch):
    out = {k: [d[k] for d in batch] for k in batch[0]}
    out = {
        k: torch.stack(v) if isinstance(v[0], torch.Tensor) else v
        for k, v in out.items()
    }
    return out


def load_pl_data(config):
    extend_cfg(config)
    tokenizer = setup_tokenizer(config)
    debug = config.data.debug

    logger.info(f'Preprocessing dataset {config.data.type}')
    data_processor = PLDataProcessor(config, train_frac=0.9)
    train_data, val_data, test_data = data_processor.split_data()

    data_dict = dict()
    for client_id in tqdm(range(config.federate.client_num + 1)):
        if not SERVER_TRAIN and client_id == 0:
            dataloader_dict = {}
        else:
            cur_train_data = create_pl_dataset(
                data=train_data[client_id],
                tokenizer=tokenizer,
                dataset_name=config.data.dataset_name,
                max_seq_len=config.data.max_seq_len,
                debug=debug)

            dataloader_dict = {
                'train': {
                    'dataloader': DataLoader(
                        dataset=cur_train_data,
                        batch_size=config.data.batch_size,
                        shuffle=config.data.shuffle,
                        num_workers=config.data.num_workers,
                        pin_memory=config.use_gpu,
                        collate_fn=collate_fn),
                },
            }

        if client_id == 0:  # server
            cur_val_data, cur_test_data = [
                create_pl_dataset(data=data,
                                  tokenizer=tokenizer,
                                  dataset_name=config.data.dataset_name,
                                  max_seq_len=config.data.max_seq_len,
                                  debug=debug)
                for data in [val_data, test_data]
            ]

            dataloader_dict.update({
                'val': {
                    'dataloader': DataLoader(
                        dataset=cur_val_data,
                        batch_size=config.data.batch_size,
                        shuffle=False,
                        num_workers=config.data.num_workers,
                        pin_memory=config.use_gpu,
                        collate_fn=collate_fn),
                },
                'test': {
                    'dataloader': DataLoader(
                        dataset=cur_test_data,
                        batch_size=config.data.batch_size,
                        shuffle=False,
                        num_workers=config.data.num_workers,
                        pin_memory=config.use_gpu,
                        collate_fn=collate_fn),
                },
            })
        data_dict[client_id] = dataloader_dict

    return data_dict, config


def call_pl_data(config, *args):
    if config.data.type == 'pl_data':
        data, modified_config = load_pl_data(config)
        return data, modified_config


register_data('pl_data', call_pl_data)
