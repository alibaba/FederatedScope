import os
import logging
import copy
from tqdm import tqdm
from torch.utils.data import DataLoader
from federatedscope.register import register_data
from federatedscope.nlp.hetero_tasks.dataset.get_data import \
    ATCDataProcessor, ATCSynthDataProcessor
from federatedscope.nlp.hetero_tasks.dataset.utils import setup_tokenizer
from federatedscope.nlp.hetero_tasks.dataset.imdb import create_imdb_dataset
from federatedscope.nlp.hetero_tasks.dataset.agnews import \
    create_agnews_dataset
from federatedscope.nlp.hetero_tasks.dataset.squad import create_squad_dataset
from federatedscope.nlp.hetero_tasks.dataset.newsqa import \
    create_newsqa_dataset
from federatedscope.nlp.hetero_tasks.dataset.cnndm import create_cnndm_dataset
from federatedscope.nlp.hetero_tasks.dataset.msqg import create_msqg_dataset
from federatedscope.nlp.hetero_tasks.dataloader.data_collator import \
    DataCollatorForATC

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
logger = logging.getLogger(__name__)


def extend_cfg(cfg, cfg_client):
    # config
    cfg.eval.result_path = cfg.outdir
    cfg.eval.temp_dir = os.path.join(cfg.outdir, 'temp')
    os.makedirs(cfg.eval.temp_dir, exist_ok=True)

    if cfg.data.debug:
        if cfg.federate.client_num > 10:
            cfg.federate.client_num = 10
            cfg.data.datasets = [
                'imdb', 'agnews', 'squad', 'newsqa', 'cnndm', 'msqg'
            ]
            cfg.data.num_grouped_clients = [2, 2, 2, 2, 1, 1]
        if cfg.federate.total_round_num > 2:
            cfg.federate.total_round_num = 2
        if cfg.train.local_update_steps > 2:
            cfg.train.local_update_steps = 2
        cfg.federate.atc_load_from = ''
        cfg.federate.save_to = ''
        cfg.data.cache_dir = ''
        cfg.data.batch_size = 1

    if cfg.federate.save_to:
        cfg.federate.save_to = os.path.join(cfg.outdir, cfg.federate.save_to)
        save_dir = cfg.federate.save_to
        os.makedirs(save_dir, exist_ok=True)

    if cfg.model.task == 'pretrain':
        downstream_tasks = []
        for group_id, num_clients in enumerate(cfg.data.num_grouped_clients):
            downstream_tasks += [cfg.data.datasets[group_id]] * num_clients
        cfg.model.downstream_tasks = downstream_tasks
    else:
        num_agg_topk = []
        if len(cfg.aggregator.num_agg_topk) > 0:
            for group_id, num_clients in enumerate(
                    cfg.data.num_grouped_clients):
                num_agg_topk += [cfg.aggregator.num_agg_topk[group_id]] * \
                                num_clients
        else:
            for group_id, num_clients in enumerate(
                    cfg.data.num_grouped_clients):
                num_agg_topk += [cfg.federate.client_num] * num_clients
        cfg.aggregator.num_agg_topk = num_agg_topk

    tokenizer = setup_tokenizer(cfg)
    cfg.model.bos_token_id = tokenizer.bos_token_id
    cfg.model.eos_token_id = tokenizer.eos_token_id
    cfg.model.eoq_token_id = tokenizer.eoq_token_id
    cfg.model.pad_token_id = tokenizer.pad_token_id

    # client_config
    if cfg_client is not None:
        with open(os.path.join(cfg.outdir, 'config_client.yaml'),
                  'w') as outfile:
            from contextlib import redirect_stdout
            with redirect_stdout(outfile):
                tmp_cfg = copy.deepcopy(cfg_client)
                tmp_cfg.cfg_check_funcs = []
                print(tmp_cfg.dump())

        num_grouped_clients = cfg.data.num_grouped_clients
        client_start_id = 1
        for group_id, num_clients in enumerate(num_grouped_clients):
            group_cfg = cfg_client['client_group_{}'.format(group_id + 1)]
            if cfg.data.debug:
                if group_cfg.train.local_update_steps > 5:
                    group_cfg.train.local_update_steps = 5
                group_cfg.data.batch_size = 1
            for client_id in range(client_start_id,
                                   client_start_id + num_clients):
                cfg_client['client_{}'.format(client_id)] = group_cfg
            client_start_id += num_clients

    return cfg, cfg_client


def create_data(data, split, tokenizer, task, model_type, max_seq_len,
                max_query_len, trunc_stride, max_tgt_len, cache_dir, client_id,
                pretrain, debug):
    if task == 'imdb':
        create_dataset_func = create_imdb_dataset
    elif task == 'agnews':
        create_dataset_func = create_agnews_dataset
    elif task == 'squad':
        create_dataset_func = create_squad_dataset
    elif task == 'newsqa':
        create_dataset_func = create_newsqa_dataset
    elif task == 'cnndm':
        create_dataset_func = create_cnndm_dataset
    elif task == 'msqg':
        create_dataset_func = create_msqg_dataset
    else:
        raise ValueError('No ATC dataset named {}'.format(task))

    return create_dataset_func(data=data,
                               split=split,
                               tokenizer=tokenizer,
                               max_seq_len=max_seq_len,
                               max_query_len=max_query_len,
                               trunc_stride=trunc_stride,
                               max_src_len=max_seq_len,
                               max_tgt_len=max_tgt_len,
                               model_type=model_type,
                               cache_dir=cache_dir,
                               raw_cache_dir=cache_dir,
                               client_id=client_id,
                               pretrain=pretrain,
                               debug=debug)


def load_atc_data(config, client_config):
    extend_cfg(config, client_config)
    model_type = config.model.model_type
    tokenizer = setup_tokenizer(config)
    pretrain = config.model.task == 'pretrain'
    use_contrastive_loss = config.model.use_contrastive_loss
    cache_dir = config.data.cache_dir
    debug = config.data.debug
    data_collator = DataCollatorForATC(tokenizer=tokenizer) \
        if pretrain else None

    logger.info('Preprocessing dataset {}'.format(config.data.type))
    data_processor = ATCDataProcessor(config)
    all_data = data_processor.get_data()
    all_data_dict = {
        'train': all_data[0],
        'val': all_data[1],
        'test': all_data[2]
    }

    data_dict = dict()
    for client_id in tqdm(range(1, config.federate.client_num + 1)):
        cfg_client = config if pretrain else \
            client_config['client_{}'.format(client_id)]
        cur_task = cfg_client.model.downstream_tasks[client_id - 1] \
            if pretrain else cfg_client.model.task
        train_data, val_data, test_data = [
            create_data(data=all_data_dict[split][client_id - 1],
                        split=split,
                        tokenizer=tokenizer,
                        task=cur_task,
                        model_type=model_type,
                        max_seq_len=getattr(cfg_client.data, 'max_seq_len',
                                            None),
                        max_query_len=getattr(cfg_client.data, 'max_query_len',
                                              None),
                        trunc_stride=getattr(cfg_client.data, 'trunc_stride',
                                             None),
                        max_tgt_len=getattr(cfg_client.data, 'max_tgt_len',
                                            None),
                        cache_dir=cache_dir,
                        client_id=client_id,
                        pretrain=pretrain,
                        debug=debug) for split in ['train', 'val', 'test']
        ]

        if not use_contrastive_loss:
            dataloader_dict = {
                'train': {
                    'dataloader': DataLoader(
                        dataset=train_data[0],
                        batch_size=cfg_client.data.batch_size,
                        shuffle=config.data.shuffle,
                        num_workers=config.data.num_workers,
                        collate_fn=data_collator,
                        pin_memory=config.use_gpu),
                    'encoded': train_data[1],
                    'examples': train_data[2]
                },
                'val': {
                    'dataloader': DataLoader(
                        dataset=val_data[0],
                        batch_size=cfg_client.data.batch_size,
                        shuffle=False,
                        num_workers=config.data.num_workers,
                        collate_fn=data_collator,
                        pin_memory=config.use_gpu),
                    'encoded': val_data[1],
                    'examples': val_data[2]
                },
                'test': {
                    'dataloader': DataLoader(
                        dataset=test_data[0],
                        batch_size=cfg_client.data.batch_size,
                        shuffle=False,
                        num_workers=config.data.num_workers,
                        collate_fn=data_collator,
                        pin_memory=config.use_gpu),
                    'encoded': test_data[1],
                    'examples': test_data[2]
                },
            }
        else:
            dataloader_dict = {
                'train_raw': {
                    'dataloader': DataLoader(
                        dataset=train_data[0],
                        batch_size=cfg_client.data.batch_size,
                        shuffle=config.data.shuffle,
                        num_workers=config.data.num_workers,
                        collate_fn=data_collator,
                        pin_memory=config.use_gpu),
                    'encoded': train_data[1],
                    'examples': train_data[2]
                },
                'train_contrast': {
                    'dataloader': DataLoader(
                        dataset=train_data[0],
                        batch_size=cfg_client.data.batch_size,
                        shuffle=False,
                        num_workers=config.data.num_workers,
                        collate_fn=data_collator,
                        pin_memory=config.use_gpu),
                    'encoded': train_data[1],
                    'examples': train_data[2]
                },
                'val': {
                    'dataloader': DataLoader(
                        dataset=val_data[0],
                        batch_size=cfg_client.data.batch_size,
                        shuffle=False,
                        num_workers=config.data.num_workers,
                        collate_fn=data_collator,
                        pin_memory=config.use_gpu),
                    'encoded': val_data[1],
                    'examples': val_data[2]
                },
                'test': {
                    'dataloader': DataLoader(
                        dataset=test_data[0],
                        batch_size=cfg_client.data.batch_size,
                        shuffle=False,
                        num_workers=config.data.num_workers,
                        collate_fn=data_collator,
                        pin_memory=config.use_gpu),
                    'encoded': test_data[1],
                    'examples': test_data[2]
                },
            }
        data_dict[client_id] = dataloader_dict

    if use_contrastive_loss:
        logger.info('Preprocessing synthetic dataset')
        synth_data_processor = ATCSynthDataProcessor(config, data_dict)
        synth_data_processor.save_data()

    return data_dict, config


def call_atc_data(config, client_config):
    if config.data.type == 'atc_data':
        data, modified_config = load_atc_data(config, client_config)
        return data, modified_config


register_data('atc_data', call_atc_data)