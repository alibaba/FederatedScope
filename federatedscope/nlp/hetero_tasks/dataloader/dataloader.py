import os
import logging
import copy
from tqdm import tqdm

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
logger = logging.getLogger(__name__)


def modified_cfg(cfg, cfg_client):

    # create the temp dir for output
    tmp_dir = os.path.join(cfg.outdir, 'temp')
    os.makedirs(tmp_dir, exist_ok=True)

    if cfg.federate.save_to:
        cfg.federate.save_to = os.path.join(cfg.outdir, cfg.federate.save_to)
        os.makedirs(cfg.federate.save_to, exist_ok=True)

    # modification for debug (load a small subset from the whole dataset)
    if cfg.data.is_debug:
        cfg.federate.client_num = 6
        cfg.data.hetero_data_name = [
            'imdb', 'agnews', 'squad', 'newsqa', 'cnndm', 'msqg'
        ]
        cfg.data.num_of_client_for_data = [1, 1, 1, 1, 1, 1]
        cfg.federate.total_round_num = 2
        cfg.train.local_update_steps = 2
        # TODO
        cfg.federate.atc_load_from = ''
        cfg.federate.save_to = ''
        cfg.data.cache_dir = ''
        cfg.data.batch_size = 1

    if cfg.model.stage == 'assign':
        downstream_tasks = []
        for group_id, num_clients in enumerate(
                cfg.data.num_of_client_for_data):
            downstream_tasks += [cfg.data.hetero_data_name[group_id]
                                 ] * num_clients
        cfg.model.downstream_tasks = downstream_tasks
    elif cfg.model.stage == 'contrast':
        num_agg_topk = []
        if len(cfg.aggregator.num_agg_topk) > 0:
            for group_id, num_clients in enumerate(
                    cfg.data.num_of_client_for_data):
                num_agg_topk += [cfg.aggregator.num_agg_topk[group_id]] * \
                                num_clients
        else:
            for group_id, num_clients in enumerate(
                    cfg.data.num_of_client_for_data):
                num_agg_topk += [cfg.federate.client_num] * num_clients
        cfg.aggregator.num_agg_topk = num_agg_topk

    # client_config
    if cfg_client is not None:
        with open(os.path.join(cfg.outdir, 'config_client.yaml'),
                  'w') as outfile:
            from contextlib import redirect_stdout
            with redirect_stdout(outfile):
                tmp_cfg = copy.deepcopy(cfg_client)
                tmp_cfg.cfg_check_funcs = []
                print(tmp_cfg.dump())

        num_of_client_for_data = cfg.data.num_of_client_for_data
        client_start_id = 1
        for group_id, num_clients in enumerate(num_of_client_for_data):
            group_cfg = cfg_client['client_group_{}'.format(group_id + 1)]
            if cfg.data.is_debug:
                if group_cfg.train.local_update_steps > 5:
                    group_cfg.train.local_update_steps = 5
                group_cfg.data.batch_size = 1
            for client_id in range(client_start_id,
                                   client_start_id + num_clients):
                cfg_client['client_{}'.format(client_id)] = group_cfg
            client_start_id += num_clients

    return cfg, cfg_client


def load_heteroNLP_data(config, client_cfgs):

    from torch.utils.data import DataLoader
    from federatedscope.nlp.hetero_tasks.dataset.utils import setup_tokenizer
    from federatedscope.nlp.hetero_tasks.dataset.get_data import \
        HeteroNLPDataLoader, SynthDataProcessor
    from federatedscope.nlp.hetero_tasks.dataloader.datacollator import \
        DataCollator

    class HeteroNLPDataset(object):
        ALL_DATA_NAME = ['imdb', 'agnews', 'squad', 'newsqa', 'cnndm', 'msqg']

        def __init__(self, config, client_cfgs):

            self.root = config.data.root
            self.num_of_clients = config.data.num_of_client_for_data
            self.data_name = list()
            for each_data in config.data.hetero_data_name:
                if each_data not in self.ALL_DATA_NAME:
                    logger.warning(f'We have not provided {each_data} in '
                                   f'HeteroNLPDataset.')
                else:
                    self.data_name.append(each_data)

            data = HeteroNLPDataLoader(
                data_dir=self.root,
                data_name=self.data_name,
                num_of_clients=self.num_of_clients).get_data()

            self.processed_data = self._preprocess(config, client_cfgs, data)

        def __getitem__(self, idx):
            return self.processed_data[idx]

        def __len__(self):
            return len(self.processed_data)

        def _preprocess(self, config, client_cfgs, data):

            use_pretrain_task = config.model.stage == 'assign'
            use_contrastive = config.model.stage == 'contrast'
            tokenizer = setup_tokenizer(config.model.model_type)
            data_collator = DataCollator(tokenizer=tokenizer) \
                if use_pretrain_task else None
            is_debug = config.data.is_debug  # load a subset of data

            processed_data = list()
            for client_id in tqdm(range(1, config.federate.client_num + 1)):
                applied_cfg = config if use_pretrain_task \
                    else client_cfgs['client_{}'.format(client_id)]

                cur_task = applied_cfg.model.downstream_tasks[client_id - 1] \
                    if use_pretrain_task else applied_cfg.model.task

                train_data, val_data, test_data = [
                    self._process_data(data=data[split][client_id - 1],
                                       data_name=cur_task,
                                       split=split,
                                       tokenizer=tokenizer,
                                       model_type=config.model.model_type,
                                       cache_dir=config.data.cache_dir,
                                       cfg=applied_cfg.data,
                                       client_id=client_id,
                                       pretrain=use_pretrain_task,
                                       is_debug=is_debug)
                    for split in ['train', 'val', 'test']
                ]

                dataloader = {}
                dataloader['val'] = {
                    'dataloader': DataLoader(
                        dataset=val_data[0],
                        batch_size=applied_cfg.data.batch_size,
                        shuffle=False,
                        num_workers=config.data.num_workers,
                        collate_fn=data_collator,
                        pin_memory=config.use_gpu),
                    'encoded': val_data[1],
                    'examples': val_data[2]
                }
                dataloader['test'] = {
                    'dataloader': DataLoader(
                        dataset=test_data[0],
                        batch_size=applied_cfg.data.batch_size,
                        shuffle=False,
                        num_workers=config.data.num_workers,
                        collate_fn=data_collator,
                        pin_memory=config.use_gpu),
                    'encoded': test_data[1],
                    'examples': test_data[2]
                }

                if not use_contrastive:
                    dataloader['train'] = {
                        'dataloader': DataLoader(
                            dataset=train_data[0],
                            batch_size=applied_cfg.data.batch_size,
                            shuffle=config.data.shuffle,
                            num_workers=config.data.num_workers,
                            collate_fn=data_collator,
                            pin_memory=config.use_gpu),
                        'encoded': train_data[1],
                        'examples': train_data[2]
                    }
                else:
                    dataloader['train_raw'] = {
                        'dataloader': DataLoader(
                            dataset=train_data[0],
                            batch_size=applied_cfg.data.batch_size,
                            shuffle=config.data.shuffle,
                            num_workers=config.data.num_workers,
                            collate_fn=data_collator,
                            pin_memory=config.use_gpu),
                        'encoded': train_data[1],
                        'examples': train_data[2]
                    }
                    dataloader['train_contrast'] = {
                        'dataloader': DataLoader(
                            dataset=train_data[0],
                            batch_size=applied_cfg.data.batch_size,
                            shuffle=False,
                            num_workers=config.data.num_workers,
                            collate_fn=data_collator,
                            pin_memory=config.use_gpu),
                        'encoded': train_data[1],
                        'examples': train_data[2]
                    }
                processed_data.append(dataloader)

            if use_contrastive:
                logger.info(
                    'Preprocessing synthetic dataset for contrastive learning')
                synth_data_processor = SynthDataProcessor(
                    config, processed_data)
                synth_data_processor.save_data()

            return processed_data

        def _process_data(self, data, data_name, split, tokenizer, model_type,
                          cache_dir, cfg, client_id, pretrain, is_debug):
            if data_name == 'imdb':
                from federatedscope.nlp.hetero_tasks.dataset.imdb import \
                    process_imdb_dataset
                process_func = process_imdb_dataset
            elif data_name == 'agnews':
                from federatedscope.nlp.hetero_tasks.dataset.agnews import \
                    process_agnews_dataset
                process_func = process_agnews_dataset
            elif data_name == 'squad':
                from federatedscope.nlp.hetero_tasks.dataset.squad import \
                    process_squad_dataset
                process_func = process_squad_dataset
            elif data_name == 'newsqa':
                from federatedscope.nlp.hetero_tasks.dataset.newsqa import \
                    process_newsqa_dataset
                process_func = process_newsqa_dataset
            elif data_name == 'cnndm':
                from federatedscope.nlp.hetero_tasks.dataset.cnndm import \
                    process_cnndm_dataset
                process_func = process_cnndm_dataset
            elif data_name == 'msqg':
                from federatedscope.nlp.hetero_tasks.dataset.msqg import \
                    process_msqg_dataset
                process_func = process_msqg_dataset
            else:
                raise NotImplementedError(
                    f'Not process function is provided for {data_name}')

            max_seq_len = getattr(cfg, 'max_seq_len', None)
            max_query_len = getattr(cfg, 'max_query_len', None)
            trunc_stride = getattr(cfg, 'trunc_stride', None)
            max_tgt_len = getattr(cfg, 'max_tgt_len', None)

            return process_func(data=data,
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
                                is_debug=is_debug)

    modified_config, modified_client_cfgs = modified_cfg(config, client_cfgs)
    dataset = HeteroNLPDataset(modified_config, modified_client_cfgs)
    # Convert to dict
    datadict = {
        client_id + 1: dataset[client_id]
        for client_id in range(len(dataset))
    }

    return datadict, modified_config
