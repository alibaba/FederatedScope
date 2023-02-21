import os
import logging
import random
import csv
import json
import gzip
import zipfile
import shutil
import copy
import torch
import numpy as np
from tqdm import tqdm
from federatedscope.core.data.utils import download_url
from federatedscope.core.gpu_manager import GPUManager
from federatedscope.nlp.hetero_tasks.model.model import ATCModel

logger = logging.getLogger(__name__)


class HeteroNLPDataLoader(object):
    """
    Load hetero NLP task datasets (including multiple datasets), split them
    into train/val/test, and partition into several clients.
    """
    def __init__(self, data_dir, data_name, num_of_clients, split=[0.9, 0.1]):
        self.data_dir = data_dir
        self.data_name = data_name
        self.num_of_clients = num_of_clients
        self.split = split  # split for train:val
        self.train_data = []
        self.val_data = []
        self.test_data = []

    def get_data(self):
        for each_data, each_client_num in zip(self.data_name,
                                              self.num_of_clients):
            train_and_val_data = self._load(each_data, 'train',
                                            each_client_num)
            each_train_data = [
                data[:int(self.split[0] * len(data))]
                for data in train_and_val_data
            ]
            each_val_data = [
                data[-int(self.split[1] * len(data)):]
                for data in train_and_val_data
            ]
            each_test_data = self._load(each_data, 'test', each_client_num)
            self.train_data.extend(each_train_data)
            self.val_data.extend(each_val_data)
            self.test_data.extend(each_test_data)

        return {
            'train': self.train_data,
            'val': self.val_data,
            'test': self.test_data
        }

    def _load(self, dataset, split, num_of_client):
        data_dir = os.path.join(self.data_dir, dataset)
        if not os.path.exists(data_dir):
            logger.info(f'Start tp download the dataset {dataset} ...')
            self._download_and_extract(dataset)

        # read data
        data = []
        if dataset == 'imdb':
            pos_files = os.listdir(os.path.join(data_dir, split, 'pos'))
            neg_files = os.listdir(os.path.join(data_dir, split, 'neg'))
            for file in pos_files:
                path = os.path.join(data_dir, split, 'pos', file)
                with open(path) as f:
                    line = f.readline()
                data.append({'text': line, 'label': 1})
            for file in neg_files:
                path = os.path.join(data_dir, split, 'neg', file)
                with open(path) as f:
                    line = f.readline()
                data.append({'text': line, 'label': 0})
            random.shuffle(data)

        elif dataset == 'agnews':
            with open(os.path.join(data_dir, split + '.csv'),
                      encoding="utf-8") as csv_file:
                csv_reader = csv.reader(csv_file,
                                        quotechar='"',
                                        delimiter=",",
                                        quoting=csv.QUOTE_ALL,
                                        skipinitialspace=True)
                for i, row in enumerate(csv_reader):
                    label, title, description = row
                    label = int(label) - 1
                    text = ' [SEP] '.join((title, description))
                    data.append({'text': text, 'label': label})

        elif dataset == 'squad':
            with open(os.path.join(data_dir, split + '.json'),
                      'r',
                      encoding='utf-8') as reader:
                raw_data = json.load(reader)['data']
            for line in raw_data:
                for para in line['paragraphs']:
                    context = para['context']
                    for qa in para['qas']:
                        data.append({'context': context, 'qa': qa})

        elif dataset == 'newsqa':
            with gzip.GzipFile(os.path.join(data_dir, split + '.jsonl.gz'),
                               'r') as reader:
                content = reader.read().decode('utf-8').strip().split('\n')[1:]
                raw_data = [json.loads(line) for line in content]
            for line in raw_data:
                context = line['context']
                for qa in line['qas']:
                    data.append({'context': context, 'qa': qa})

        elif dataset in {'cnndm', 'msqg'}:
            src_file = os.path.join(data_dir, split + '.src')
            tgt_file = os.path.join(data_dir, split + '.tgt')
            with open(src_file) as f:
                src_data = [
                    line.strip().replace('<S_SEP>', '[SEP]') for line in f
                ]
            with open(tgt_file) as f:
                tgt_data = [
                    line.strip().replace('<S_SEP>', '[SEP]') for line in f
                ]
            for src, tgt in zip(src_data, tgt_data):
                data.append({'src': src, 'tgt': tgt})

        # split data
        logger.info(f'Spliting dataset {dataset} ({split})')
        splited_data = []
        n = len(data) // num_of_client
        data_idx = 0
        for i in range(num_of_client):
            num_split = n if i < num_of_client - 1 else \
                len(data) - n * (num_of_client - 1)
            cur_data = data[data_idx:data_idx + num_split]
            data_idx += num_split
            splited_data.append(cur_data)
        logger.info(f'Dataset {dataset} ({split}) is splited into '
                    f'{[len(x) for x in splited_data]}')

        return splited_data

    def _download_and_extract(self, dataset):
        url = 'https://federatedscope.oss-cn-beijing.aliyuncs.com'
        os.makedirs(self.data_dir, exist_ok=True)
        download_url(f'{url}/{dataset}.zip', self.data_dir)

        raw_dir = os.path.join(self.data_dir, dataset + '_raw')
        extract_dir = os.path.join(self.data_dir, dataset)
        with zipfile.ZipFile(os.path.join(self.data_dir, f'{dataset}.zip'),
                             'r') as zip_ref:
            zip_ref.extractall(raw_dir)
        shutil.move(os.path.join(raw_dir, dataset), self.data_dir)
        if os.path.exists(os.path.join(extract_dir, '.DS_Store')):
            os.remove(os.path.join(extract_dir, '.DS_Store'))
        os.remove(os.path.join(self.data_dir, f'{dataset}.zip'))
        shutil.rmtree(raw_dir)


class SynthDataProcessor(object):
    def __init__(self, config, datasets):
        self.cfg = config
        self.device = GPUManager(
            gpu_available=self.cfg.use_gpu,
            specified_device=self.cfg.device).auto_choice()
        self.pretrain_dir = config.federate.atc_load_from
        self.cache_dir = 'cache_debug' if \
            config.data.is_debug else config.data.cache_dir
        self.save_dir = os.path.join(self.cache_dir, 'synthetic')
        self.batch_size = config.data.hetero_synth_batch_size
        self.datasets = datasets
        self.num_clients = len(datasets)
        self.synth_prim_weight = config.data.hetero_synth_prim_weight
        self.synth_feat_dim = config.data.hetero_synth_feat_dim
        self.models = {}

    def save_data(self):
        if os.path.exists(self.save_dir):
            return

        max_sz, max_len = 1e8, 0
        for client_id in range(1, self.num_clients + 1):
            dataset = self.datasets[client_id -
                                    1]['train_contrast']['dataloader'].dataset
            max_sz = min(max_sz, len(dataset))
            max_len = max(max_len, len(dataset[0]['token_ids']))
        enc_hiddens = np.memmap(filename=os.path.join(self.cfg.outdir,
                                                      'tmp_feat.memmap'),
                                shape=(self.num_clients, max_sz, max_len,
                                       self.synth_feat_dim),
                                mode='w+',
                                dtype=np.float32)
        self._get_models()

        logger.info('Generating synthetic encoder hidden states')
        for client_id in tqdm(range(1, self.num_clients + 1)):
            dataloader = self.datasets[client_id -
                                       1]['train_contrast']['dataloader']
            model = self.models[client_id]
            model.eval()
            model.to(self.device)
            enc_hid = []
            for batch_i, data_batch in tqdm(enumerate(dataloader),
                                            total=len(dataloader)):
                token_ids = data_batch['token_ids']
                token_type_ids = data_batch['token_type_ids']
                attention_mask = data_batch['attention_mask']
                enc_out = model.model.encoder(
                    input_ids=token_ids.to(self.device),
                    attention_mask=attention_mask.to(self.device),
                    token_type_ids=token_type_ids.to(self.device),
                )
                enc_hid.append(enc_out.last_hidden_state.detach().cpu())

            enc_hid = torch.cat(enc_hid)
            if enc_hid.size(1) < max_len:
                enc_hid = torch.cat([
                    enc_hid,
                    torch.zeros(enc_hid.size(0), max_len - enc_hid.size(1),
                                self.synth_feat_dim)
                ],
                                    dim=1)
            enc_hiddens[client_id - 1] = enc_hid[:max_sz]
            model.to('cpu')

        all_hids = torch.from_numpy(enc_hiddens)
        prim_indices = [
            random.randint(0,
                           len(all_hids) - 1) for _ in range(len(all_hids[0]))
        ]  # avoid over-smooth results when setting
        # equal merging weights to all clients
        all_weights = torch.ones(len(all_hids), len(all_hids[0]))
        all_weights *= (1 - self.synth_prim_weight) / (len(all_hids) - 1)
        for i, pi in enumerate(prim_indices):
            all_weights[pi, i] = self.synth_prim_weight
        avg_hids = (all_hids * all_weights[:, :, None, None]).sum(0)

        logger.info('Generating synthetic input tokens')
        lm_head = self._get_avg_lm_head().to(self.device)
        with torch.no_grad():
            pred_toks = torch.cat([
                lm_head(avg_hids[i:i + self.batch_size].to(
                    self.device)).detach().cpu().argmax(dim=-1)
                for i in tqdm(range(0, avg_hids.size(0), self.batch_size))
            ])

        if self.cache_dir:
            logger.info('Saving synthetic data to \'{}\''.format(
                self.save_dir))
            os.makedirs(self.save_dir, exist_ok=True)
            saved_feats = np.memmap(
                filename=os.path.join(
                    self.save_dir,
                    'feature_{}.memmap'.format(self.synth_prim_weight)),
                shape=avg_hids.size(),
                mode='w+',
                dtype=np.float32,
            )
            saved_toks = np.memmap(
                filename=os.path.join(
                    self.save_dir,
                    'token_{}.memmap'.format(self.synth_prim_weight)),
                shape=pred_toks.size(),
                mode='w+',
                dtype=np.int64,
            )
            for i in range(len(avg_hids)):
                saved_feats[i] = avg_hids[i]
                saved_toks[i] = pred_toks[i]
            shapes = {'feature': avg_hids.size(), 'token': pred_toks.size()}
            with open(os.path.join(self.save_dir, 'shapes.json'), 'w') as f:
                json.dump(shapes, f)

        if os.path.exists(os.path.join(self.cfg.outdir, 'tmp_feat.memmap')):
            os.remove(os.path.join(self.cfg.outdir, 'tmp_feat.memmap'))

    def _get_models(self):
        for client_id in range(1, self.num_clients + 1):
            self.models[client_id] = self._load_model(ATCModel(self.cfg.model),
                                                      client_id)

    def _get_avg_lm_head(self):
        all_params = copy.deepcopy([
            self.models[k].lm_head.state_dict()
            for k in range(1, self.num_clients + 1)
        ])
        avg_param = all_params[0]
        for k in avg_param:
            for i in range(len(all_params)):
                local_param = all_params[i][k].float()
                if i == 0:
                    avg_param[k] = local_param / len(all_params)
                else:
                    avg_param[k] += local_param / len(all_params)
        avg_lm_head = copy.deepcopy(self.models[1].lm_head)
        avg_lm_head.load_state_dict(avg_param)
        return avg_lm_head

    def _load_model(self, model, client_id):
        global_dir = os.path.join(self.pretrain_dir, 'global')
        client_dir = os.path.join(self.pretrain_dir, 'client')
        global_ckpt_path = os.path.join(global_dir,
                                        'global_model_{}.pt'.format(client_id))
        client_ckpt_path = os.path.join(client_dir,
                                        'client_model_{}.pt'.format(client_id))
        if os.path.exists(global_ckpt_path):
            model_ckpt = model.state_dict()
            logger.info('Loading model from \'{}\''.format(global_ckpt_path))
            global_ckpt = torch.load(global_ckpt_path,
                                     map_location='cpu')['model']
            model_ckpt.update(global_ckpt)
            if os.path.exists(client_ckpt_path):
                logger.info(
                    'Updating model from \'{}\''.format(client_ckpt_path))
                client_ckpt = torch.load(client_ckpt_path,
                                         map_location='cpu')['model']
                model_ckpt.update(client_ckpt)
            model.load_state_dict(model_ckpt)
        return model
