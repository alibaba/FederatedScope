import os
import logging
import random
import csv
import json
import gzip
import zipfile
import shutil
from federatedscope.core.auxiliaries.utils import download_url

HFL_NAMES = ['imdb', 'agnews', 'squad', 'newsqa', 'cnndm', 'msqg']
logger = logging.getLogger(__name__)


class HFLDataProcessor(object):
    def __init__(self, config, train_frac=0.9):
        self.data_dir = config.data.root
        self.datasets = config.data.datasets
        self.num_grouped_clients = config.data.num_grouped_clients
        self.train_frac = train_frac
        self.all_train_data = []
        self.all_val_data = []
        self.all_test_data = []

    def get_data(self):
        for i, dataset in enumerate(self.datasets):
            if dataset not in HFL_NAMES:
                raise ValueError(f'No HFL dataset named {dataset}')
            train_val_data = self._load_data(dataset, 'train',
                                             self.num_grouped_clients[i])
            train_data = [
                data[:int(self.train_frac * len(data))]
                for data in train_val_data
            ]
            val_data = [
                data[int(self.train_frac * len(data)):]
                for data in train_val_data
            ]
            test_data = self._load_data(dataset, 'test',
                                        self.num_grouped_clients[i])
            self.all_train_data.extend(train_data)
            self.all_val_data.extend(val_data)
            self.all_test_data.extend(test_data)

        return self.all_train_data, self.all_val_data, self.all_test_data

    def _load_data(self, dataset, split, num_clients):
        data_dir = os.path.join(self.data_dir, dataset)
        if not os.path.exists(data_dir):
            self._download(dataset)
            self._extract(dataset)

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
        all_split_data = []
        n = len(data) // num_clients
        data_idx = 0
        for i in range(num_clients):
            num_split = n if i < num_clients - 1 else \
                len(data) - n * (num_clients - 1)
            cur_data = data[data_idx:data_idx + num_split]
            data_idx += num_split
            all_split_data.append(cur_data)
            logger.info(f'Client id: {len(self.all_train_data) + i + 1}, '
                        f'num samples: {num_split}')
        return all_split_data

    def _download(self, dataset):
        url = 'https://federatedscope.oss-cn-beijing.aliyuncs.com'
        os.makedirs(self.data_dir, exist_ok=True)
        download_url(f'{url}/{dataset}.zip', self.data_dir)

    def _extract(self, dataset):
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
