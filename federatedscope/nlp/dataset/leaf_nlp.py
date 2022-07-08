import os
import random
import pickle
import json
import torch
import math

import os.path as osp

from tqdm import tqdm
from collections import defaultdict

from sklearn.model_selection import train_test_split

from federatedscope.core.auxiliaries.utils import save_local_data, download_url
from federatedscope.cv.dataset.leaf import LEAF
from federatedscope.nlp.dataset.utils import *


class LEAF_NLP(LEAF):
    """
    LEAF NLP dataset from

    leaf.cmu.edu

    Arguments:
        root (str): root path.
        name (str): name of dataset, ‘shakespeare’ or ‘xxx’.
        s_frac (float): fraction of the dataset to be used; default=0.3.
        tr_frac (float): train set proportion for each task; default=0.8.
        val_frac (float): valid set proportion for each task; default=0.0.
        transform: transform for x.
        target_transform: transform for y.

    """
    def __init__(self,
                 root,
                 name,
                 s_frac=0.3,
                 tr_frac=0.8,
                 val_frac=0.0,
                 seed=123,
                 transform=None,
                 target_transform=None):
        # TODO: remove twitter
        self.s_frac = s_frac
        self.tr_frac = tr_frac
        self.val_frac = val_frac
        self.seed = seed
        super(LEAF_NLP, self).__init__(root, name, transform, target_transform)
        files = os.listdir(self.processed_dir)
        files = [f for f in files if f.startswith('task_')]
        if len(files):
            # Sort by idx
            files.sort(key=lambda k: int(k[5:]))

            for file in files:
                train_data, train_targets = torch.load(
                    osp.join(self.processed_dir, file, 'train.pt'))
                test_data, test_targets = torch.load(
                    osp.join(self.processed_dir, file, 'test.pt'))
                self.data_dict[int(file[5:])] = {
                    'train': (train_data, train_targets),
                    'test': (test_data, test_targets)
                }
                if osp.exists(osp.join(self.processed_dir, file, 'val.pt')):
                    val_data, val_targets = torch.load(
                        osp.join(self.processed_dir, file, 'val.pt'))
                    self.data_dict[int(file[5:])]['val'] = (val_data,
                                                            val_targets)
        else:
            raise RuntimeError(
                'Please delete ‘processed’ folder and try again!')

    @property
    def raw_file_names(self):
        names = [f'{self.name}_all_data.zip']
        return names

    def download(self):
        # Download to `self.raw_dir`.
        url = 'https://federatedscope.oss-cn-beijing.aliyuncs.com'
        os.makedirs(self.raw_dir, exist_ok=True)
        for name in self.raw_file_names:
            download_url(f'{url}/{name}', self.raw_dir)

    def __getitem__(self, index):
        """
        Arguments:
            index (int): Index

        :returns:
            dict: {'train':[(text, target)],
                   'test':[(text, target)],
                   'val':[(text, target)]}
            where target is the target class.
        """
        text_dict = {}
        data = self.data_dict[index]
        for key in data:
            text_dict[key] = []
            texts, targets = data[key]
            for idx in range(targets.shape[0]):
                text = texts[idx]
                target = targets[idx]

                if self.transform is not None:
                    text = self.transform(text)

                if self.target_transform is not None:
                    target = self.target_transform(target)

                text_dict[key].append((text, target))

        return text_dict

    def tokenizer(self, data, targets):
        """
        TOKENIZER = {
            'shakespeare': {
                'x': word_to_indices,
                'y': letter_to_vec
            },
            'twitter': {
                'x': bag_of_words,
                'y': target_to_binary
            },
            'subreddit': {
                'x': token_to_ids,
                'y': token_to_ids
            }
        }
        """
        if self.name == 'shakespeare':
            data = [
                word_to_indices(re.sub(r"   *", r' ', raw_text))
                for raw_text in data
            ]
            targets = [letter_to_vec(raw_target) for raw_target in targets]

        elif self.name == 'twitter':
            # Loading bag of word embeddings
            with open(osp.join(self.raw_dir, 'embs.json'), 'r') as inf:
                embs = json.load(inf)
            id2word = embs['vocab']
            word2id = {v: k for k, v in enumerate(id2word)}
            # [ID, Date, Query, User, Content]
            data = [bag_of_words(raw_text[4], word2id) for raw_text in data]
            targets = [target_to_binary(raw_target) for raw_target in targets]

        elif self.name == 'subreddit':
            with open(osp.join(self.raw_dir, 'reddit_vocab.pck'), 'rb') as inf:
                vocab_file = pickle.load(inf)
            vocab = defaultdict(lambda: vocab_file['unk_symbol'])
            vocab.update(vocab_file['vocab'])

            data_x_by_seq, data_y_by_seq, mask_by_seq = [], [], []

            for c, l in zip(data, targets):
                data_x_by_seq.extend(c)
                data_y_by_seq.extend(l['target_tokens'])
                mask_by_seq.extend(l['count_tokens'])

            data, targets, _ = data_x_by_seq, data_y_by_seq, mask_by_seq

            data = token_to_ids(data, vocab)
            targets = token_to_ids(targets, vocab)
            # Next word prediction
            targets = [words[-1] for words in targets]

        return data, targets

    def process(self):
        raw_path = osp.join(self.raw_dir, "all_data")
        files = os.listdir(raw_path)
        files = [f for f in files if f.endswith('.json')]

        if self.name == 'subreddit':
            self.s_frac = 1.0

        n_tasks = math.ceil(len(files) * self.s_frac)
        random.shuffle(files)
        files = files[:n_tasks]

        print("Preprocess data (Please leave enough space)...")

        idx = 0
        reddit_idx = []
        for num, file in enumerate(tqdm(files)):
            with open(osp.join(raw_path, file), 'r') as f:
                raw_data = json.load(f)

            user_list = list(raw_data['user_data'].keys())
            n_tasks = math.ceil(len(user_list) * self.s_frac)
            random.shuffle(user_list)
            user_list = user_list[:n_tasks]
            for user in user_list:
                data, targets = raw_data['user_data'][user]['x'], raw_data[
                    'user_data'][user]['y']

                # Filter the user within 50 contents
                if self.name == 'twitter' and len(data) <= 50:
                    continue
                if self.name == 'subreddit':
                    if user not in reddit_idx:
                        reddit_idx.append(user)

                # Tokenize
                data, targets = self.tokenizer(data, targets)

                if len(data) > 2:
                    data = torch.LongTensor(np.stack(data))
                    targets = torch.LongTensor(np.stack(targets))
                else:
                    data = torch.LongTensor(data)
                    targets = torch.LongTensor(targets)

                if self.name == 'subreddit':
                    # subreddit has fixed split
                    train_data, test_data, val_data = None, None, None
                    train_targets, test_targets, val_targets = None, None, None
                    if file.startswith('train'):
                        train_data = data
                        train_targets = targets
                    elif file.startswith('test'):
                        test_data = data
                        test_targets = targets
                    elif file.startswith('val'):
                        val_data = data
                        val_targets = targets
                    else:
                        continue
                    save_path = osp.join(self.processed_dir,
                                         f"task_{reddit_idx.index(user)}")
                else:
                    train_data, test_data, train_targets, test_targets =\
                        train_test_split(
                            data,
                            targets,
                            train_size=self.tr_frac,
                            random_state=self.seed
                        )

                    if self.val_frac > 0:
                        try:
                            val_data, test_data, val_targets, test_targets = \
                                train_test_split(
                                    test_data,
                                    test_targets,
                                    train_size=self.val_frac / (
                                            1.-self.tr_frac),
                                    random_state=self.seed
                                )
                        except:
                            val_data, val_targets = None, None

                    else:
                        val_data, val_targets = None, None
                    save_path = osp.join(self.processed_dir, f"task_{idx}")
                os.makedirs(save_path, exist_ok=True)

                save_local_data(dir_path=save_path,
                                train_data=train_data,
                                train_targets=train_targets,
                                test_data=test_data,
                                test_targets=test_targets,
                                val_data=val_data,
                                val_targets=val_targets)
                idx += 1
