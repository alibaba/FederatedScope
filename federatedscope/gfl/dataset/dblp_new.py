import os.path as osp
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.utils import from_networkx
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS as \
    sklearn_stopwords


class LemmaTokenizer(object):
    def __init__(self):
        from nltk.stem import WordNetLemmatizer
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        from nltk import word_tokenize
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


def build_feature(words, threshold):
    from nltk.corpus import stopwords as nltk_stopwords
    # use bag-of-words representation of paper titles as the features of papers
    stopwords = sklearn_stopwords.union(set(nltk_stopwords.words('english')))
    vectorizer = CountVectorizer(min_df=int(threshold),
                                 stop_words=stopwords,
                                 tokenizer=LemmaTokenizer())
    features_paper = vectorizer.fit_transform(words)

    return features_paper


def build_graph(path, filename, FL=0, threshold=15):
    with open(osp.join(path, filename), 'r') as f:
        node_cnt = sum([1 for line in f])

    G = nx.DiGraph()
    desc = node_cnt * [None]
    neighbors = node_cnt * [None]
    if FL == 1:
        conf2paper = dict()
    elif FL == 2:
        org2paper = dict()

    # Build node feature from title
    with open(osp.join(path, filename), 'r') as f:
        for line in f:
            cols = line.strip().split('\t')
            nid, title = int(cols[0]), cols[3]
            desc[nid] = title

    features = np.array(build_feature(desc, threshold).todense(),
                        dtype=np.float32)

    # Build graph structure
    with open(osp.join(path, filename), 'r') as f:
        for line in f:
            cols = line.strip().split('\t')
            nid, conf, org, label = int(cols[0]), cols[1], cols[2], int(
                cols[4])
            neighbors[nid] = [int(val) for val in cols[-1].split(',')]

            if FL == 1:
                if conf not in conf2paper:
                    conf2paper[conf] = [nid]
                else:
                    conf2paper[conf].append(nid)
            elif FL == 2:
                if org not in org2paper:
                    org2paper[org] = [nid]
                else:
                    org2paper[org].append(nid)

            G.add_node(nid, y=label, x=features[nid], index_orig=nid)

    for nid, nbs in enumerate(neighbors):
        for vid in nbs:
            G.add_edge(nid, vid)

    # Sort node id for index_orig
    H = nx.Graph()
    H.add_nodes_from(sorted(G.nodes(data=True)))
    H.add_edges_from(G.edges(data=True))
    G = H
    graphs = []
    if FL == 1:
        for conf in conf2paper:
            graphs.append(from_networkx(nx.subgraph(G, conf2paper[conf])))
    elif FL == 2:
        for org in org2paper:
            graphs.append(from_networkx(nx.subgraph(G, org2paper[org])))
    else:
        graphs.append(from_networkx(G))

    return graphs


class DBLPNew(InMemoryDataset):
    r"""
    Args:
        root (string): Root directory where the dataset should be saved.
        FL (Bool): Federated setting, `0` for DBLP, `1` for FLDBLPbyConf,
        `2` for FLDBLPbyOrg
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """
    def __init__(self,
                 root,
                 FL=0,
                 splits=[0.5, 0.2, 0.3],
                 transform=None,
                 pre_transform=None):
        self.FL = FL
        if self.FL == 0:
            self.name = 'DBLPNew'
        elif self.FL == 1:
            self.name = 'FLDBLPbyConf'
        else:
            self.name = 'FLDBLPbyOrg'
        self._customized_splits = splits
        super(DBLPNew, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        names = ['dblp_new.tsv']
        return names

    @property
    def processed_file_names(self):
        return ['data.pt']

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    def download(self):
        # Download to `self.raw_dir`.
        url = 'https://federatedscope.oss-cn-beijing.aliyuncs.com'
        for name in self.raw_file_names:
            download_url(f'{url}/{name}', self.raw_dir)

    def process(self):
        # Read data into huge `Data` list.
        data_list = build_graph(self.raw_dir, self.raw_file_names[0], self.FL)

        data_list_w_masks = []
        for data in data_list:
            if data.num_nodes == 0:
                continue
            indices = torch.randperm(data.num_nodes)
            data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            data.train_mask[indices[:round(self._customized_splits[0] *
                                           len(data.y))]] = True
            data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            data.val_mask[
                indices[round(self._customized_splits[0] *
                              len(data.y)):round((self._customized_splits[0] +
                                                  self._customized_splits[1]) *
                                                 len(data.y))]] = True
            data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            data.test_mask[indices[round((self._customized_splits[0] +
                                          self._customized_splits[1]) *
                                         len(data.y)):]] = True
            data_list_w_masks.append(data)
        data_list = data_list_w_masks

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
