import logging
import numpy as np

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

logger = logging.getLogger(__name__)

RDLogger.DisableLog('rdApp.*')


def generate_scaffold(smiles, include_chirality=False):
    """return scaffold string of target molecule"""
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold\
        .MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold


def lda(dataset, scaffold_idxs, client_num, alpha=0.1):
    # TODO:alpha在哪设置？
    """Obtain sample index list for each client from the Dirichlet distribution.

        This LDA method is first proposed by :
        Measuring the Effects of Non-Identical Data Distribution for
        Federated Visual Classification (https://arxiv.org/pdf/1909.06335.pdf).

        This can generate nonIIDness with unbalance sample number in each label.
        The Dirichlet distribution is a density over a K dimensional vector p whose K components are positive and sum to 1.
        Dirichlet can support the probabilities of a K-way categorical event.
        In FL, we can view K clients' sample number obeys the Dirichlet distribution.
        For more details of the Dirichlet distribution, please check https://en.wikipedia.org/wiki/Dirichlet_distribution
    """
    label_list = [dataset[i].y.item() for i in range(len(scaffold_idxs))]
    label_list = np.array(label_list)
    # TODO:For multiclass labels, the list is ragged and not a numpy array
    K = len(np.unique(label_list))
    N = label_list.shape[0]

    # guarantee the minimum number of sample in each client
    min_size = 0
    while min_size < 10:
        idx_batch = [[] for _ in range(client_num)]
        # for each classification in the dataset
        for k in range(K):
            # get a list of batch indexes which are belong to label k
            idx_k = scaffold_idxs[np.where(label_list == k)[0]]
            idx_batch, min_size = partition_class_samples_with_dirichlet_distribution(N, alpha, client_num,
                                                                                      idx_batch, idx_k)
    for i in range(client_num):
        np.random.shuffle(idx_batch[i])

    return idx_batch


def partition_class_samples_with_dirichlet_distribution(N, alpha, client_num, idx_batch, idx_k):
    np.random.shuffle(idx_k)
    # using dirichlet distribution to determine the unbalanced proportion for each client (client_num in total)
    # e.g., when client_num = 4, proportions = [0.29543505 0.38414498 0.31998781 0.00043216], sum(proportions) = 1
    proportions = np.random.dirichlet(np.repeat(alpha, client_num))

    # get the index in idx_k according to the dirichlet distribution
    proportions = np.array([p * (len(idx_j) < N / client_num) for p, idx_j in zip(proportions, idx_batch)])
    proportions = proportions / proportions.sum()
    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

    # generate the batch list for each client
    idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
    min_size = min([len(idx_j) for idx_j in idx_batch])

    return idx_batch, min_size


def gen_scaffold_lda_split(dataset, client_num=5):
    r"""
    return dict{ID:[idxs]}
    """
    logger.info('Scaffold split might take minutes, please wait...')
    scaffolds = {}
    for idx, data in enumerate(dataset):
        smiles = data.smiles
        mol = Chem.MolFromSmiles(smiles)
        scaffold = generate_scaffold(smiles)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [idx]
        else:
            scaffolds[scaffold].append(idx)
    # Sort from largest to smallest scaffold sets
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_list = [
        list(scaffold_set)
        for (scaffold,
             scaffold_set) in sorted(scaffolds.items(),
                                     key=lambda x: (len(x[1]), x[1][0]),
                                     reverse=True)
    ]
    scaffold_idxs = sum(scaffold_list, [])
    # TODO:将split方法修改为LDA
    # Split data to list
    # idx_dict = {}
    # splits = np.array_split(scaffold_idxs, client_num)
    # return [splits[ID] for ID in range(client_num)]
    idx_batch = lda(dataset, scaffold_idxs, client_num)
    return idx_batch


# TODO:在dataloader_graph.py中将ScaffoldLdaSplitter加入到splitter中
class ScaffoldLdaSplitter:
    def __init__(self, client_num):
        self.client_num = client_num

    def __call__(self, dataset):
        r"""Split dataset with smiles string into scaffold split
        
        Arguments:
            dataset (List or PyG.dataset): The molecular datasets.
            
        Returns:
            data_list (List(List(PyG.data))): Splited dataset via scaffold split.
        """
        dataset = [ds for ds in dataset]
        idx_slice = gen_scaffold_lda_split(dataset)
        data_list = [[dataset[idx] for idx in idxs] for idxs in idx_slice]
        return data_list

    def __repr__(self):
        return f'{self.__class__.__name__}()'
