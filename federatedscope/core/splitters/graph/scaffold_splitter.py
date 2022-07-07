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


def gen_scaffold_split(dataset, client_num=5):
    r"""
    return dict{ID:[idxs]}
    """
    logger.info('Scaffold split might take minutes, please wait...')
    scaffolds = {}
    for idx, data in enumerate(dataset):
        smiles = data.smiles
        _ = Chem.MolFromSmiles(smiles)
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
    # Split data to list
    splits = np.array_split(scaffold_idxs, client_num)
    return [splits[ID] for ID in range(client_num)]


class ScaffoldSplitter:
    def __init__(self, client_num):
        self.client_num = client_num

    def __call__(self, dataset):
        r"""Split dataset with smiles string into scaffold split

        Arguments:
            dataset (List or PyG.dataset): The molecular datasets.

        Returns:
            data_list (List(List(PyG.data))): Splited dataset via scaffold
            split.
        """
        dataset = [ds for ds in dataset]
        idx_slice = gen_scaffold_split(dataset)
        data_list = [[dataset[idx] for idx in idxs] for idxs in idx_slice]
        return data_list

    def __repr__(self):
        return f'{self.__class__.__name__}()'
