import torch
import os
import numpy as np
from copy import deepcopy

from torch_geometric.datasets import TUDataset, MoleculeNet, QM9
from torch_geometric.data import InMemoryDataset, Data
from federatedscope.register import register_data

from rdkit import Chem



class GenFeatures:
    def __init__(self):
        self.symbols = [
            'B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br',
            'Te', 'I', 'At', 'other'
        ]

        self.hybridizations = [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
            'other',
        ]

        self.stereos = [
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOANY,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE
        ]

    def __call__(self, data):
        mol = Chem.MolFromSmiles(data.smiles)

        xs = []
        for atom in mol.GetAtoms():
            symbol = [0.] * len(self.symbols)
            if atom.GetSymbol() in self.symbols:
                symbol[self.symbols.index(atom.GetSymbol())] = 1.
            else:
                symbol[self.symbols.index('other')] = 1.
            hybridization = [0.] * len(self.hybridizations)
            if atom.GetHybridization() in self.hybridizations:
                hybridization[self.hybridizations.index(atom.GetHybridization())] = 1.
            else:
                hybridization[self.hybridizations.index('other')] = 1.

            x = torch.tensor(symbol + hybridization)
            xs.append(x)

        data.x = torch.stack(xs, dim=0)

        edge_attrs = []
        for bond in mol.GetBonds():

            bond_type = bond.GetBondType()
            single = 1. if bond_type == Chem.rdchem.BondType.SINGLE else 0.
            double = 1. if bond_type == Chem.rdchem.BondType.DOUBLE else 0.
            triple = 1. if bond_type == Chem.rdchem.BondType.TRIPLE else 0.
            aromatic = 1. if bond_type == Chem.rdchem.BondType.AROMATIC else 0.
            stereo = [0.] * 4
            stereo[self.stereos.index(bond.GetStereo())] = 1.

            edge_attr = torch.tensor(
                [single, double, triple, aromatic] + stereo)

            edge_attrs += [edge_attr, edge_attr]

        if len(edge_attrs) == 0:
            data.edge_attr = torch.zeros((0, 10), dtype=torch.float)
        else:
            data.edge_attr = torch.stack(edge_attrs, dim=0)

        return data


class CIKMCupDataset(InMemoryDataset):
    name = 'CIKM_Cup'

    def __init__(self, root):
        super(CIKMCupDataset, self).__init__(root)

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, 'processed')

    @property
    def processed_file_names(self):
        return ['pre_transform.pt', 'pre_filter.pt']

    def __len__(self):
        return len([
            x for x in os.listdir(self.processed_dir)
            if not x.startswith('pre')
        ])

    def _load(self, idx, split):
        try:
            data = torch.load(
                os.path.join(self.processed_dir, str(idx), f'{split}.pt'))
        except:
            data = None
        return data

    def process(self):
        pass

    def __getitem__(self, idx):
        data = {}
        for split in ['train', 'val', 'test']:
            split_data = self._load(idx, split)
            if split_data:
                data[split] = split_data
        return data


def load_cikmcup_data(config):
    from torch_geometric.loader import DataLoader

    # Build data
    dataset = CIKMCupDataset(config.data.root)
    config.merge_from_list(['federate.client_num', len(dataset)])

    data_dict = {}
    # Build DataLoader dict
    for client_idx in range(1, config.federate.client_num + 1):
        dataloader_dict = {}
        tmp_dataset = []
        if 'train' in dataset[client_idx - 1]:
            dataloader_dict['train'] = DataLoader(dataset[client_idx -
                                                          1]['train'],
                                                  config.data.batch_size,
                                                  shuffle=config.data.shuffle)
            tmp_dataset += dataset[client_idx - 1]['train']
        if 'val' in dataset[client_idx - 1]:
            dataloader_dict['val'] = DataLoader(dataset[client_idx - 1]['val'],
                                                config.data.batch_size,
                                                shuffle=False)
            tmp_dataset += dataset[client_idx - 1]['val']
        if 'test' in dataset[client_idx - 1]:
            dataloader_dict['test'] = DataLoader(dataset[client_idx -
                                                         1]['test'],
                                                 config.data.batch_size,
                                                 shuffle=False)
            tmp_dataset += dataset[client_idx - 1]['test']
        if tmp_dataset:
            dataloader_dict['num_label'] = 0 # todo: set to 0, used in gfl/model_builder.py line74
            # dataloader_dict['num_label'] = get_numGraphLabels(tmp_dataset)
        data_dict[client_idx] = dataloader_dict

    return data_dict, config

#
# def call_cikmcup_data(config):
#     if config.data.type == "cikmcup":
#         data, modified_config = load_cikmcup_data(config)
#         return data, modified_config
#
#
# register_data("cikmcup", call_cikmcup_data)
