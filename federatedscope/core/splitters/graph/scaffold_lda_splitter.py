import logging
import numpy as np
import torch

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from federatedscope.core.splitters.utils import \
    dirichlet_distribution_noniid_slice
from federatedscope.core.splitters.graph.scaffold_splitter import \
    generate_scaffold

logger = logging.getLogger(__name__)

RDLogger.DisableLog('rdApp.*')


class GenFeatures:
    r"""Implementation of 'CanonicalAtomFeaturizer' and
    'CanonicalBondFeaturizer' in DGL.
    Source: https://lifesci.dgl.ai/_modules/dgllife/utils/featurizers.html

    Arguments:
        data: PyG.data in PyG.dataset.

    Returns:
        data: PyG.data, data passing featurizer.

    """
    def __init__(self):
        self.symbols = [
            'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
            'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag',
            'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni',
            'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'other'
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
            Chem.rdchem.BondStereo.STEREOE,
            Chem.rdchem.BondStereo.STEREOCIS,
            Chem.rdchem.BondStereo.STEREOTRANS,
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
            degree = [0.] * 10
            degree[atom.GetDegree()] = 1.
            implicit = [0.] * 6
            implicit[atom.GetImplicitValence()] = 1.
            formal_charge = atom.GetFormalCharge()
            radical_electrons = atom.GetNumRadicalElectrons()
            hybridization = [0.] * len(self.hybridizations)
            if atom.GetHybridization() in self.hybridizations:
                hybridization[self.hybridizations.index(
                    atom.GetHybridization())] = 1.
            else:
                hybridization[self.hybridizations.index('other')] = 1.
            aromaticity = 1. if atom.GetIsAromatic() else 0.
            hydrogens = [0.] * 5
            hydrogens[atom.GetTotalNumHs()] = 1.

            x = torch.tensor(symbol + degree + implicit + [formal_charge] +
                             [radical_electrons] + hybridization +
                             [aromaticity] + hydrogens)
            xs.append(x)

        data.x = torch.stack(xs, dim=0)

        edge_attrs = []
        for bond in mol.GetBonds():
            bond_type = bond.GetBondType()
            single = 1. if bond_type == Chem.rdchem.BondType.SINGLE else 0.
            double = 1. if bond_type == Chem.rdchem.BondType.DOUBLE else 0.
            triple = 1. if bond_type == Chem.rdchem.BondType.TRIPLE else 0.
            aromatic = 1. if bond_type == Chem.rdchem.BondType.AROMATIC else 0.
            conjugation = 1. if bond.GetIsConjugated() else 0.
            ring = 1. if bond.IsInRing() else 0.
            stereo = [0.] * 6
            stereo[self.stereos.index(bond.GetStereo())] = 1.

            edge_attr = torch.tensor(
                [single, double, triple, aromatic, conjugation, ring] + stereo)

            edge_attrs += [edge_attr, edge_attr]

        if len(edge_attrs) == 0:
            data.edge_index = torch.zeros((2, 0), dtype=torch.long)
            data.edge_attr = torch.zeros((0, 10), dtype=torch.float)
        else:
            num_atoms = mol.GetNumAtoms()
            feats = torch.stack(edge_attrs, dim=0)
            feats = torch.cat([feats, torch.zeros(feats.shape[0], 1)], dim=1)
            self_loop_feats = torch.zeros(num_atoms, feats.shape[1])
            self_loop_feats[:, -1] = 1
            feats = torch.cat([feats, self_loop_feats], dim=0)
            data.edge_attr = feats

        return data


def gen_scaffold_lda_split(dataset, client_num=5, alpha=0.1):
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
    label = np.zeros(len(dataset))
    for i in range(len(scaffold_list)):
        label[scaffold_list[i]] = i + 1
    label = torch.LongTensor(label)
    # Split data to list
    idx_slice = dirichlet_distribution_noniid_slice(label, client_num, alpha)
    return idx_slice


class ScaffoldLdaSplitter:
    r"""First adopt scaffold splitting and then assign the samples to
    clients according to Latent Dirichlet Allocation.

    Arguments:
        dataset (List or PyG.dataset): The molecular datasets.
        alpha (float): Partition hyperparameter in LDA, smaller alpha
        generates more extreme heterogeneous scenario.

    Returns:
        data_list (List(List(PyG.data))): Splited dataset via scaffold split.

    """
    def __init__(self, client_num, alpha):
        self.client_num = client_num
        self.alpha = alpha

    def __call__(self, dataset):
        featurizer = GenFeatures()
        data = []
        for ds in dataset:
            ds = featurizer(ds)
            data.append(ds)
        dataset = data
        idx_slice = gen_scaffold_lda_split(dataset, self.client_num,
                                           self.alpha)
        data_list = [[dataset[idx] for idx in idxs] for idxs in idx_slice]
        return data_list

    def __repr__(self):
        return f'{self.__class__.__name__}()'
