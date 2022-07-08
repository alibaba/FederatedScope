import torch
import os
import numpy as np
from copy import deepcopy

from torch_geometric.data import InMemoryDataset, Data
from federatedscope.register import register_data

from rdkit import Chem
import torch_geometric.transforms as T

from torch_geometric.data import Dataset
from torch_geometric.datasets import TUDataset, MoleculeNet, QM9


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
        # data.y = torch.argmax(data.y).view(-1)

        return data


class GraphDTDataset(InMemoryDataset):
    def __init__(self, root):
        self.root = root
        super().__init__(root)

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'graph_dt', 'processed')

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
        np.random.seed(0)
        splits = [0.8, 0.1, 0.1]
        data_name_list = ['ESOL', 'FREESOLV', 'LIPO', 'BACE', 'BBBP', 'CLINTOX',
                          'MUTAG', 'PTC_MR', 'PTC_MM', 'PTC_FM', 'PTC_FR', 'NCI109', 'NCI1',
                          'alchemy_full', 'ZINC_full', 'QM9']

        for idx, name in enumerate(data_name_list):
            if name in ['ESOL', 'FREESOLV', 'LIPO', 'BACE', 'BBBP', 'CLINTOX']:
                dataset = MoleculeNet(self.root, name)
                featurizer = GenFeatures()
                ds = []
                for graph in dataset:
                    graph = featurizer(graph)
                    ds.append(Data(edge_attr=graph.edge_attr, edge_index=graph.edge_index, x=graph.x, y=graph.y))
                dataset = ds
            elif name == 'QM9':
                path_ = self.root + '/QM9/'
                dataset = QM9(path_)
                dataset = [Data(edge_attr=graph.edge_attr, edge_index=graph.edge_index, x=graph.x, y=graph.y) for graph in dataset]
            else:
                dataset = TUDataset(self.root, name)
                dataset = [Data(edge_attr=graph.edge_attr, edge_index=graph.edge_index, x=graph.x, y=graph.y) for graph in dataset]

            if name in ['ESOL']:
                save_idx = []
                for i in range(len(dataset)):
                    if dataset[i].edge_attr.shape[1] == 8:
                        save_idx.append(i)
                dataset = [dataset[i] for i in save_idx]
            if name in ['BACE', 'BBBP']:
                for i in range(len(dataset)):
                    dataset[i].y = dataset[i].y.long()
            if name in ['CLINTOX']:
                save_idx = []
                for i in range(len(dataset)):
                    dataset[i].y = torch.argmax(dataset[i].y).view(-1).unsqueeze(0)
                    if dataset[i].edge_attr.shape[1] == 8:
                        save_idx.append(i)
                        if dataset[i].y.item() == 1:
                            for j in range(9):
                                save_idx.append(i)
                dataset = [deepcopy(dataset[i]) for i in save_idx]
            if name in ['MUTAG', 'PTC_MR', 'PTC_MM', 'PTC_FM', 'NCI109', 'NCI1', 'ZINC_full']:
                for i in range(len(dataset)):
                    dataset[i].y = dataset[i].y.unsqueeze(0)
            if name in ['PTC_FR']:
                save_idx = []
                for i in range(len(dataset)):
                    dataset[i].y = dataset[i].y.unsqueeze(0)
                    save_idx.append(i)
                    if dataset[i].y.item() == 1:
                        for j in range(9):
                            save_idx.append(i)
                dataset = [deepcopy(dataset[i]) for i in save_idx]

            index = np.random.permutation(np.arange(len(dataset)))
            train_idx = index[:int(len(dataset) * splits[0])]
            valid_idx = index[int(len(dataset) * splits[0]):int(len(dataset) * sum(splits[:2]))]
            test_idx = index[int(len(dataset) * sum(splits[:2])):]
            pro_train = [dataset[i] for i in train_idx]
            pro_test = [dataset[i] for i in test_idx]
            pro_valid = [dataset[i] for i in valid_idx]

            if not os.path.isdir(os.path.join(self.processed_dir, str(idx))):
                os.makedirs(os.path.join(self.processed_dir, str(idx)))

            train_path = os.path.join(self.processed_dir, str(idx), 'train.pt')
            test_path = os.path.join(self.processed_dir, str(idx), 'test.pt')
            valid_path = os.path.join(self.processed_dir, str(idx), 'val.pt')

            if idx == 1:
                for pro_data, data_path in zip([pro_train, pro_test, pro_valid], [train_path, test_path, valid_path]):
                    save_idx = []
                    for i in range(len(pro_data)):
                        pro_data[i].y = np.log(-pro_data[i].y + 5)
                        if pro_data[i].edge_attr.shape[1] == 8:
                            save_idx.append(i)
                    pro_data = [pro_data[i] for i in save_idx]
                    torch.save(pro_data, data_path)
            elif idx == 13:
                mini = [0.05484385788440704, 3.590000051190145e-05, 0.036166608333587646, 5.850073337554932,
                        -0.3293163776397705, 5.849961757659912, 59.01787185668945, 5.849983215332031,
                        5.849958896636963, -0.2201365828514099, -3.4974498748779297, 6.7523512840271]
                std_ = [0.3041064292192459, 6.874400060041808e-05, 0.3704751431941986, 1.6029706001281738,
                        0.25437575578689575, 1.6030550003051758, 103.84016036987305, 1.6030397415161133,
                        1.6030573844909668, 0.32082635164260864, 5.663499355316162, 1.8080096244812012]
                for pro_data, data_path in zip([pro_train, pro_test, pro_valid], [train_path, test_path, valid_path]):
                    for i in range(len(pro_data)):
                        for cls in range(pro_data[0].y.shape[1]):
                            if cls in [3, 5, 7, 8]:
                                pro_data[i].y[0][cls] = np.log(-pro_data[i].y[0][cls])
                            elif cls in [10]:
                                pro_data[i].y[0][cls] = np.log(pro_data[i].y[0][cls])
                                if pro_data[i].y[0][cls].item() < -3.4974497878551483:
                                    pro_data[i].y[0][cls] = -3.4974497878551483
                                elif pro_data[i].y[0][cls].item() > 2.166049448490173:
                                    pro_data[i].y[0][cls] = 2.166049448490173
                            elif cls in [11]:
                                pro_data[i].y[0][cls] = np.log(pro_data[i].y[0][cls])
                            pro_data[i].y[0][cls] = (pro_data[i].y[0][cls] - mini[cls]) / std_[cls]
                    torch.save(pro_data, data_path)
            elif idx == 14:
                for pro_data, data_path in zip([pro_train, pro_test, pro_valid], [train_path, test_path, valid_path]):
                    for i in range(len(pro_data)):
                        pro_data[i].y = np.log(-pro_data[i].y + 5)
                    torch.save(pro_data, data_path)
            elif idx == 15:
                mini = [-13.815510749816895, 9.460000038146973, -11.662799835205078, -4.51708984375,
                        1.1428781747817993, 26.156299591064453, 0.4340488314628601, -18374.904296875,
                        -18374.71875, -18374.693359375, -18375.76953125, 6.2779998779296875,
                        -113.11315155029297, -113.89193725585938, -114.61173248291016, -104.81642150878906,
                        0.4717443585395813, -0.9457370042800903, -0.9673479199409485]
                std_ = [16.949286222457886, 134.0699987411499, 8.895401954650879, 9.782493114471436,
                        15.785325407981873, 3348.5968742370605, 7.0162652134895325, 17273.41650390625,
                        17273.308959960938, 17273.309326171875, 17273.74658203125, 40.68299865722656,
                        101.10729694366455, 101.8098087310791, 102.45245933532715, 93.57041645050049,
                        2.7117268443107605, 2.688384175300598, 2.444916546344757]
                for pro_data, data_path in zip([pro_train, pro_test, pro_valid], [train_path, test_path, valid_path]):
                    for i in range(len(pro_data)):
                        for cls in range(pro_data[0].y.shape[1]):
                            if cls == 0:
                                pro_data[i].y[0][cls] = np.log(pro_data[i].y[0][cls] + 0.000001)
                            elif cls == 16:
                                if pro_data[i].y[0][cls].item() < 1.6027875780463219:
                                    pro_data[i].y[0][cls] = 1.6027875780463219
                                elif pro_data[i].y[0][cls].item() > 24.13037023258204:
                                    pro_data[i].y[0][cls] = 24.13037023258204
                                pro_data[i].y[0][cls] = np.log(pro_data[i].y[0][cls])
                            elif cls == 17:
                                if pro_data[i].y[0][cls].item() < 0.3883932141512632:
                                    pro_data[i].y[0][cls] = 0.3883932141512632
                                elif pro_data[i].y[0][cls].item() > 5.712445258378982:
                                    pro_data[i].y[0][cls] = 5.712445258378982
                                pro_data[i].y[0][cls] = np.log(pro_data[i].y[0][cls])
                            elif cls == 18:
                                if pro_data[i].y[0][cls].item() < 0.38008972319960593:
                                    pro_data[i].y[0][cls] = 0.38008972319960593
                                elif pro_data[i].y[0][cls].item() > 4.3822778129577635:
                                    pro_data[i].y[0][cls] = 4.3822778129577635
                                pro_data[i].y[0][cls] = np.log(pro_data[i].y[0][cls])
                            pro_data[i].y[0][cls] = (pro_data[i].y[0][cls] - mini[cls]) / std_[cls]
                    torch.save(pro_data, data_path)
            else:
                torch.save(pro_train, train_path)
                torch.save(pro_test, test_path)
                torch.save(pro_valid, valid_path)

    def __getitem__(self, idx):
        data = {}
        for split in ['train', 'val', 'test']:
            split_data = self._load(idx, split)
            if split_data:
                data[split] = split_data
        return data


def load_graph_dt_data(config):
    from torch_geometric.loader import DataLoader
    # from federatedscope.gfl.dataloader.dataloader_graph import get_numGraphLabels

    # Build data
    dataset = GraphDTDataset(config.data.root)
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


def call_graph_dt_data(config):
    if config.data.type == "graph-dt":
        data, modified_config = load_graph_dt_data(config)
        return data, modified_config


register_data("graph-dt", call_graph_dt_data)