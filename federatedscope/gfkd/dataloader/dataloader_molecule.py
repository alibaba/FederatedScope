from torch_geometric import transforms
from torch_geometric.datasets import TUDataset, MoleculeNet, QM7b
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
m = Chem.MolFromSmiles('c1ccccc1')
m3d=Chem.AddHs(m)
AllChem.EmbedMolecule(m3d, randomSeed=1)



from federatedscope.core.auxiliaries.transform_builder import get_transform


def load_heteromolecule_dataset(config=None):
    r"""Convert dataset to Dataloader.
    :returns:
         data_local_dict
    :rtype: Dict {
                  'client_id': {
                      'train': DataLoader(),
                      'val': DataLoader(),
                      'test': DataLoader()
                               }
                  }
    """
    splits = config.data.splits
    path = config.data.root
    name = config.data.type.upper()

    # Transforms
    transforms_funcs = get_transform(config, 'torch_geometric')

    if name.startswith('heterogeneous molecule dataset'.upper()):
        dataset = []

        TUDdataset_names = ['BZR', 'ENZYMES', 'MUTAG']
        MoleculeNet_names = ['ESOL', 'FreeSolv', 'BACE']
        for dname in TUDdataset_names:
            tmp_dataset = TUDataset(path, dname, **transforms_funcs)
            dataset.append(tmp_dataset)
        for dname in MoleculeNet_names:
            tmp_dataset = MoleculeNet(path, dname, **transforms_funcs)
            
            if dname in ['FreeSolv', 'BACE']:
                for i in len(tmp_dataset):
                    smiles = dataset[i].smiles
                    mol = Chem.MolFromSmiles(smiles)
                    mol = AllChem.AddHs(mol)
                    res = AllChem.EmbedMolecule(mol, randomSeed=1)
                    # will random generate conformer with seed equal to -1. else fixed random seed.
                    if res == 0:
                        try:
                            AllChem.MMFFOptimizeMolecule(mol)# some conformer can not use MMFF optimize
                        except:
                            pass
                        mol = AllChem.RemoveHs(mol)
                        coordinates = mol.GetConformer().GetPositions()

                    elif res == -1:
                        mol_tmp = Chem.MolFromSmiles(smiles)
                        AllChem.EmbedMolecule(mol_tmp, maxAttempts=5000, randomSeed=1)
                        mol_tmp = AllChem.AddHs(mol_tmp, addCoords=True)
                        try:
                            AllChem.MMFFOptimizeMolecule(mol_tmp)# some conformer can not use MMFF optimize
                        except:
                            pass
                        mol_tmp = AllChem.RemoveHs(mol_tmp)
                        coordinates = mol_tmp.GetConformer().GetPositions()

                    assert dataset[i].x.shape[0] == len(coordinates), "coordinates shape is not align with {}".format(smiles)
                    tmp_dataset[i] = [tmp_dataset[i], coordinates]
            dataset.append(tmp_dataset)
        tmp_dataset = QM7b(path, dname, **transforms_funcs)
        dataset.append(tmp_dataset)
    else:
        raise ValueError(f'No dataset named: {name}!')

    client_num = min(len(dataset), config.federate.client_num
                     ) if config.federate.client_num > 0 else len(dataset)
    config.merge_from_list(['federate.client_num', client_num])

    # get local dataset
    data_dict = dict()
    for client_idx in range(1, len(dataset) + 1):
        data_dict[client_idx] = dataset[client_idx - 1]
    return data_dict, config
