import torch
from federatedscope.attack.auxiliary.utils import get_data_info


def get_target_data(dataset_name, pth=None):
    '''

    Args:
        dataset_name (str): the dataset name
        pth (str): the path storing the target data

    Returns:

    '''
    # JUST FOR SHOWCASE
    if pth is not None:
        pass
    else:
        # generate the synthetic data
        if dataset_name == 'femnist':
            data_feature_dim, num_class, is_one_hot_label = get_data_info(
                dataset_name)

            # generate random data
            num_syn_data = 20
            data_dim = [num_syn_data]
            data_dim.extend(data_feature_dim)
            syn_data = torch.randn(data_dim)
            syn_label = torch.randint(low=0,
                                      high=num_class,
                                      size=(num_syn_data, ))
            return [syn_data, syn_label]
