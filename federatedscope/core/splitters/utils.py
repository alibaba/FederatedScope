import numpy as np


def get_splitter(config):
    client_num = config.federate.client_num
    args = eval(config.data.splitter_args)

    # Delay import
    if config.data.splitter == 'LDA':
        from federatedscope.core.splitters import LDASplitter
        splitter = LDASplitter(client_num, **args)
    else:
        splitter = None

    return splitter


def dirichlet_distribution_noniid_slice(label, client_num, alpha):
    r"""Get sample index list for each client from the Dirichlet distribution.
    https://github.com/FedML-AI/FedML/blob/master/fedml_core/non_iid_partition/noniid_partition.py

    Arguments:
        label (np.array): Label list to be split.
        client_num (int): Split label into client_num parts.
        alpha (float): alpha of LDA.
    Returns:
        idx_slice (List): List of splited label index slice.
    """
    if len(label.shape) != 1:
        raise ValueError('Only support single-label tasks!')
    num = len(label)
    classes = len(np.unique(label))
    # min number of sample in each client
    min_size = 0
    while min_size < 10:
        idx_slice = [[] for _ in range(client_num)]
        for k in range(classes):
            # for label k
            idx_k = np.where(label == k)[0]
            np.random.shuffle(idx_k)
            prop = np.random.dirichlet(np.repeat(alpha, client_num))
            prop = np.array([
                p * (len(idx_j) < num / client_num)
                for p, idx_j in zip(prop, idx_slice)
            ])
            prop = prop / sum(prop)
            prop = (np.cumsum(prop) * len(idx_k)).astype(int)[:-1]
            idx_slice = [
                idx_j + idx.tolist()
                for idx_j, idx in zip(idx_slice, np.split(idx_k, prop))
            ]
            min_size = min([len(idx_j) for idx_j in idx_slice])
    for i in range(client_num):
        np.random.shuffle(idx_slice[i])
    return idx_slice