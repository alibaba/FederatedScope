import logging

import numpy as np

logger = logging.getLogger(__name__)


def dirichlet_distribution_noniid_slice(label, client_num, alpha, min_size=10):
    r"""Get sample index list for each client from the Dirichlet distribution.
    https://github.com/FedML-AI/FedML/blob/master/fedml_core/non_iid_partition/noniid_partition.py

    Arguments:
        label (np.array): Label list to be split.
        client_num (int): Split label into client_num parts.
        alpha (float): alpha of LDA.
        min_size (int): min number of sample in each client
    Returns:
        idx_slice (List): List of splited label index slice.
    """
    if len(label.shape) != 1:
        raise ValueError('Only support single-label tasks!')
    num = len(label)
    classes = len(np.unique(label))
    assert num > client_num * min_size, f'The number of sample should be greater than {client_num *min_size}.'
    size = 0
    tried_time = 0
    while size < min_size:
        idx_slice = [[] for _ in range(client_num)]
        tried_time += 1
        if tried_time > 50:
            logger.warning(
                f"In the dirichlet non.i.i.d. split, we tried {tried_time} times but still do not fulfill the min_size requirement with {min_size}, Please try to increase the min_size or consider other splitter. "
            )
            if tried_time > 60:
                logger.warning(f"Too many tried times, we stop the trying")
                break
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
            size = min([len(idx_j) for idx_j in idx_slice])

    for i in range(client_num):
        np.random.shuffle(idx_slice[i])

    return idx_slice
