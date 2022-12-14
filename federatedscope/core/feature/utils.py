import logging
import numpy as np

logger = logging.getLogger(__name__)


def merge_splits_feat(data):
    """

    Args:
        data: ``federatedscope.core.data.ClientData`` with Tabular format.

    Returns:
        Merged data feature/x.
    """
    merged_feat = None
    merged_y = None
    for split in ['train_data', 'val_data', 'test_data']:
        if hasattr(data, split):
            split_data = getattr(data, split)
            if split_data is not None and 'x' in split_data:
                if merged_feat is None:
                    merged_feat = split_data['x']
                else:
                    merged_feat = \
                        np.concatenate((merged_feat, split_data['x']), axis=0)
            if split_data is not None and 'y' in split_data:
                if merged_y is None:
                    merged_y = split_data['y']
                else:
                    merged_y = \
                        np.concatenate((merged_y, split_data['y']), axis=0)
    return merged_feat, merged_y


def vfl_binning(feat, num_bins, strategy='uniform'):
    """

    Args:
        feat: feature to be binned, which must be 2D numpy.array
        num_bins: list for bins
        strategy: binning strategy, ``'uniform'`` or ``'quantile'``

    Returns:
        Bin edges for binning
    """
    num_features = feat.shape[1]
    bin_edges = np.zeros(num_features, dtype=object)

    for i in range(num_features):
        col = feat[:, i]
        col_min, col_max = np.min(col), np.max(col)
        if col_min == col_max:
            logger.warning(
                f'Feature {i} is constant and will be replaced with 0.')
            bin_edges[i] = np.array([-np.inf, np.inf])
            continue
        if strategy == 'uniform':
            bin_edges[i] = np.linspace(col_min, col_max, num_bins[i] + 1)
        elif strategy == 'quantile':
            quantiles = np.linspace(0, 100, num_bins[i] + 1)
            bin_edges[i] = np.asarray(np.percentile(col, quantiles))

    return bin_edges


def secure_builder(cfg):
    if cfg.feat_engr.secure.type == 'encrypt':
        if cfg.feat_engr.secure.encrypt.type == 'dummy':
            from federatedscope.core.secure.encrypt.dummy_encrypt import \
                DummyEncryptKeypair
            keypair_generator = DummyEncryptKeypair(
                cfg.feat_engr.secure.key_size)
        else:
            raise NotImplementedError(f'Not implemented encrypt method'
                                      f' {cfg.feat_engr.secure.encrypt.type}.')
        return keypair_generator
    else:
        raise NotImplementedError(f'Not implemented secure method'
                                  f' {cfg.feat_engr.secure.type}.')
