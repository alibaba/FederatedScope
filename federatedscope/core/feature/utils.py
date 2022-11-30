import numpy as np


def merge_splits_feat(data):
    merged_feat = None
    for split in ['train_data', 'val_data', 'test_data']:
        if hasattr(data, split):
            split_data = getattr(data, split)
            if split_data is not None and 'x' in split_data:
                if merged_feat is None:
                    merged_feat = split_data['x']
                else:
                    merged_feat = \
                        np.concatenate((merged_feat, split_data['x']), axis=0)
    if merged_feat is None:
        raise ValueError('Not support data type for merged feature.')
    return merged_feat
