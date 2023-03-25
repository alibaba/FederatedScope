import torch


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        self.val = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def merge_param_dict(raw_param, filtered_param):
    for key in filtered_param.keys():
        if key not in raw_param:
            continue
        raw_sz, merge_sz = raw_param[key].size(), filtered_param[key].size()
        if raw_sz == merge_sz:
            raw_param[key] = filtered_param[key]
        else:
            # print(f'skip params with different size: '
            #       f'{key} ({merge_sz} vs. {raw_sz})')
            if 'prefix_encoder' in key:
                assert len(raw_sz) == len(merge_sz) == 2 and \
                       raw_sz[0] == merge_sz[0]
                if raw_sz[1] > merge_sz[1]:  # merge from client to server
                    raw_param[key] = filtered_param[key].repeat(
                        1, raw_sz[1] // merge_sz[1])
                else:  # merge from server to client
                    raw_param[key] = torch.stack(filtered_param[key].split(
                        raw_sz[1], dim=-1)).mean(0)

    return raw_param
