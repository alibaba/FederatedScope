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
        raw_p, merge_p = raw_param[key], filtered_param[key]
        raw_sz, merge_sz = raw_p.size(), merge_p.size()
        if raw_sz == merge_sz:
            raw_param[key] = filtered_param[key]
        else:
            assert len(raw_sz) == len(merge_sz) == 2 and \
                   raw_sz[0] == merge_sz[0]
            if raw_sz[1] > merge_sz[1]:  # merge from client to server
                raw_param[key] = merge_p.repeat(1, raw_sz[1] // merge_sz[1])
            else:  # merge from server to client
                raw_param[key] = torch.stack(merge_p.split(raw_sz[1],
                                                           dim=-1)).mean(0)

    return raw_param
