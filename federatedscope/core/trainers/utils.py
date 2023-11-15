import collections
import json
import math


def format_log_hooks(hooks_set):
    def format_dict(target_dict):
        print_dict = collections.defaultdict(list)
        for k, v in target_dict.items():
            for element in v:
                print_dict[k].append(element.__name__)
        return print_dict

    if isinstance(hooks_set, list):
        print_obj = [format_dict(_) for _ in hooks_set]
    elif isinstance(hooks_set, dict):
        print_obj = format_dict(hooks_set)
    return json.dumps(print_obj, indent=2).replace('\n', '\n\t')


def filter_by_specified_keywords(param_name, filter_keywords):
    """
    Arguments:
        param_name (str): parameter name.
    Returns:
        preserve (bool): whether to preserve this parameter.
    """
    preserve = True
    for kw in filter_keywords:
        if kw in param_name:
            preserve = False
            break
    return preserve


def move_to(obj, device):
    import torch
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return res
    else:
        raise TypeError("Invalid type for move_to")


def get_random(dis_type, sample_shape, params, device):
    import torch.distributions as distributions
    if not hasattr(distributions, dis_type):
        raise NotImplementedError("Distribution {} is not implemented, "
                                  "please refer to ```torch.distributions```"
                                  "(https://pytorch.org/docs/stable/ "
                                  "distributions.html).".format(dis_type))
    generator = getattr(distributions, dis_type)(**params)
    return generator.sample(sample_shape=sample_shape).to(device)


def calculate_batch_epoch_num(steps, batch_or_epoch, num_data, batch_size,
                              drop_last):
    num_batch_per_epoch = num_data // batch_size + int(
        not drop_last and bool(num_data % batch_size))
    if num_batch_per_epoch == 0:
        raise RuntimeError(
            "The number of batch is 0, please check 'batch_size' or set "
            "'drop_last' as False")
    elif batch_or_epoch == "epoch":
        num_epoch = steps
        num_batch_last_epoch = num_batch_per_epoch
        num_total_batch = steps * num_batch_per_epoch
    else:
        num_batch_per_epoch = min(num_batch_per_epoch, steps)
        num_epoch = math.ceil(steps / num_batch_per_epoch)
        num_batch_last_epoch = steps % num_batch_per_epoch or \
            num_batch_per_epoch
        num_total_batch = steps
    return num_batch_per_epoch, num_batch_last_epoch, num_epoch, \
        num_total_batch
