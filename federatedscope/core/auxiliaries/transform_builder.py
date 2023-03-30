from importlib import import_module
import federatedscope.register as register


def get_transform(config, package):
    """
    This function is to build transforms applying to dataset.

    Args:
        config: ``CN`` from ``federatedscope/core/configs/config.py``
        package: one of package from \
        ``['torchvision', 'torch_geometric', 'torchtext', 'torchaudio']``

    Returns:
        Dict of transform functions.
    """
    transform_funcs = {}
    for name in ['transform', 'target_transform', 'pre_transform']:
        if config.data[name]:
            transform_funcs[name] = config.data[name]

    val_transform_funcs = {}
    for name in ['val_transform', 'val_target_transform', 'val_pre_transform']:
        suf_name = name.split('val_')[1]
        if config.data[name]:
            val_transform_funcs[suf_name] = config.data[name]

    test_transform_funcs = {}
    for name in [
            'test_transform', 'test_target_transform', 'test_pre_transform'
    ]:
        suf_name = name.split('test_')[1]
        if config.data[name]:
            test_transform_funcs[suf_name] = config.data[name]

    # Transform are all `[]`, do not import package and return dict with
    # None value
    if len(transform_funcs) == 0 and len(val_transform_funcs) == 0 and len(
            test_transform_funcs) == 0:
        return {}, {}, {}

    transforms = getattr(import_module(package), 'transforms')

    def convert(trans):
        # Recursively converting expressions to functions
        if isinstance(trans[0], str):
            if len(trans) == 1:
                trans.append({})
            transform_type, transform_args = trans
            for func in register.transform_dict.values():
                transform_func = func(transform_type, transform_args)
                if transform_func is not None:
                    return transform_func
            transform_func = getattr(transforms,
                                     transform_type)(**transform_args)
            return transform_func
        else:
            transform = [convert(x) for x in trans]
            if hasattr(transforms, 'Compose'):
                return transforms.Compose(transform)
            elif hasattr(transforms, 'Sequential'):
                return transforms.Sequential(transform)
            else:
                return transform

    # return composed transform or return list of transform
    if transform_funcs:
        for key in transform_funcs:
            transform_funcs[key] = convert(config.data[key])

    if val_transform_funcs:
        for key in val_transform_funcs:
            val_transform_funcs[key] = convert(config.data[key])
    else:
        val_transform_funcs = transform_funcs

    if test_transform_funcs:
        for key in test_transform_funcs:
            test_transform_funcs[key] = convert(config.data[key])
    else:
        test_transform_funcs = transform_funcs

    return transform_funcs, val_transform_funcs, test_transform_funcs
