from importlib import import_module


def get_transform(config, package):
    transform_funcs = {
        name: config.data[name] if config.data[name] else None
        for name in ['transform', 'target_transform', 'pre_transform']
    }

    if not any(transform_funcs.values()):
        # Transform are all None, do not import package and return dict with None value
        return transform_funcs

    transforms = getattr(import_module(package), 'transforms')
    for key in transform_funcs:
        # return composed transform or return list of transform
        transform_funcs[key] = eval(
            transform_funcs[key]) if transform_funcs[key] else None
        if isinstance(transform_funcs[key], tuple):
            try:
                transform_funcs[key] = transforms.Compose(transform_funcs[key])
            except AttributeError:
                continue
    return transform_funcs
