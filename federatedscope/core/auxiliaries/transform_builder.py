from importlib import import_module


def get_transform(config, package):
    transform_funcs = {
        name: config.data[name] if config.data[name] else None
        for name in ['transform', 'target_transform', 'pre_transform']
    }

    # Transform are all None, do not import package and return dict with None value
    if not any(transform_funcs.values()):
        return transform_funcs

    transforms = getattr(import_module(package), 'transforms')

    def convert(transform):
        # Recursively converting expressions to functions
        if isinstance(transform[0], str):
            if len(transform) == 1:
                transform.append({})
            transform_type, transform_args = transform
            transform_func = getattr(transforms,
                                     transform_type)(**transform_args)
            return transform_func
        else:
            transform = (convert(x) for x in transform)
            if hasattr(transforms, 'Compose'):
                return transforms.Compose(transform)
            elif hasattr(transforms, 'Sequential'):
                return transforms.Sequential(transform)
            else:
                return transform

    # return composed transform or return list of transform
    for key in transform_funcs:
        if not config.data[key]:
            continue
        transform_funcs[key] = convert(config.data[key])
    return transform_funcs
