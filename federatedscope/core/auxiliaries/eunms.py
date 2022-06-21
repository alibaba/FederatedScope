class MODE:
    """

    Note:
        Currently StrEnum cannot be imported with the environment `sys.version_info < (3, 11)`, so we simply create a
        MODE class here.
    """
    TRAIN = 'train'
    TEST = 'test'
    VAL = 'val'
    FINETUNE = 'finetune'
