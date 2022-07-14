class MODE:
    """

    Note:
        Currently StrEnum cannot be imported with the environment
        `sys.version_info < (3, 11)`, so we simply create a MODE class here.
    """
    TRAIN = 'train'
    TEST = 'test'
    VAL = 'val'
    FINETUNE = 'finetune'


class TRIGGER:
    ON_FIT_START = 'on_fit_start'
    ON_EPOCH_START = 'on_epoch_start'
    ON_BATCH_START = 'on_batch_start'
    ON_BATCH_FORWARD = 'on_batch_forward'
    ON_BATCH_BACKWARD = 'on_batch_backward'
    ON_BATCH_END = 'on_batch_end'
    ON_EPOCH_END = 'on_epoch_end'
    ON_FIT_END = 'on_fit_end'

    @classmethod
    def contains(cls, item):
        return item in [
            "on_fit_start", "on_epoch_start", "on_batch_start",
            "on_batch_forward", "on_batch_backward", "on_batch_end",
            "on_epoch_end", "on_fit_end"
        ]


class LIFECYCLE:
    ROUTINE = 'routine'
    EPOCH = 'epoch'
    BATCH = 'batch'
    NONE = None
