class BasicEnum(object):
    @classmethod
    def assert_value(cls, value):
        """
        Check if the **value** is legal for the given class (If the value equals one of the class attributes)
        """
        if not value in [v for k, v in cls.__dict__.items() if not k.startswith('__')]:
            raise ValueError(f"Value {value} is not in {cls.__name__}.")


class MODE(BasicEnum):
    """

    Note:
        Currently StrEnum cannot be imported with the environment
        `sys.version_info < (3, 11)`, so we simply create a MODE class here.
    """
    TRAIN = 'train'
    TEST = 'test'
    VAL = 'val'
    FINETUNE = 'finetune'


class STAGE(BasicEnum):
    TRAIN = 'train'
    EVAL = 'eval'
    CONSULT = 'consult'


class TRIGGER(BasicEnum):
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


class LIFECYCLE(BasicEnum):
    ROUTINE = 'routine'
    EPOCH = 'epoch'
    BATCH = 'batch'
    NONE = None


class CLIENT_STATE(BasicEnum):
    OFFLINE = -1    # not join in
    CONSULTING = 2  # join in and is consulting
    IDLE = 1        # join in but not working, available for training
    WORKING = 0     # join in and is working
    SIDELINE = 0    # join in but won't participate in federated training
