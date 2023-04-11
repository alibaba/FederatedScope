from federatedscope.register import register_trainer
from federatedscope.core.trainers import GeneralTorchTrainer


class CVTrainer(GeneralTorchTrainer):
    pass


def call_cv_trainer(trainer_type):
    if trainer_type == 'cvtrainer':
        trainer_builder = CVTrainer
        return trainer_builder


register_trainer('cvtrainer', call_cv_trainer)
