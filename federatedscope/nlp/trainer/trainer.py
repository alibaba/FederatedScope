from federatedscope.register import register_trainer
from federatedscope.core.trainers.trainer import GeneralTorchTrainer


class NLPTrainer(GeneralTorchTrainer):
    pass


def call_nlp_trainer(trainer_type):
    if trainer_type == 'nlptrainer':
        trainer_builder = NLPTrainer
        return trainer_builder


register_trainer('nlptrainer', call_nlp_trainer)
