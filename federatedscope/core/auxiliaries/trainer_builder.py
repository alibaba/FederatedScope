import logging
import importlib

import federatedscope.register as register

logger = logging.getLogger(__name__)

try:
    from federatedscope.contrib.trainer import *
except ImportError as error:
    logger.warning(
        f'{error} in `federatedscope.contrib.trainer`, some modules are not '
        f'available.')

TRAINER_CLASS_DICT = {
    "cvtrainer": "CVTrainer",
    "nlptrainer": "NLPTrainer",
    "graphminibatch_trainer": "GraphMiniBatchTrainer",
    "linkfullbatch_trainer": "LinkFullBatchTrainer",
    "linkminibatch_trainer": "LinkMiniBatchTrainer",
    "nodefullbatch_trainer": "NodeFullBatchTrainer",
    "nodeminibatch_trainer": "NodeMiniBatchTrainer",
    "flitplustrainer": "FLITPlusTrainer",
    "flittrainer": "FLITTrainer",
    "fedvattrainer": "FedVATTrainer",
    "fedfocaltrainer": "FedFocalTrainer",
    "mftrainer": "MFTrainer",
}


def get_trainer(model=None,
                data=None,
                device=None,
                config=None,
                only_for_eval=False,
                is_attacker=False,
                monitor=None):
    if config.trainer.type == 'general':
        if config.backend == 'torch':
            from federatedscope.core.trainers import GeneralTorchTrainer
            trainer = GeneralTorchTrainer(model=model,
                                          data=data,
                                          device=device,
                                          config=config,
                                          only_for_eval=only_for_eval,
                                          monitor=monitor)
        elif config.backend == 'tensorflow':
            from federatedscope.core.trainers import GeneralTFTrainer
            trainer = GeneralTFTrainer(model=model,
                                       data=data,
                                       device=device,
                                       config=config,
                                       only_for_eval=only_for_eval,
                                       monitor=monitor)
        else:
            raise ValueError
    elif config.trainer.type == 'none':
        return None
    elif config.trainer.type.lower() in TRAINER_CLASS_DICT:
        if config.trainer.type.lower() in ['cvtrainer']:
            dict_path = "federatedscope.cv.trainer.trainer"
        elif config.trainer.type.lower() in ['nlptrainer']:
            dict_path = "federatedscope.nlp.trainer.trainer"
        elif config.trainer.type.lower() in [
                'graphminibatch_trainer',
        ]:
            dict_path = "federatedscope.gfl.trainer.graphtrainer"
        elif config.trainer.type.lower() in [
                'linkfullbatch_trainer', 'linkminibatch_trainer'
        ]:
            dict_path = "federatedscope.gfl.trainer.linktrainer"
        elif config.trainer.type.lower() in [
                'nodefullbatch_trainer', 'nodeminibatch_trainer'
        ]:
            dict_path = "federatedscope.gfl.trainer.nodetrainer"
        elif config.trainer.type.lower() in [
                'flitplustrainer', 'flittrainer', 'fedvattrainer',
                'fedfocaltrainer'
        ]:
            dict_path = "federatedscope.gfl.flitplus.trainer"
        elif config.trainer.type.lower() in ['mftrainer']:
            dict_path = "federatedscope.mf.trainer.trainer"
        else:
            raise ValueError

        trainer_cls = getattr(importlib.import_module(name=dict_path),
                              TRAINER_CLASS_DICT[config.trainer.type.lower()])
        trainer = trainer_cls(model=model,
                              data=data,
                              device=device,
                              config=config,
                              only_for_eval=only_for_eval,
                              monitor=monitor)
    else:
        # try to find user registered trainer
        trainer = None
        for func in register.trainer_dict.values():
            trainer_cls = func(config.trainer.type)
            if trainer_cls is not None:
                trainer = trainer_cls(model=model,
                                      data=data,
                                      device=device,
                                      config=config,
                                      only_for_eval=only_for_eval,
                                      monitor=monitor)
        if trainer is None:
            raise ValueError('Trainer {} is not provided'.format(
                config.trainer.type))

    # differential privacy plug-in
    if config.nbafl.use:
        from federatedscope.core.trainers import wrap_nbafl_trainer
        trainer = wrap_nbafl_trainer(trainer)
    if config.sgdmf.use:
        from federatedscope.mf.trainer import wrap_MFTrainer
        trainer = wrap_MFTrainer(trainer)

    # personalization plug-in
    if config.federate.method.lower() == "pfedme":
        from federatedscope.core.trainers import wrap_pFedMeTrainer
        # wrap style: instance a (class A) -> instance a (class A)
        trainer = wrap_pFedMeTrainer(trainer)
    elif config.federate.method.lower() == "ditto":
        from federatedscope.core.trainers import wrap_DittoTrainer
        # wrap style: instance a (class A) -> instance a (class A)
        trainer = wrap_DittoTrainer(trainer)
    elif config.federate.method.lower() == "fedem":
        from federatedscope.core.trainers import FedEMTrainer
        # copy construct style: instance a (class A) -> instance b (class B)
        trainer = FedEMTrainer(model_nums=config.model.model_num_per_trainer,
                               base_trainer=trainer)

    # attacker plug-in
    if is_attacker:
        logger.info(
            '---------------- This client is an attacker --------------------')
        from federatedscope.attack.auxiliary.attack_trainer_builder import \
            wrap_attacker_trainer
        trainer = wrap_attacker_trainer(trainer, config)

    # fed algorithm plug-in
    if config.fedprox.use:
        from federatedscope.core.trainers import wrap_fedprox_trainer
        trainer = wrap_fedprox_trainer(trainer)

    return trainer
