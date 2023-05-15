import logging
import importlib

import federatedscope.register as register
from federatedscope.core.trainers import Trainer

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
    "cltrainer": "CLTrainer",
    "lptrainer": "LPTrainer",
    "atc_trainer": "ATCTrainer",
    "llmtrainer": "LLMTrainer"
}


def get_trainer(model=None,
                data=None,
                device=None,
                config=None,
                only_for_eval=False,
                is_attacker=False,
                monitor=None):
    """
    This function builds an instance of trainer.

    Arguments:
        model: model used in FL course
        data: data used in FL course
        device: where to train model (``cpu`` or ``gpu``)
        config: configurations for FL, see ``federatedscope.core.configs``
        only_for_eval: ``True`` or ``False``, if ``True``, ``train`` \
        routine will be removed in this trainer
        is_attacker: ``True`` or ``False`` to determine whether this client \
        is an attacker
        monitor: an instance of ``federatedscope.core.monitors.Monitor`` to \
        observe the evaluation and system metrics

    Returns:
        An instance of trainer.

    Note:
      The key-value pairs of ``cfg.trainer.type`` and trainers:
        ==================================  ===========================
        Trainer Type                        Source
        ==================================  ===========================
        ``general``                         \
        ``core.trainers.GeneralTorchTrainer`` and \
        ``core.trainers.GeneralTFTrainer``
        ``cvtrainer``                       ``cv.trainer.trainer.CVTrainer``
        ``nlptrainer``                      ``nlp.trainer.trainer.NLPTrainer``
        ``graphminibatch_trainer``          \
        ``gfl.trainer.graphtrainer.GraphMiniBatchTrainer``
        ``linkfullbatch_trainer``           \
        ``gfl.trainer.linktrainer.LinkFullBatchTrainer``
        ``linkminibatch_trainer``           \
        ``gfl.trainer.linktrainer.LinkMiniBatchTrainer``
        ``nodefullbatch_trainer``           \
        ``gfl.trainer.nodetrainer.NodeFullBatchTrainer``
        ``nodeminibatch_trainer``           \
        ``gfl.trainer.nodetrainer.NodeMiniBatchTrainer``
        ``flitplustrainer``                 \
        ``gfl.flitplus.trainer.FLITPlusTrainer``
        ``flittrainer``                     \
        ``gfl.flitplus.trainer.FLITTrainer``
        ``fedvattrainer``                   \
        ``gfl.flitplus.trainer.FedVATTrainer``
        ``fedfocaltrainer``                 \
        ``gfl.flitplus.trainer.FedFocalTrainer``
        ``mftrainer``                       \
        ``federatedscope.mf.trainer.MFTrainer``
        ``mytorchtrainer``                  \
        ``contrib.trainer.torch_example.MyTorchTrainer``
        ==================================  ===========================
      Wrapper functions are shown below:
        ==================================  ===========================
        Wrapper Functions                   Source
        ==================================  ===========================
        ``nbafl``                           \
        ``core.trainers.wrap_nbafl_trainer``
        ``sgdmf``                           ``mf.trainer.wrap_MFTrainer``
        ``pfedme``                          \
        ``core.trainers.wrap_pFedMeTrainer``
        ``ditto``                           ``core.trainers.wrap_DittoTrainer``
        ``fedem``                           ``core.trainers.FedEMTrainer``
        ``fedprox``                         \
        ``core.trainers.wrap_fedprox_trainer``
        ``attack``                          \
        ``attack.trainer.wrap_benignTrainer`` and \
        ``attack.auxiliary.attack_trainer_builder.wrap_attacker_trainer``
        ==================================  ===========================
    """
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
        elif config.trainer.type.lower() in ['cltrainer', 'lptrainer']:
            dict_path = "federatedscope.cl.trainer.trainer"
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
        elif config.trainer.type.lower() in ['atc_trainer']:
            dict_path = "federatedscope.nlp.hetero_tasks.trainer"
        elif config.trainer.type.lower() in ['llmtrainer']:
            dict_path = "federatedscope.llm.trainer.trainer"
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
    elif config.trainer.type.lower() in ['verticaltrainer']:
        from federatedscope.vertical_fl.tree_based_models.trainer.utils \
            import get_vertical_trainer
        trainer = get_vertical_trainer(config=config,
                                       model=model,
                                       data=data,
                                       device=device,
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

    if not isinstance(trainer, Trainer):
        logger.warning(f'Hook-like plug-in functions cannot be enabled when '
                       f'using {trainer}. If you want use our wrapper '
                       f'functions for your trainer please consider '
                       f'inheriting from '
                       f'`federatedscope.core.trainers.Trainer` instead.')
        return trainer

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
    elif config.federate.method.lower() == "fedrep":
        from federatedscope.core.trainers import wrap_FedRepTrainer
        # wrap style: instance a (class A) -> instance a (class A)
        trainer = wrap_FedRepTrainer(trainer)

    # attacker plug-in
    if 'backdoor' in config.attack.attack_method:
        from federatedscope.attack.trainer import wrap_benignTrainer
        trainer = wrap_benignTrainer(trainer)

    if is_attacker:
        if 'backdoor' in config.attack.attack_method:
            logger.info('--------This client is a backdoor attacker --------')
        else:
            logger.info('-------- This client is an privacy attacker --------')
        from federatedscope.attack.auxiliary.attack_trainer_builder \
            import wrap_attacker_trainer
        trainer = wrap_attacker_trainer(trainer, config)

    elif 'backdoor' in config.attack.attack_method:
        logger.info(
            '----- This client is a benign client for backdoor attacks -----')

    # fed algorithm plug-in
    if config.fedprox.use:
        from federatedscope.core.trainers import wrap_fedprox_trainer
        trainer = wrap_fedprox_trainer(trainer)

    # different fine-tuning
    if config.finetune.before_eval and config.finetune.simple_tuning:
        from federatedscope.core.trainers import wrap_Simple_tuning_Trainer
        trainer = wrap_Simple_tuning_Trainer(trainer)

    return trainer
