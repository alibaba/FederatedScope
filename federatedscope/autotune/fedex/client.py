import logging

from federatedscope.core.message import Message
from federatedscope.core.worker import Client
from federatedscope.core.auxiliaries.utils import formatted_logging

logger = logging.getLogger(__name__)


class FedExClient(Client):
    """Some code snippets are borrowed from the open-sourced FedEx (https://github.com/mkhodak/FedEx)
    """

    def _apply_hyperparams(self, hyperparams):
        cmd_args = []
        for k, v in hyperparams.items():
            cmd_args.append(k)
            cmd_args.append(v)

        self._cfg.defrost()
        self._cfg.merge_from_list(cmd_args)
        self._cfg.freeze()

        self.trainer.ctx.setup_vars()

    def callback_funcs_for_model_para(self, message: Message):
        round, sender, content = message.state, message.sender, message.content
        model_params, hyperparams = content["model_param"], content["hyperparam"]

        self._apply_hyperparams(hyperparams)

        self.trainer.update(model_params)

        #self.model.load_state_dict(content)
        self.state = round
        sample_size, model_para_all, results = self.trainer.train()
        logger.info(
            formatted_logging(results,
                              rnd=self.state,
                              role='Client #{}'.format(self.ID)))

        self.comm_manager.send(
            Message(msg_type='model_para',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    content=(sample_size, model_para_all)))
