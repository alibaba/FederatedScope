import logging
import json
import copy

from federatedscope.core.message import Message
from federatedscope.core.worker import Client

logger = logging.getLogger(__name__)


class FedExClient(Client):
    """Some code snippets are borrowed from the open-sourced FedEx (
    https://github.com/mkhodak/FedEx)
    """
    def _apply_hyperparams(self, hyperparams):
        """Apply the given hyperparameters
        Arguments:
            hyperparams (dict): keys are hyperparameter names \
                and values are specific choices.
        """

        cmd_args = []
        for k, v in hyperparams.items():
            cmd_args.append(k)
            cmd_args.append(v)

        self._cfg.defrost()
        self._cfg.merge_from_list(cmd_args)
        self._cfg.freeze(inform=False)

        self.trainer.ctx.setup_vars()

    def callback_funcs_for_model_para(self, message: Message):
        round, sender, content = message.state, message.sender, message.content
        model_params, arms, hyperparams = content["model_param"], content[
            "arms"], content["hyperparam"]
        attempt = {
            'Role': 'Client #{:d}'.format(self.ID),
            'Round': self.state + 1,
            'Arms': arms,
            'Hyperparams': hyperparams
        }
        logger.info(json.dumps(attempt))

        self._apply_hyperparams(hyperparams)

        self.trainer.update(model_params)

        # self.model.load_state_dict(content)
        self.state = round
        sample_size, model_para_all, results = self.trainer.train()
        if self._cfg.federate.share_local_model and not \
                self._cfg.federate.online_aggr:
            model_para_all = copy.deepcopy(model_para_all)
        logger.info(
            self._monitor.format_eval_res(results,
                                          rnd=self.state,
                                          role='Client #{}'.format(self.ID),
                                          return_raw=True))

        results['arms'] = arms
        content = (sample_size, model_para_all, results)
        self.comm_manager.send(
            Message(msg_type='model_para',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    content=content))

    def callback_funcs_for_evaluate(self, message: Message):
        sender = message.sender
        self.state = message.state
        if message.content is not None:
            model_params = message.content["model_param"]
            self.trainer.update(model_params)
        if self._cfg.finetune.before_eval:
            self.trainer.finetune()
        metrics = {}
        for split in self._cfg.eval.split:
            eval_metrics = self.trainer.evaluate(target_data_split_name=split)
            for key in eval_metrics:

                if self._cfg.federate.mode == 'distributed':
                    logger.info('Client #{:d}: (Evaluation ({:s} set) at '
                                'Round #{:d}) {:s} is {:.6f}'.format(
                                    self.ID, split, self.state, key,
                                    eval_metrics[key]))
                metrics.update(**eval_metrics)
        self.comm_manager.send(
            Message(msg_type='metrics',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    content=metrics))
