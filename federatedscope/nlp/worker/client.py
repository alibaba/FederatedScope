import copy
import logging

from federatedscope.core.message import Message
from federatedscope.core.communication import StandaloneCommManager, gRPCCommManager
from federatedscope.core.monitors.early_stopper import EarlyStopper
from federatedscope.core.monitors.monitor import update_best_result
from federatedscope.core.worker import Worker
from federatedscope.core.auxiliaries.trainer_builder import get_trainer
from federatedscope.core.secret_sharing import AdditiveSecretSharing
from federatedscope.core.auxiliaries.utils import merge_dict
from federatedscope.core.worker.client import Client

logger = logging.getLogger(__name__)


class TextDTClient(Client):
    """
    The Client class, which describes the behaviors of client in an FL course.
    The attributes include:
        ID: The unique ID of the client, which is assigned by the server when joining the FL course
        server_id: (Default) 0
        state: The training round
        config: the configuration
        data: The data owned by the client
        model: The local model
        device: The device to run local training and evaluation
        strategy: redundant attribute
    The behaviors are described by the handled functions (named as callback_funcs_for_xxx)
    """
    def __init__(self,
                 ID=-1,
                 server_id=None,
                 state=-1,
                 config=None,
                 data=None,
                 model=None,
                 device='cpu',
                 strategy=None,
                 *args,
                 **kwargs):

        super().__init__(ID=ID,
                         server_id=server_id,
                         state=state,
                         config=config,
                         data=data,
                         model=model,
                         device=device,
                         strategy=strategy,
                         *args,
                         **kwargs,)

    def callback_funcs_for_evaluate(self, message: Message):
        sender = message.sender
        self.state = message.state
        if message.content != None:
            self.trainer.update(message.content)
        if self.early_stopper.early_stopped:
            metrics = list(self.best_results.values())[0]
        else:
            metrics = {}
            if self._cfg.trainer.finetune.before_eval:
                self.trainer.finetune()
            for split in self._cfg.eval.split:
                eval_metrics = self.trainer.evaluate(
                    target_data_split_name=split)

                if self._cfg.federate.mode == 'distributed':
                    logger.info(
                        self._monitor.format_eval_res(eval_metrics,
                                                      rnd=self.state,
                                                      role='Client #{}'.format(
                                                          self.ID)))

                metrics.update(**eval_metrics)

            formatted_eval_res = self._monitor.format_eval_res(
                metrics,
                rnd=self.state + 1,
                role='Client #{}'.format(self.ID),
                forms='raw',
                return_raw=True)
            update_best_result(self.best_results,
                               formatted_eval_res['Results_raw'],
                               results_type=f"client #{self.ID}",
                               round_wise_update_key=self._cfg.eval.
                               best_res_update_round_wise_key)
            self.history_results = merge_dict(
                self.history_results, formatted_eval_res['Results_raw'])
            if self._cfg.federate.method in ["local", "global"]:
                self.early_stopper.track_and_check_best(self.history_results[
                    self._cfg.eval.best_res_update_round_wise_key])

        self.comm_manager.send(
            Message(msg_type='metrics',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    content=metrics))
