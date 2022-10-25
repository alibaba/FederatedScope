import torch
import logging
import copy
import numpy as np

from federatedscope.core.message import Message
from federatedscope.core.workers.server import Server
from federatedscope.core.auxiliaries.utils import merge_dict
from federatedscope.cl.fedgc.utils import global_NT_xentloss

logger = logging.getLogger(__name__)


class GlobalContrastFLServer(Server):
    r"""
    GlobalContrastFL(Fedgc) Server contain two part in training: Fedavg
    aggragator for client model weight and calculate global loss from
    all sampled client embedding then broadcast all client to train model.
    """
    def __init__(self,
                 ID=-1,
                 state=0,
                 config=None,
                 data=None,
                 model=None,
                 client_num=5,
                 total_round_num=10,
                 device='cpu',
                 strategy=None,
                 **kwargs):
        super(GlobalContrastFLServer,
              self).__init__(ID, state, config, data, model, client_num,
                             total_round_num, device, strategy, **kwargs)
        # Initial seqs_embedding
        self.seqs_embedding = {
            idx: ()
            for idx in range(1, self._cfg.federate.client_num + 1)
        }
        self.loss_list = {
            idx: 0
            for idx in range(1, self._cfg.federate.client_num + 1)
        }

    def _register_default_handlers(self):
        self.register_handlers('join_in', self.callback_funcs_for_join_in)
        self.register_handlers('join_in_info', self.callback_funcs_for_join_in)
        self.register_handlers('model_para', self.callback_funcs_model_para)
        self.register_handlers('metrics', self.callback_funcs_for_metrics)
        self.register_handlers('pred_embedding',
                               self.callback_funcs_global_loss)

    def check_and_move_on_for_global_loss(self):

        minimal_number = self.sample_client_num

        if self.check_buffer(self.state,
                             minimal_number,
                             check_eval_result=False):

            # Receiving enough feedback in the training process

            # Get all the message
            train_msg_buffer = self.msg_buffer['train'][self.state]
            for model_idx in range(self.model_num):
                model = self.models[model_idx]
                msg_list = list()
                for client_id in train_msg_buffer:
                    if self.model_num == 1:
                        pred_embedding = train_msg_buffer[client_id]
                        self.seqs_embedding[client_id] = pred_embedding
                    else:
                        raise ValueError(
                            'GlobalContrastFL server not support multi-model.')

                global_loss_fn = global_NT_xentloss(device=self.device)
                for client_id in train_msg_buffer:
                    z1 = self.seqs_embedding[client_id][0]
                    z2 = self.seqs_embedding[client_id][1]
                    others_z2 = [
                        self.seqs_embedding[other_client_id][1]
                        for other_client_id in train_msg_buffer
                        if other_client_id != client_id
                    ]
                    self.loss_list[client_id] = global_loss_fn(
                        z1, z2, others_z2)
                    logger.info(f'client {client_id}'
                                f'global_loss:{self.loss_list[client_id]}')

            self.state += 1
            if self.state <= self.total_round_num:

                for client_id in train_msg_buffer:

                    msg_list = {
                        'global_loss': self.loss_list[client_id],
                    }

                    self.comm_manager.send(
                        Message(msg_type='global_loss',
                                sender=self.ID,
                                receiver=[client_id],
                                state=self.state,
                                content=msg_list))

    def check_and_move_on(self,
                          check_eval_result=False,
                          min_received_num=None):
        """
        To check the message_buffer. When enough messages are receiving,
        some events (such as perform aggregation, evaluation, and move to
        the next training round) would be triggered.

        Arguments:
            check_eval_result (bool): If True, check the message buffer for
            evaluation; and check the message buffer for training otherwise.
        """
        if min_received_num is None:
            if self._cfg.asyn.use:
                min_received_num = self._cfg.asyn.min_received_num
            else:
                min_received_num = self._cfg.federate.sample_client_num
        assert min_received_num <= self.sample_client_num

        if check_eval_result and self._cfg.federate.mode.lower(
        ) == "standalone":
            # in evaluation stage and standalone simulation mode, we assume
            # strong synchronization that receives responses from all clients
            min_received_num = len(self.comm_manager.get_neighbors().keys())

        move_on_flag = True  # To record whether moving to a new training
        # round or finishing the evaluation
        if self.check_buffer(self.state, min_received_num, check_eval_result):
            if not check_eval_result:
                # Receiving enough feedback in the training process
                aggregated_num = self._perform_federated_aggregation()

                if self.state % self._cfg.eval.freq == 0 and self.state != \
                        self.total_round_num:
                    #  Evaluate
                    logger.info(f'Server: Starting evaluation at the end '
                                f'of round {self.state - 1}.')
                    self.eval()

                if self.state < self.total_round_num:
                    # Move to next round of training
                    logger.info(
                        f'----------- Starting a new training round (Round '
                        f'#{self.state}) -------------')
                    # Clean the msg_buffer
                    self.msg_buffer['train'][self.state - 1].clear()
                    self.msg_buffer['train'][self.state] = dict()
                    self.staled_msg_buffer.clear()
                    # Start a new training round
                    self._start_new_training_round(aggregated_num)
                else:
                    # Final Evaluate
                    logger.info('Server: Training is finished! Starting '
                                'evaluation.')
                    self.eval()

            else:
                # Receiving enough feedback in the evaluation process
                self._merge_and_format_eval_results()

        else:
            move_on_flag = False

        return move_on_flag

    def callback_funcs_global_loss(self, message: Message):
        """
        The handling function for receiving model embeddings, which triggers
            check_and_move_on (calculate global loss when enough feedback has
            been received).

        Arguments:
            message: The received message, which includes sender, receiver,
                state, and content. More detail can be found in
                federatedscope.core.message
        """
        if self.is_finish:
            return 'finish'

        round = message.state
        sender = message.sender
        timestamp = message.timestamp
        content = message.content
        self.sampler.change_state(sender, 'idle')

        # update the currency timestamp according to the received message
        assert timestamp >= self.cur_timestamp  # for test
        self.cur_timestamp = timestamp

        if round == self.state:
            if round not in self.msg_buffer['train']:
                self.msg_buffer['train'][round] = dict()
            # Save the messages in this round
            self.msg_buffer['train'][round][sender] = content
        elif round >= self.state - self.staleness_toleration:
            # Save the staled messages
            self.staled_msg_buffer.append((round, sender, content))

        move_on_flag = self.check_and_move_on_for_global_loss()

        return move_on_flag

    def callback_funcs_model_para(self, message: Message):
        """
        The handling function for receiving model parameters, which triggers
            check_and_move_on (perform aggregation when enough feedback has
            been received).
        This handling function is widely used in various FL courses.

        Arguments:
            message: The received message, which includes sender, receiver,
                state, and content. More detail can be found in
                federatedscope.core.message
        """
        if self.is_finish:
            return 'finish'

        round = message.state
        sender = message.sender
        timestamp = message.timestamp
        content = message.content
        self.sampler.change_state(sender, 'idle')

        # update the currency timestamp according to the received message
        assert timestamp >= self.cur_timestamp  # for test
        self.cur_timestamp = timestamp

        if round == self.state:
            if round not in self.msg_buffer['train']:
                self.msg_buffer['train'][round] = dict()
            # Save the messages in this round
            self.msg_buffer['train'][round][sender] = content
        elif round >= self.state - self.staleness_toleration:
            # Save the staled messages
            self.staled_msg_buffer.append((round, sender, content))
        else:
            # Drop the out-of-date messages
            logger.info(f'Drop a out-of-date message from round #{round}')
            self.dropout_num += 1

        if self._cfg.federate.online_aggr:
            self.aggregator.inc(content[:2])

        move_on_flag = self.check_and_move_on()
        if self._cfg.asyn.use and self._cfg.asyn.broadcast_manner == \
                'after_receiving':
            self.broadcast_model_para(msg_type='model_para',
                                      sample_client_num=1)

        return move_on_flag
