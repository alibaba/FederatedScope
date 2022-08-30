import torch
import logging
import copy
import numpy as np

from federatedscope.core.message import Message
from federatedscope.core.workers.server import Server
from federatedscope.core.workers.client import Client
from federatedscope.core.auxiliaries.utils import merge_dict
from federatedscope.cl.fedgc.utils import compute_global_NT_xentloss

logger = logging.getLogger(__name__)


class GlobalContrastFLServer(Server):
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
        self.loss_list= {
            idx: 0
            for idx in range(1, self._cfg.federate.client_num + 1)
        }

    
    def check_and_move_on(self, check_eval_result=False):

        if check_eval_result:
            # all clients are participating in evaluation
            minimal_number = self.client_num
        else:
            # sampled clients are participating in training
            minimal_number = self.sample_client_num

        if self.check_buffer(self.state, minimal_number, check_eval_result):

            if not check_eval_result:  # in the training process
                # Receiving enough feedback in the training process
                aggregated_num = self._perform_federated_aggregation()
                
                # Get all the message
                train_msg_buffer = self.msg_buffer['train'][self.state]
                for model_idx in range(self.model_num):
                    model = self.models[model_idx]
                    aggregator = self.aggregators[model_idx]
                    msg_list = list()
                    for client_id in train_msg_buffer:
                        if self.model_num == 1:
                            train_data_size, model_para, pred_embedding = \
                                train_msg_buffer[client_id]
                            self.seqs_embedding[client_id] = pred_embedding
                            msg_list.append((train_data_size, model_para, pred_embedding))
                        else:
                            raise ValueError(
                                'GlobalContrastFL server not support multi-model.')

                    for client_id in train_msg_buffer:
                        z1, z2 = self.seqs_embedding[client_id][0], self.seqs_embedding[client_id][1]
                        others_z2 = [self.seqs_embedding[other_client_id][1] 
                                     for other_client_id in train_msg_buffer 
                                     if other_client_id != client_id]
                        print("start cal loss\n")
                        self.loss_list[client_id] = compute_global_NT_xentloss(z1, z2, others_z2)
                        print(self.loss_list[client_id])
                        print("end cal loss\n")


                self.state += 1
                if self.state % self._cfg.eval.freq == 0 and self.state != \
                        self.total_round_num:
                    #  Evaluate
                    logger.info(
                        'Server: Starting evaluation at round {:d}.'.format(
                            self.state))
                    self.eval()

                if self.state < self.total_round_num:
                    self._start_new_training_round(aggregated_num)
                    for client_id in train_msg_buffer:

                        msg_list = {
                            'global_loss': self.loss_list[client_id],
                        }

                        # Send loss to Clients
                        self.comm_manager.send(
                            Message(msg_type='global_loss',
                                    sender=self.ID,
                                    receiver=[client_id],
                                    state=self.state,
                                    content=msg_list))


                    # Move to next round of training
                    logger.info(
                        f'----------- Starting a new traininground(Round '
                        f'#{self.state}) -------------')
                    # Clean the msg_buffer
                    self.msg_buffer['train'][self.state - 1].clear()
                    self.msg_buffer['train'][self.state] = dict()
                    self.staled_msg_buffer.clear()
                    # Start a new training round
                    

                else:
                    # Final Evaluate
                    logger.info('Server: Training is finished! Starting '
                                'evaluation.')
                    self.eval()

            else:  # in the evaluation process
                # Get all the message & aggregate
                formatted_eval_res = self.merge_eval_results_from_all_clients()
                self.history_results = merge_dict(self.history_results,
                                                  formatted_eval_res)
                self.check_and_save()
                
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


class GlobalContrastFLClient(Client):
    def _register_default_handlers(self):
        self.register_handlers('assign_client_id',
                               self.callback_funcs_for_assign_id)
        self.register_handlers('ask_for_join_in_info',
                               self.callback_funcs_for_join_in_info)
        self.register_handlers('address', self.callback_funcs_for_address)
        self.register_handlers('model_para',
                               self.callback_funcs_for_model_para)
        self.register_handlers('global_loss', self.callback_funcs_for_global_loss)
        self.register_handlers('ss_model_para',
                               self.callback_funcs_for_model_para)
        
        self.register_handlers('evaluate', self.callback_funcs_for_evaluate)
        self.register_handlers('finish', self.callback_funcs_for_finish)
        self.register_handlers('converged', self.callback_funcs_for_converged)
        
    def callback_funcs_for_global_loss(self, message: Message):
        round, sender, content = message.state, message.sender, message.content
        global_loss = content['global_loss']
        model_para_old = self.trainer.get_model_para()
        model_para = self.trainer.train_with_global_loss(model_para_old, global_loss)
        self.trainer.update(model_para)
        
    def callback_funcs_for_model_para(self, message: Message):
        round, sender, content = message.state, message.sender, message.content
        self.trainer.update(content)
        self.state = round
        sample_size, model_para, results = self.trainer.train()
        pred_embedding = self.trainer.get_train_pred_embedding()
        if self._cfg.federate.share_local_model and not \
                self._cfg.federate.online_aggr:
            model_para = copy.deepcopy(model_para)
        logger.info(
            self._monitor.format_eval_res(results,
                                          rnd=self.state,
                                          role='Client #{}'.format(self.ID)))

        self.comm_manager.send(
            Message(msg_type='model_para',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    content=(sample_size, model_para, pred_embedding)))
