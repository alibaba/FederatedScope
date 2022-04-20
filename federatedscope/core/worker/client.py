import copy
import logging

from federatedscope.core.message import Message
from federatedscope.core.communication import StandaloneCommManager, gRPCCommManager
from federatedscope.core.worker import Worker
from federatedscope.core.auxiliaries.utils import formatted_logging
from federatedscope.core.auxiliaries.trainer_builder import get_trainer
from federatedscope.core.optimizer import wrap_regularized_optimizer
from federatedscope.core.secret_sharing import AdditiveSecretSharing


class Client(Worker):
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

        super(Client, self).__init__(ID, state, config, model, strategy)

        # Attack only support the stand alone model; Check if is a attacker; a client is a attacker if the config.attack.attack_method is provided
        self.is_attacker = config.attack.attacker_id == ID and config.attack.attack_method != '' and config.federate.mode == 'standalone'

        # Build Trainer
        # trainer might need configurations other than those of trainer node
        self.trainer = get_trainer(model=model,
                                   data=data,
                                   device=device,
                                   config=self._cfg,
                                   is_attacker=self.is_attacker)

        # Secret Sharing Manager and message buffer
        self.ss_manager = AdditiveSecretSharing(
            shared_party_num=int(self._cfg.federate.sample_client_num
                                 )) if self._cfg.federate.use_ss else None
        self.msg_buffer = {'train': dict(), 'eval': dict()}

        # Register message handlers
        self.msg_handlers = dict()
        self._register_default_handlers()

        # Initialize communication manager
        self.server_id = server_id
        if self.mode == 'standalone':
            comm_queue = kwargs['shared_comm_queue']
            self.comm_manager = StandaloneCommManager(comm_queue=comm_queue)
            self.local_address = None
        elif self.mode == 'distributed':
            host = kwargs['host']
            port = kwargs['port']
            server_host = kwargs['server_host']
            server_port = kwargs['server_port']
            self.comm_manager = gRPCCommManager(
                host=host, port=port, client_num=self._cfg.federate.client_num)
            logging.info('Client: Listen to {}:{}...'.format(host, port))
            self.comm_manager.add_neighbors(neighbor_id=server_id,
                                            address={
                                                'host': server_host,
                                                'port': server_port
                                            })
            self.local_address = {
                'host': self.comm_manager.host,
                'port': self.comm_manager.port
            }

    def register_handlers(self, msg_type, callback_func):
        """
        To bind a message type with a handled function
        """
        self.msg_handlers[msg_type] = callback_func

    def _register_default_handlers(self):
        self.register_handlers('assign_client_id',
                               self.callback_funcs_for_assign_id)
        self.register_handlers('ask_for_join_in_info',
                               self.callback_funcs_for_join_in_info)
        self.register_handlers('address', self.callback_funcs_for_address)
        self.register_handlers('model_para',
                               self.callback_funcs_for_model_para)
        self.register_handlers('ss_model_para',
                               self.callback_funcs_for_model_para)
        self.register_handlers('evaluate', self.callback_funcs_for_evaluate)
        self.register_handlers('finish', self.callback_funcs_for_finish)

    def join_in(self):
        """
        To send 'join_in' message to the server
        """
        self.comm_manager.send(
            Message(msg_type='join_in',
                    sender=self.ID,
                    receiver=[self.server_id],
                    content=self.local_address))

    def run(self):
        """
        To wait for the messages and handle them (for distributed mode)
        """
        while True:
            msg = self.comm_manager.receive()
            if self.state <= msg.state:
                self.msg_handlers[msg.msg_type](msg)

            if msg.msg_type == 'finish':
                break

    def callback_funcs_for_model_para(self, message: Message):
        if 'ss' in message.msg_type:
            # A fragment of the shared secret
            state, content = message.state, message.content
            self.msg_buffer['train'][state].append(content)

            if len(self.msg_buffer['train']
                   [state]) == self._cfg.federate.client_num:
                # Check whether the received fragments are enough
                model_list = self.msg_buffer['train'][state]
                sample_size, first_aggregate_model_para = model_list[0]
                single_model_case = True
                if isinstance(first_aggregate_model_para, list):
                    assert isinstance(first_aggregate_model_para[0], dict), \
                        "aggregate_model_para should a list of multiple state_dict for multiple models"
                    single_model_case = False
                else:
                    assert isinstance(first_aggregate_model_para, dict), \
                        "aggregate_model_para should a state_dict for single model case"
                    first_aggregate_model_para = [first_aggregate_model_para]
                    model_list = [[model] for model in model_list]

                for sub_model_idx, aggregate_single_model_para in enumerate(
                        first_aggregate_model_para):
                    for key in aggregate_single_model_para:
                        for i in range(1, len(model_list)):
                            aggregate_single_model_para[key] += model_list[i][
                                sub_model_idx][key]

                self.comm_manager.send(
                    Message(msg_type='model_para',
                            sender=self.ID,
                            receiver=[self.server_id],
                            state=self.state,
                            content=(sample_size, first_aggregate_model_para[0]
                                     if single_model_case else
                                     first_aggregate_model_para)))

        else:
            round, sender, content = message.state, message.sender, message.content
            self.trainer.update(content)
            #self.model.load_state_dict(content)
            self.state = round
            sample_size, model_para_all, results = self.trainer.train()
            logging.info(
                formatted_logging(results,
                                  rnd=self.state,
                                  role='Client #{}'.format(self.ID)))

            # Return the feedbacks to the server after local update
            if self._cfg.federate.use_ss:
                single_model_case = True
                if isinstance(model_para_all, list):
                    assert isinstance(model_para_all[0], dict), \
                        "model_para should a list of multiple state_dict for multiple models"
                    single_model_case = False
                else:
                    assert isinstance(model_para_all, dict), \
                        "model_para should a state_dict for single model case"
                    model_para_all = [model_para_all]
                model_para_list_all = []
                for model_para in model_para_all:
                    for key in model_para:
                        model_para[key] = model_para[key] * sample_size
                    model_para_list = self.ss_manager.secret_split(model_para)
                    model_para_list_all.append(model_para_list)
                    #print(model_para)
                    #print(self.ss_manager.secret_reconstruct(model_para_list))
                frame_idx = 0
                for neighbor in self.comm_manager.neighbors:
                    if neighbor != self.server_id:
                        content_frame = model_para_list_all[0][frame_idx] if single_model_case else \
                            [model_para_list[frame_idx] for model_para_list in model_para_list_all]
                        self.comm_manager.send(
                            Message(msg_type='ss_model_para',
                                    sender=self.ID,
                                    receiver=[neighbor],
                                    state=self.state,
                                    content=content_frame))
                        frame_idx += 1
                content_frame = model_para_list_all[0][frame_idx] if single_model_case else \
                        [model_para_list[frame_idx] for model_para_list in model_para_list_all]
                self.msg_buffer['train'][self.state] = [(sample_size,
                                                         content_frame)]
            else:
                self.comm_manager.send(
                    Message(msg_type='model_para',
                            sender=self.ID,
                            receiver=[sender],
                            state=self.state,
                            content=(sample_size, model_para_all)))

    def callback_funcs_for_assign_id(self, message: Message):
        content = message.content
        self.ID = int(content)
        logging.info('Client (address {}:{}) is assigned with #{:d}.'.format(
            self.comm_manager.host, self.comm_manager.port, self.ID))

    def callback_funcs_for_join_in_info(self, message: Message):
        requirements = message.content
        join_in_info = dict()
        for requirement in requirements:
            if requirement.lower() == 'num_sample':
                if self._cfg.federate.batch_or_epoch == 'batch':
                    num_sample = self._cfg.federate.local_update_steps * self._cfg.data.batch_size
                else:
                    num_sample = self._cfg.federate.local_update_steps * self.trainer.ctx.num_train_batch
                join_in_info['num_sample'] = num_sample
            else:
                raise ValueError(
                    'Fail to get the join in information with type {}'.format(
                        requirement))
        self.comm_manager.send(
            Message(msg_type='join_in_info',
                    sender=self.ID,
                    receiver=[self.server_id],
                    state=self.state,
                    content=join_in_info))

    def callback_funcs_for_address(self, message: Message):
        content = message.content
        for neighbor_id, address in content.items():
            if int(neighbor_id) != self.ID:
                self.comm_manager.add_neighbors(neighbor_id, address)

    def callback_funcs_for_evaluate(self, message: Message):
        sender = message.sender
        self.state = message.state
        if message.content != None:
            self.trainer.update(message.content)
        metrics = {}
        for split in self._cfg.eval.split:
            mode = split if split != "train" else "test"
            eval_metrics = self.trainer.evaluate(mode=mode, target_data_split_name=split)
            for key in eval_metrics:
                logging.info(
                    'Client #{:d}: (Evaluation ({:s} set) at Round #{:d}) {:s} is {:.6f}'
                    .format(self.ID, split, self.state, key,
                            eval_metrics[key]))
            metrics.update(**eval_metrics)
        self.comm_manager.send(
            Message(msg_type='metrics',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    content=metrics))

    def callback_funcs_for_finish(self, message: Message):
        logging.info(
            "================= receiving Finish Message ============================"
        )

        if message.content != None:
            self.trainer.update(message.content)
