from federatedscope.core.workers import Server
from federatedscope.core.message import Message

from federatedscope.core.auxiliaries.criterion_builder import get_criterion
import copy
from federatedscope.attack.auxiliary.utils import get_data_sav_fn, \
    get_reconstructor

import logging

import torch
import numpy as np
from federatedscope.attack.privacy_attacks.passive_PIA import \
    PassivePropertyInference

logger = logging.getLogger(__name__)


class BackdoorServer(Server):
    '''
    For backdoor attacks, we will choose different sampling stratergies.
    fix-frequency, all-round ,or random sampling.
    '''
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
                 unseen_clients_id=None,
                 **kwargs):
        super(BackdoorServer, self).__init__(ID=ID,
                                             state=state,
                                             data=data,
                                             model=model,
                                             config=config,
                                             client_num=client_num,
                                             total_round_num=total_round_num,
                                             device=device,
                                             strategy=strategy,
                                             **kwargs)

    def broadcast_model_para(self,
                             msg_type='model_para',
                             sample_client_num=-1,
                             filter_unseen_clients=True):
        """
        To broadcast the message to all clients or sampled clients

        Arguments:
            msg_type: 'model_para' or other user defined msg_type
            sample_client_num: the number of sampled clients in the broadcast
                behavior. And sample_client_num = -1 denotes to broadcast to
                all the clients.
            filter_unseen_clients: whether filter out the unseen clients that
                do not contribute to FL process by training on their local
                data and uploading their local model update. The splitting is
                useful to check participation generalization gap in [ICLR'22,
                What Do We Mean by Generalization in Federated Learning?]
                You may want to set it to be False when in evaluation stage
        """

        if filter_unseen_clients:
            # to filter out the unseen clients when sampling
            self.sampler.change_state(self.unseen_clients_id, 'unseen')

        if sample_client_num > 0:  # only activated at training process
            attacker_id = self._cfg.attack.attacker_id
            setting = self._cfg.attack.setting
            insert_round = self._cfg.attack.insert_round

            if attacker_id == -1 or self._cfg.attack.attack_method == '':

                receiver = np.random.choice(np.arange(1, self.client_num + 1),
                                            size=sample_client_num,
                                            replace=False).tolist()

            elif setting == 'fix':
                if self.state % self._cfg.attack.freq == 0:
                    client_list = np.delete(np.arange(1, self.client_num + 1),
                                            self._cfg.attack.attacker_id - 1)
                    receiver = np.random.choice(client_list,
                                                size=sample_client_num - 1,
                                                replace=False).tolist()
                    receiver.insert(0, self._cfg.attack.attacker_id)
                    logger.info('starting the fix-frequency poisoning attack')
                    logger.info(
                        'starting poisoning round: {:d}, the attacker ID: {:d}'
                        .format(self.state, self._cfg.attack.attacker_id))
                else:
                    client_list = np.delete(np.arange(1, self.client_num + 1),
                                            self._cfg.attack.attacker_id - 1)
                    receiver = np.random.choice(client_list,
                                                size=sample_client_num,
                                                replace=False).tolist()

            elif setting == 'single' and self.state == insert_round:
                client_list = np.delete(np.arange(1, self.client_num + 1),
                                        self._cfg.attack.attacker_id - 1)
                receiver = np.random.choice(client_list,
                                            size=sample_client_num - 1,
                                            replace=False).tolist()
                receiver.insert(0, self._cfg.attack.attacker_id)
                logger.info('starting the single-shot poisoning attack')
                logger.info(
                    'starting poisoning round: {:d}, the attacker ID: {:d}'.
                    format(self.state, self._cfg.attack.attacker_id))

            elif self._cfg.attack.setting == 'all':

                client_list = np.delete(np.arange(1, self.client_num + 1),
                                        self._cfg.attack.attacker_id - 1)
                receiver = np.random.choice(client_list,
                                            size=sample_client_num - 1,
                                            replace=False).tolist()
                receiver.insert(0, self._cfg.attack.attacker_id)
                logger.info('starting the all-round poisoning attack')
                logger.info(
                    'starting poisoning round: {:d}, the attacker ID: {:d}'.
                    format(self.state, self._cfg.attack.attacker_id))

            else:
                receiver = np.random.choice(np.arange(1, self.client_num + 1),
                                            size=sample_client_num,
                                            replace=False).tolist()

        else:
            # broadcast to all clients
            receiver = list(self.comm_manager.neighbors.keys())

        if self._noise_injector is not None and msg_type == 'model_para':
            # Inject noise only when broadcast parameters
            for model_idx_i in range(len(self.models)):
                num_sample_clients = [
                    v["num_sample"] for v in self.join_in_info.values()
                ]
                self._noise_injector(self._cfg, num_sample_clients,
                                     self.models[model_idx_i])

        skip_broadcast = self._cfg.federate.method in ["local", "global"]
        if self.model_num > 1:
            model_para = [{} if skip_broadcast else model.state_dict()
                          for model in self.models]
        else:
            model_para = {} if skip_broadcast else self.model.state_dict()

        self.comm_manager.send(
            Message(msg_type=msg_type,
                    sender=self.ID,
                    receiver=receiver,
                    state=min(self.state, self.total_round_num),
                    content=model_para))
        if self._cfg.federate.online_aggr:
            for idx in range(self.model_num):
                self.aggregators[idx].reset()

        if filter_unseen_clients:
            # restore the state of the unseen clients within sampler
            self.sampler.change_state(self.unseen_clients_id, 'seen')


class PassiveServer(Server):
    '''
    In passive attack, the server store the model and the message collected
    from the client,and perform the optimization based reconstruction,
    such as DLG, InvertGradient.
    '''
    def __init__(self,
                 ID=-1,
                 state=0,
                 data=None,
                 model=None,
                 client_num=5,
                 total_round_num=10,
                 device='cpu',
                 strategy=None,
                 state_to_reconstruct=None,
                 client_to_reconstruct=None,
                 **kwargs):
        super(PassiveServer, self).__init__(ID=ID,
                                            state=state,
                                            data=data,
                                            model=model,
                                            client_num=client_num,
                                            total_round_num=total_round_num,
                                            device=device,
                                            strategy=strategy,
                                            **kwargs)

        # self.offline_reconstruct = offline_reconstruct
        self.atk_method = self._cfg.attack.attack_method
        self.state_to_reconstruct = state_to_reconstruct
        self.client_to_reconstruct = client_to_reconstruct
        self.reconstruct_data = dict()

        # the loss function of the global model; the global model can be
        # obtained in self.aggregator.model
        self.model_criterion = get_criterion(self._cfg.criterion.type,
                                             device=self.device)

        from federatedscope.attack.auxiliary.utils import get_data_info
        self.data_dim, self.num_class, self.is_one_hot_label = get_data_info(
            self._cfg.data.type)

        self.reconstructor = self._get_reconstructor()

        self.reconstructed_data_sav_fn = get_data_sav_fn(self._cfg.data.type)

        self.reconstruct_data_summary = dict()

    def _get_reconstructor(self):

        return get_reconstructor(
            self.atk_method,
            max_ite=self._cfg.attack.max_ite,
            lr=self._cfg.attack.reconstruct_lr,
            federate_loss_fn=self.model_criterion,
            device=self.device,
            federate_lr=self._cfg.train.optimizer.lr,
            optim=self._cfg.attack.reconstruct_optim,
            info_diff_type=self._cfg.attack.info_diff_type,
            federate_method=self._cfg.federate.method,
            alpha_TV=self._cfg.attack.alpha_TV)

    def _reconstruct(self, model_para, batch_size, state, sender):
        logger.info('-------- reconstruct round:{}, client:{}---------'.format(
            state, sender))
        dummy_data, dummy_label = self.reconstructor.reconstruct(
            model=copy.deepcopy(self.model).to(torch.device(self.device)),
            original_info=model_para,
            data_feature_dim=self.data_dim,
            num_class=self.num_class,
            batch_size=batch_size)
        if state not in self.reconstruct_data.keys():
            self.reconstruct_data[state] = dict()
        self.reconstruct_data[state][sender] = [
            dummy_data.cpu(), dummy_label.cpu()
        ]

    def run_reconstruct(self, state_list=None, sender_list=None):

        if state_list is None:
            state_list = self.msg_buffer['train'].keys()

        # After FL running, using gradient based reconstruction method to
        # recover client's private training data
        for state in state_list:
            if sender_list is None:
                sender_list = self.msg_buffer['train'][state].keys()
            for sender in sender_list:
                content = self.msg_buffer['train'][state][sender]
                self._reconstruct(model_para=content[1],
                                  batch_size=content[0],
                                  state=state,
                                  sender=sender)

    def callback_funcs_model_para(self, message: Message):
        if self.is_finish:
            return 'finish'

        round, sender, content = message.state, message.sender, message.content
        self.sampler.change_state(sender, 'idle')
        if round not in self.msg_buffer['train']:
            self.msg_buffer['train'][round] = dict()
        self.msg_buffer['train'][round][sender] = content

        # run reconstruction before the clear of self.msg_buffer

        if self.state_to_reconstruct is None or message.state in \
                self.state_to_reconstruct:
            if self.client_to_reconstruct is None or message.sender in \
                    self.client_to_reconstruct:
                self.run_reconstruct(state_list=[message.state],
                                     sender_list=[message.sender])
                if self.reconstructed_data_sav_fn is not None:
                    self.reconstructed_data_sav_fn(
                        data=self.reconstruct_data[message.state][
                            message.sender][0],
                        sav_pth=self._cfg.outdir,
                        name='image_state_{}_client_{}.png'.format(
                            message.state, message.sender))

        self.check_and_move_on()


class PassivePIAServer(Server):
    '''
    The implementation of the batch property classifier, the algorithm 3 in
    paper: Exploiting Unintended Feature Leakage in Collaborative Learning

    References:

    Melis, Luca, Congzheng Song, Emiliano De Cristofaro and Vitaly
    Shmatikov. “Exploiting Unintended Feature Leakage in Collaborative
    Learning.” 2019 IEEE Symposium on Security and Privacy (SP) (2019): 691-706
    '''
    def __init__(self,
                 ID=-1,
                 state=0,
                 data=None,
                 model=None,
                 client_num=5,
                 total_round_num=10,
                 device='cpu',
                 strategy=None,
                 **kwargs):
        super(PassivePIAServer, self).__init__(ID=ID,
                                               state=state,
                                               data=data,
                                               model=model,
                                               client_num=client_num,
                                               total_round_num=total_round_num,
                                               device=device,
                                               strategy=strategy,
                                               **kwargs)

        # self.offline_reconstruct = offline_reconstruct
        self.atk_method = self._cfg.attack.attack_method
        self.pia_attacker = PassivePropertyInference(
            classier=self._cfg.attack.classifier_PIA,
            fl_model_criterion=get_criterion(self._cfg.criterion.type,
                                             device=self.device),
            device=self.device,
            grad_clip=self._cfg.grad.grad_clip,
            dataset_name=self._cfg.data.type,
            fl_local_update_num=self._cfg.train.local_update_steps,
            fl_type_optimizer=self._cfg.train.optimizer.type,
            fl_lr=self._cfg.train.optimizer.lr,
            batch_size=100)

        # self.optimizer = get_optimizer(
        # type=self._cfg.fedopt.type_optimizer, model=self.model,
        # lr=self._cfg.fedopt.optimizer.lr)
        # print(self.optimizer)
    def callback_funcs_model_para(self, message: Message):
        if self.is_finish:
            return 'finish'

        round, sender, content = message.state, message.sender, message.content
        self.sampler.change_state(sender, 'idle')
        if round not in self.msg_buffer['train']:
            self.msg_buffer['train'][round] = dict()
        self.msg_buffer['train'][round][sender] = content

        # collect the updates
        self.pia_attacker.collect_updates(
            previous_para=self.model.state_dict(),
            updated_parameter=content[1],
            round=round,
            client_id=sender)
        self.pia_attacker.get_data_for_dataset_prop_classifier(
            model=self.model)

        if self._cfg.federate.online_aggr:
            # TODO: put this line to `check_and_move_on`
            # currently, no way to know the latest `sender`
            self.aggregator.inc(content)
        self.check_and_move_on()

        if self.state == self.total_round_num:
            self.pia_attacker.train_property_classifier()
            self.pia_results = self.pia_attacker.infer_collected()
            print(self.pia_results)
