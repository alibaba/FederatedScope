from federatedscope.core.worker import Server
from federatedscope.core.message import Message

from federatedscope.core.auxiliaries.criterion_builder import get_criterion
import copy
from federatedscope.attack.auxiliary.utils import get_data_sav_fn, \
    get_reconstructor

import logging

import torch
from federatedscope.attack.privacy_attacks.passive_PIA import \
    PassivePropertyInference

logger = logging.getLogger(__name__)


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

    def _reconstruct(self, state, sender):
        # print(self.msg_buffer['train'][state].keys())
        dummy_data, dummy_label = self.reconstructor.reconstruct(
            model=copy.deepcopy(self.model).to(torch.device(self.device)),
            original_info=self.msg_buffer['train'][state][sender][1],
            data_feature_dim=self.data_dim,
            num_class=self.num_class,
            batch_size=self.msg_buffer['train'][state][sender][0])
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
                logger.info(
                    '------------- reconstruct round:{}, client:{}-----------'.
                    format(state, sender))

                # the context of buffer: self.model_buffer[state]: (
                # sample_size, model_para)
                self._reconstruct(state, sender)

    def callback_funcs_model_para(self, message: Message):
        round, sender, content = message.state, message.sender, message.content
        self.sampler.change_state(sender, 'idle')
        # For a new round
        if round not in self.msg_buffer['train'].keys():
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
            fl_type_optimizer=self._cfg.fedopt.optimizer.type,
            fl_lr=self._cfg.train.optimizer.lr,
            batch_size=100)

        # self.optimizer = get_optimizer(
        # type=self._cfg.fedopt.type_optimizer, model=self.model,
        # lr=self._cfg.fedopt.optimizer.lr)
        # print(self.optimizer)
    def callback_funcs_model_para(self, message: Message):
        round, sender, content = message.state, message.sender, message.content
        self.sampler.change_state(sender, 'idle')
        # For a new round
        if round not in self.msg_buffer['train'].keys():
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
