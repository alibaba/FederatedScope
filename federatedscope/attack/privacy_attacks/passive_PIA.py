from federatedscope.attack.auxiliary.utils import get_classifier, \
    get_passive_PIA_auxiliary_dataset
import torch
import numpy as np
import copy
from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer

import logging

logger = logging.getLogger(__name__)


class PassivePropertyInference():
    '''
    This is an implementation of the passive property inference
    （algorithm 3 in Exploiting Unintended Feature Leakage
    in Collaborative Learning: https://arxiv.org/pdf/1805.04049.pdf
    '''
    def __init__(self,
                 classier: str,
                 fl_model_criterion,
                 device,
                 grad_clip,
                 dataset_name,
                 fl_local_update_num,
                 fl_type_optimizer,
                 fl_lr,
                 batch_size=100):
        # self.auxiliary_dataset['x']: n * d_feature; x is the parameter
        # updates
        # self.auxiliary_dataset['y']: n * 1; y is the
        self.dataset_prop_classifier = {"x": None, 'prop': None}

        self.classifier = get_classifier(classier)

        self.auxiliary_dataset = get_passive_PIA_auxiliary_dataset(
            dataset_name)

        self.fl_model_criterion = fl_model_criterion
        self.fl_local_update_num = fl_local_update_num
        self.fl_type_optimizer = fl_type_optimizer
        self.fl_lr = fl_lr

        self.device = device

        self.batch_size = batch_size

        self.grad_clip = grad_clip

        self.collect_updates_summary = dict()

    # def _get_batch_auxiliary(self):
    #     train_data_batch = self._get_batch(self.auxiliary_dataset['train'])
    #     test_data_batch = self._get_batch(self.auxiliary_dataset['test'])
    #
    #     return train_data_batch, test_data_batch

    def _get_batch(self, data):
        prop_ind = np.random.choice(np.where(data['prop'] == 1)[0],
                                    self.batch_size,
                                    replace=True)
        x_batch_prop = data['x'][prop_ind, :]
        y_batch_prop = data['y'][prop_ind, :]

        nprop_ind = np.random.choice(np.where(data['prop'] == 0)[0],
                                     self.batch_size,
                                     replace=True)
        x_batch_nprop = data['x'][nprop_ind, :]
        y_batch_nprop = data['y'][nprop_ind, :]

        return [x_batch_prop, y_batch_prop, x_batch_nprop, y_batch_nprop]

    def get_data_for_dataset_prop_classifier(self, model, local_runs=10):

        previous_para = model.state_dict()
        self.current_model_para = previous_para
        for _ in range(local_runs):
            x_batch_prop, y_batch_prop, x_batch_nprop, y_batch_nprop = \
                self._get_batch(self.auxiliary_dataset)
            para_update_prop = self._get_parameter_updates(
                model, previous_para, x_batch_prop, y_batch_prop)
            prop = torch.tensor([[1]]).to(torch.device(self.device))
            self.add_parameter_updates(para_update_prop, prop)

            para_update_nprop = self._get_parameter_updates(
                model, previous_para, x_batch_nprop, y_batch_nprop)
            prop = torch.tensor([[0]]).to(torch.device(self.device))
            self.add_parameter_updates(para_update_nprop, prop)

    def _get_parameter_updates(self, model, previous_para, x_batch, y_batch):

        model = copy.deepcopy(model)
        # get last phase model parameters
        # print(model)
        model.load_state_dict(previous_para, strict=False)

        optimizer = get_optimizer(type=self.fl_type_optimizer,
                                  model=model,
                                  lr=self.fl_lr)

        for _ in range(self.fl_local_update_num):
            optimizer.zero_grad()
            loss_auxiliary_prop = self.fl_model_criterion(
                model(torch.Tensor(x_batch).to(torch.device(self.device))),
                torch.Tensor(y_batch).to(torch.device(self.device)))
            loss_auxiliary_prop.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               self.grad_clip)
            optimizer.step()

        para_prop = model.state_dict()
        # print('update: ')
        # print(para_prop)

        updates_prop = torch.hstack([
            (previous_para[name] - para_prop[name]).flatten().cpu()
            for name in previous_para.keys()
        ])
        model.load_state_dict(previous_para, strict=False)
        return updates_prop

    def collect_updates(self, previous_para, updated_parameter, round,
                        client_id):

        updates_prop = torch.hstack([
            (previous_para[name] - updated_parameter[name]).flatten().cpu()
            for name in previous_para.keys()
        ])
        if round not in self.collect_updates_summary.keys():
            self.collect_updates_summary[round] = dict()
        self.collect_updates_summary[round][client_id] = updates_prop

    def add_parameter_updates(self, parameter_updates, prop):
        '''

        Args:
            parameter_updates: Tensor with dimension n * d_feature
            prop: Tensor with dimension  n * 1

        Returns:

        '''
        if self.dataset_prop_classifier['x'] is None:
            self.dataset_prop_classifier['x'] = parameter_updates.cpu()
            self.dataset_prop_classifier['y'] = prop.reshape([-1]).cpu()
        else:
            self.dataset_prop_classifier['x'] = torch.vstack(
                (self.dataset_prop_classifier['x'], parameter_updates.cpu()))
            self.dataset_prop_classifier['y'] = torch.vstack(
                (self.dataset_prop_classifier['y'], prop.cpu()))

    def train_property_classifier(self):
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(
            self.dataset_prop_classifier['x'],
            self.dataset_prop_classifier['y'],
            test_size=0.33,
            random_state=42)
        self.classifier.fit(x_train, y_train)

        y_pred = self.property_inference(x_test)
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
        logger.info(
            '=============== PIA accuracy on auxiliary test dataset: {}'.
            format(accuracy))

    def property_inference(self, parameter_updates):
        return self.classifier.predict(parameter_updates)

    def infer_collected(self):
        pia_results = dict()

        for round in self.collect_updates_summary.keys():
            for id in self.collect_updates_summary[round].keys():
                if round not in pia_results.keys():
                    pia_results[round] = dict()
                pia_results[round][id] = self.property_inference(
                    self.collect_updates_summary[round][id].reshape(1, -1))
        return pia_results
