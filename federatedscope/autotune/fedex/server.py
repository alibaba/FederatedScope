import os
import logging
from itertools import product

import yaml

import numpy as np
from numpy.linalg import norm
from scipy.special import logsumexp
import torch

from federatedscope.core.message import Message
from federatedscope.core.workers import Server
from federatedscope.core.auxiliaries.utils import merge_dict_of_results
from federatedscope.autotune.fedex.utils import HyperNet

logger = logging.getLogger(__name__)


def discounted_mean(trace, factor=1.0):

    weight = factor**np.flip(np.arange(len(trace)), axis=0)

    return np.inner(trace, weight) / weight.sum()


class FedExServer(Server):
    """Some code snippets are borrowed from the open-sourced FedEx (
    https://github.com/mkhodak/FedEx)
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

        super(FedExServer,
              self).__init__(ID, state, config, data, model, client_num,
                             total_round_num, device, strategy, **kwargs)

        # initialize action space and the policy
        with open(config.hpo.fedex.ss, 'r') as ips:
            ss = yaml.load(ips, Loader=yaml.FullLoader)

        if next(iter(ss.keys())).startswith('arm'):
            # This is a flattened action space
            # ensure the order is unchanged
            ss = sorted([(int(k[3:]), v) for k, v in ss.items()],
                        key=lambda x: x[0])
            self._grid = []
            self._cfsp = [[tp[1] for tp in ss]]
        else:
            # This is not a flat search space
            # be careful for the order
            self._grid = sorted(ss.keys())
            self._cfsp = [ss[pn] for pn in self._grid]

        sizes = [len(cand_set) for cand_set in self._cfsp]
        eta0 = 'auto' if config.hpo.fedex.eta0 <= .0 else float(
            config.hpo.fedex.eta0)
        self._eta0 = [
            np.sqrt(2.0 * np.log(size)) if eta0 == 'auto' else eta0
            for size in sizes
        ]
        self._sched = config.hpo.fedex.sched
        self._cutoff = config.hpo.fedex.cutoff
        self._baseline = config.hpo.fedex.gamma
        self._diff = config.hpo.fedex.diff
        if self._cfg.hpo.fedex.psn:
            # personalized policy
            # TODO: client-wise RFF
            self._client_encodings = torch.randn(
                (client_num, 8), device=device) / np.sqrt(8)
            self._policy_net = HyperNet(
                self._client_encodings.shape[-1],
                sizes,
                client_num,
                device,
            ).to(device)
            self._policy_net.eval()
            theta4stat = [
                theta.detach().cpu().numpy()
                for theta in self._policy_net(self._client_encodings)
            ]
            self._pn_optimizer = torch.optim.Adam(
                self._policy_net.parameters(),
                lr=self._cfg.hpo.fedex.pi_lr,
                weight_decay=1e-5)
        else:
            self._z = [np.full(size, -np.log(size)) for size in sizes]
            self._theta = [np.exp(z) for z in self._z]
            theta4stat = self._theta
            self._store = [0.0 for _ in sizes]
        self._stop_exploration = False
        self._trace = {
            'global': [],
            'refine': [],
            'entropy': [self.entropy(theta4stat)],
            'mle': [self.mle(theta4stat)]
        }

        if self._cfg.federate.restore_from != '':
            if not os.path.exists(self._cfg.federate.restore_from):
                logger.warning(f'Invalid `restore_from`:'
                               f' {self._cfg.federate.restore_from}.')
            else:
                pi_ckpt_path = self._cfg.federate.restore_from[
                               :self._cfg.federate.restore_from.rfind('.')] \
                               + "_fedex.yaml"
                with open(pi_ckpt_path, 'r') as ips:
                    ckpt = yaml.load(ips, Loader=yaml.FullLoader)
                if self._cfg.hpo.fedex.psn:
                    psn_pi_ckpt_path = self._cfg.federate.restore_from[
                               :self._cfg.federate.restore_from.rfind('.')] \
                               + "_pfedex.pt"
                    psn_pi = torch.load(psn_pi_ckpt_path, map_location=device)
                    self._client_encodings = psn_pi['client_encodings']
                    self._policy_net.load_state_dict(psn_pi['policy_net'])
                else:
                    self._z = [np.asarray(z) for z in ckpt['z']]
                    self._theta = [np.exp(z) for z in self._z]
                    self._store = ckpt['store']
                self._stop_exploration = ckpt['stop']
                self._trace = dict()
                self._trace['global'] = ckpt['global']
                self._trace['refine'] = ckpt['refine']
                self._trace['entropy'] = ckpt['entropy']
                self._trace['mle'] = ckpt['mle']

    def entropy(self, thetas):
        if self._cfg.hpo.fedex.psn:
            entropy = 0.0
            for i in range(thetas[0].shape[0]):
                for probs in product(*(theta[i][theta[i] > 0.0]
                                       for theta in thetas)):
                    prob = np.prod(probs)
                    entropy -= prob * np.log(prob)
            return entropy / float(thetas[0].shape[0])
        else:
            entropy = 0.0
            for probs in product(*(theta[theta > 0.0] for theta in thetas)):
                prob = np.prod(probs)
                entropy -= prob * np.log(prob)
            return entropy

    def mle(self, thetas):
        if self._cfg.hpo.fedex.psn:
            return np.prod([theta.max(-1) for theta in thetas], 0).mean()
        else:
            return np.prod([theta.max() for theta in thetas])

    def trace(self, key):
        '''returns trace of one of three tracked quantities
        Args:
            key (str): 'entropy', 'global', or 'refine'
        Returns:
            numpy vector with length equal to number of rounds up to now.
        '''

        return np.array(self._trace[key])

    def sample(self, thetas):
        """samples from configs using current probability vector
        Arguments:
          thetas (list): probabilities for the hyperparameters.
        """

        # determine index
        if self._stop_exploration:
            cfg_idx = [theta.argmax() for theta in thetas]
        else:
            cfg_idx = [
                np.random.choice(len(theta), p=theta) for theta in thetas
            ]

        # get the sampled value(s)
        if self._grid:
            sampled_cfg = {
                pn: cands[i]
                for pn, cands, i in zip(self._grid, self._cfsp, cfg_idx)
            }
        else:
            sampled_cfg = self._cfsp[0][cfg_idx[0]]

        return cfg_idx, sampled_cfg

    def broadcast_model_para(self,
                             msg_type='model_para',
                             sample_client_num=-1,
                             filter_unseen_clients=True):
        """
        To broadcast the message to all clients or sampled clients
        """
        if filter_unseen_clients:
            # to filter out the unseen clients when sampling
            self.sampler.change_state(self.unseen_clients_id, 'unseen')

        if sample_client_num > 0:
            receiver = self.sampler.sample(size=sample_client_num)
        else:
            # broadcast to all clients
            receiver = list(self.comm_manager.neighbors.keys())
            if msg_type == 'model_para':
                self.sampler.change_state(receiver, 'working')

        if self._noise_injector is not None and msg_type == 'model_para':
            # Inject noise only when broadcast parameters
            for model_idx_i in range(len(self.models)):
                num_sample_clients = [
                    v["num_sample"] for v in self.join_in_info.values()
                ]
                self._noise_injector(self._cfg, num_sample_clients,
                                     self.models[model_idx_i])

        if self.model_num > 1:
            model_para = [model.state_dict() for model in self.models]
        else:
            model_para = self.model.state_dict()

        # sample the hyper-parameter config specific to the clients
        if self._cfg.hpo.fedex.psn:
            self._policy_net.train()
            self._pn_optimizer.zero_grad()
            self._theta = self._policy_net(self._client_encodings)
        for rcv_idx in receiver:
            if self._cfg.hpo.fedex.psn:
                cfg_idx, sampled_cfg = self.sample([
                    theta[rcv_idx - 1].detach().cpu().numpy()
                    for theta in self._theta
                ])
            else:
                cfg_idx, sampled_cfg = self.sample(self._theta)
            content = {
                'model_param': model_para,
                "arms": cfg_idx,
                'hyperparam': sampled_cfg
            }
            self.comm_manager.send(
                Message(msg_type=msg_type,
                        sender=self.ID,
                        receiver=[rcv_idx],
                        state=self.state,
                        content=content))
        if self._cfg.federate.online_aggr:
            for idx in range(self.model_num):
                self.aggregators[idx].reset()

        if filter_unseen_clients:
            # restore the state of the unseen clients within sampler
            self.sampler.change_state(self.unseen_clients_id, 'seen')

    def callback_funcs_model_para(self, message: Message):
        round, sender, content = message.state, message.sender, message.content
        self.sampler.change_state(sender, 'idle')
        # For a new round
        if round not in self.msg_buffer['train'].keys():
            self.msg_buffer['train'][round] = dict()

        self.msg_buffer['train'][round][sender] = content

        if self._cfg.federate.online_aggr:
            self.aggregator.inc(tuple(content[0:2]))

        return self.check_and_move_on()

    def update_policy(self, feedbacks):
        """Update the policy. This implementation is borrowed from the
        open-sourced FedEx (
        https://github.com/mkhodak/FedEx/blob/ \
        150fac03857a3239429734d59d319da71191872e/hyper.py#L151)
        Arguments:
            feedbacks (list): each element is a dict containing "arms" and
            necessary feedback.
        """

        index = [elem['arms'] for elem in feedbacks]
        cids = [elem['client_id'] for elem in feedbacks]
        before = np.asarray(
            [elem['val_avg_loss_before'] for elem in feedbacks])
        after = np.asarray([elem['val_avg_loss_after'] for elem in feedbacks])
        weight = np.asarray([elem['val_total'] for elem in feedbacks],
                            dtype=np.float64)
        weight /= np.sum(weight)

        if self._trace['refine']:
            trace = self.trace('refine')
            if self._diff:
                trace -= self.trace('global')
            baseline = discounted_mean(trace, self._baseline)
        else:
            baseline = 0.0
        self._trace['global'].append(np.inner(before, weight))
        self._trace['refine'].append(np.inner(after, weight))
        if self._stop_exploration:
            self._trace['entropy'].append(0.0)
            self._trace['mle'].append(1.0)
            return

        if self._cfg.hpo.fedex.psn:
            # policy gradients
            pg_obj = .0
            for i, theta in enumerate(self._theta):
                for idx, cidx, s, w in zip(
                        index, cids, after - before if self._diff else after,
                        weight):
                    pg_obj += w * -1.0 * (s - baseline) * torch.log(
                        torch.clip(theta[cidx][idx[i]], min=1e-8, max=1.0))
            pg_loss = -1.0 * pg_obj
            pg_loss.backward()
            self._pn_optimizer.step()
            self._policy_net.eval()
            thetas4stat = [
                theta.detach().cpu().numpy()
                for theta in self._policy_net(self._client_encodings)
            ]
        else:
            for i, (z, theta) in enumerate(zip(self._z, self._theta)):
                grad = np.zeros(len(z))
                for idx, s, w in zip(index,
                                     after - before if self._diff else after,
                                     weight):
                    grad[idx[i]] += w * (s - baseline) / theta[idx[i]]
                if self._sched == 'adaptive':
                    self._store[i] += norm(grad, float('inf'))**2
                    denom = np.sqrt(self._store[i])
                elif self._sched == 'aggressive':
                    denom = 1.0 if np.all(
                        grad == 0.0) else norm(grad, float('inf'))
                elif self._sched == 'auto':
                    self._store[i] += 1.0
                    denom = np.sqrt(self._store[i])
                elif self._sched == 'constant':
                    denom = 1.0
                elif self._sched == 'scale':
                    denom = 1.0 / np.sqrt(2.0 * np.log(len(grad))) if len(
                        grad) > 1 else float('inf')
                else:
                    raise NotImplementedError
                eta = self._eta0[i] / denom
                z -= eta * grad
                z -= logsumexp(z)
                self._theta[i] = np.exp(z)
            thetas4stat = self._theta

        self._trace['entropy'].append(self.entropy(thetas4stat))
        self._trace['mle'].append(self.mle(thetas4stat))
        if self._trace['entropy'][-1] < self._cutoff:
            self._stop_exploration = True

        logger.info(
            'Server: Updated policy as {} with entropy {:f} and mle {:f}'.
            format(thetas4stat, self._trace['entropy'][-1],
                   self._trace['mle'][-1]))

    def check_and_move_on(self,
                          check_eval_result=False,
                          min_received_num=None):
        """
        To check the message_buffer, when enough messages are receiving,
        trigger some events (such as perform aggregation, evaluation,
        and move to the next training round)
        """
        if min_received_num is None:
            min_received_num = self._cfg.federate.sample_client_num
        assert min_received_num <= self.sample_client_num

        if check_eval_result:
            min_received_num = len(list(self.comm_manager.neighbors.keys()))

        move_on_flag = True  # To record whether moving to a new training
        # round or finishing the evaluation
        if self.check_buffer(self.state, min_received_num, check_eval_result):

            if not check_eval_result:  # in the training process
                mab_feedbacks = list()
                # Get all the message
                train_msg_buffer = self.msg_buffer['train'][self.state]
                for model_idx in range(self.model_num):
                    model = self.models[model_idx]
                    aggregator = self.aggregators[model_idx]
                    msg_list = list()
                    for client_id in train_msg_buffer:
                        if self.model_num == 1:
                            msg_list.append(
                                tuple(train_msg_buffer[client_id][0:2]))
                        else:
                            train_data_size, model_para_multiple = \
                                train_msg_buffer[client_id][0:2]
                            msg_list.append((train_data_size,
                                             model_para_multiple[model_idx]))

                        # collect feedbacks for updating the policy
                        if model_idx == 0:
                            mab_feedbacks.append(
                                train_msg_buffer[client_id][2])

                    # Trigger the monitor here (for training)
                    self._monitor.calc_model_metric(self.model.state_dict(),
                                                    msg_list,
                                                    rnd=self.state)

                    # Aggregate
                    agg_info = {
                        'client_feedback': msg_list,
                        'recover_fun': self.recover_fun
                    }
                    result = aggregator.aggregate(agg_info)
                    model.load_state_dict(result, strict=False)
                    # aggregator.update(result)

                # update the policy
                self.update_policy(mab_feedbacks)

                self.state += 1
                if self.state % self._cfg.eval.freq == 0 and self.state != \
                        self.total_round_num:
                    #  Evaluate
                    logger.info(
                        'Server: Starting evaluation at round {:d}.'.format(
                            self.state))
                    self.eval()

                if self.state < self.total_round_num:
                    # Move to next round of training
                    logger.info(
                        f'----------- Starting a new training round (Round '
                        f'#{self.state}) -------------')
                    # Clean the msg_buffer
                    self.msg_buffer['train'][self.state - 1].clear()

                    self.broadcast_model_para(
                        msg_type='model_para',
                        sample_client_num=self.sample_client_num)
                else:
                    # Final Evaluate
                    logger.info('Server: Training is finished! Starting '
                                'evaluation.')
                    self.eval()

            else:  # in the evaluation process
                # Get all the message & aggregate
                formatted_eval_res = self.merge_eval_results_from_all_clients()
                self.history_results = merge_dict_of_results(
                    self.history_results, formatted_eval_res)
                self.check_and_save()
        else:
            move_on_flag = False

        return move_on_flag

    def check_and_save(self):
        """
        To save the results and save model after each evaluation
        """
        # early stopping
        should_stop = False

        if "Results_weighted_avg" in self.history_results and \
                self._cfg.eval.best_res_update_round_wise_key in \
                self.history_results['Results_weighted_avg']:
            should_stop = self.early_stopper.track_and_check(
                self.history_results['Results_weighted_avg'][
                    self._cfg.eval.best_res_update_round_wise_key])
        elif "Results_avg" in self.history_results and \
                self._cfg.eval.best_res_update_round_wise_key in \
                self.history_results['Results_avg']:
            should_stop = self.early_stopper.track_and_check(
                self.history_results['Results_avg'][
                    self._cfg.eval.best_res_update_round_wise_key])
        else:
            should_stop = False

        if should_stop:
            self.state = self.total_round_num + 1

        if should_stop or self.state == self.total_round_num:
            logger.info('Server: Final evaluation is finished! Starting '
                        'merging results.')
            # last round
            self.save_best_results()

            if self._cfg.federate.save_to != '':
                # save the policy
                ckpt = dict()
                if self._cfg.hpo.fedex.psn:
                    psn_pi_ckpt_path = self._cfg.federate.save_to[:self._cfg.
                                                                  federate.
                                                                  save_to.
                                                                  rfind(
                                                                      '.'
                                                                  )] + \
                                       "_pfedex.pt"
                    torch.save(
                        {
                            'client_encodings': self._client_encodings,
                            'policy_net': self._policy_net.state_dict()
                        }, psn_pi_ckpt_path)
                else:
                    z_list = [z.tolist() for z in self._z]
                    ckpt['z'] = z_list
                    ckpt['store'] = self._store
                ckpt['stop'] = self._stop_exploration
                ckpt['global'] = self.trace('global').tolist()
                ckpt['refine'] = self.trace('refine').tolist()
                ckpt['entropy'] = self.trace('entropy').tolist()
                ckpt['mle'] = self.trace('mle').tolist()
                pi_ckpt_path = self._cfg.federate.save_to[:self._cfg.federate.
                                                          save_to.rfind(
                                                              '.'
                                                          )] + "_fedex.yaml"
                with open(pi_ckpt_path, 'w') as ops:
                    yaml.dump(ckpt, ops)

            if self.model_num > 1:
                model_para = [model.state_dict() for model in self.models]
            else:
                model_para = self.model.state_dict()
            self.comm_manager.send(
                Message(msg_type='finish',
                        sender=self.ID,
                        receiver=list(self.comm_manager.neighbors.keys()),
                        state=self.state,
                        content=model_para))

        if self.state == self.total_round_num:
            # break out the loop for distributed mode
            self.state += 1
