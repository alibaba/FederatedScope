import os
import logging
from itertools import product

import yaml

import numpy as np
from numpy.linalg import norm
from scipy.special import logsumexp

from federatedscope.core.message import Message
from federatedscope.core.worker import Server
from federatedscope.core.auxiliaries.utils import merge_dict

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
        self._z = [np.full(size, -np.log(size)) for size in sizes]
        self._theta = [np.exp(z) for z in self._z]
        self._store = [0.0 for _ in sizes]
        self._stop_exploration = False
        self._trace = {
            'global': [],
            'refine': [],
            'entropy': [self.entropy()],
            'mle': [self.mle()]
        }

        super(FedExServer,
              self).__init__(ID, state, config, data, model, client_num,
                             total_round_num, device, strategy, **kwargs)

        if self._cfg.federate.restore_from != '':
            pi_ckpt_path = self._cfg.federate.restore_from[:self._cfg.federate.
                                                           restore_from.rfind(
                                                               '.'
                                                           )] + "_fedex.yaml"
            with open(pi_ckpt_path, 'r') as ips:
                ckpt = yaml.load(ips, Loader=yaml.FullLoader)
            self._z = [np.asarray(z) for z in ckpt['z']]
            self._theta = [np.exp(z) for z in self._z]
            self._store = ckpt['store']
            self._stop_exploration = ckpt['stop']
            self._trace = dict()
            self._trace['global'] = ckpt['global']
            self._trace['refine'] = ckpt['refine']
            self._trace['entropy'] = ckpt['entropy']
            self._trace['mle'] = ckpt['mle']

    def entropy(self):

        entropy = 0.0
        for probs in product(*(theta[theta > 0.0] for theta in self._theta)):
            prob = np.prod(probs)
            entropy -= prob * np.log(prob)
        return entropy

    def mle(self):

        return np.prod([theta.max() for theta in self._theta])

    def trace(self, key):
        '''returns trace of one of three tracked quantities
        Args:
            key (str): 'entropy', 'global', or 'refine'
        Returns:
            numpy vector with length equal to number of rounds up to now.
        '''

        return np.array(self._trace[key])

    def sample(self):
        """samples from configs using current probability vector"""

        # determine index
        if self._stop_exploration:
            cfg_idx = [theta.argmax() for theta in self._theta]
        else:
            cfg_idx = [
                np.random.choice(len(theta), p=theta) for theta in self._theta
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

        for rcv_idx in receiver:
            cfg_idx, sampled_cfg = self.sample()
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
                denom = 1.0 / np.sqrt(
                    2.0 * np.log(len(grad))) if len(grad) > 1 else float('inf')
            else:
                raise NotImplementedError
            eta = self._eta0[i] / denom
            z -= eta * grad
            z -= logsumexp(z)
            self._theta[i] = np.exp(z)

        self._trace['entropy'].append(self.entropy())
        self._trace['mle'].append(self.mle())
        if self._trace['entropy'][-1] < self._cutoff:
            self._stop_exploration = True

        logger.info(
            'Server #{:d}: Updated policy as {} with entropy {:f} and mle {:f}'
            .format(self.ID, self._theta, self._trace['entropy'][-1],
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
                    if 'dissim' in self._cfg.eval.monitoring:
                        from federatedscope.core.auxiliaries.utils import \
                            calc_blocal_dissim
                        B_val = calc_blocal_dissim(
                            model.load_state_dict(strict=False), msg_list)
                        formatted_eval_res = self._monitor.format_eval_res(
                            B_val, rnd=self.state, role='Server #')
                        logger.info(formatted_eval_res)

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
                        'Server #{:d}: Starting evaluation at round {:d}.'.
                        format(self.ID, self.state))
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
                    logger.info('Server #{:d}: Training is finished! Starting '
                                'evaluation.'.format(self.ID))
                    self.eval()

            else:  # in the evaluation process
                # Get all the message & aggregate
                formatted_eval_res = self.merge_eval_results_from_all_clients()
                self.history_results = merge_dict(self.history_results,
                                                  formatted_eval_res)
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
            logger.info('Server #{:d}: Final evaluation is finished! Starting '
                        'merging results.'.format(self.ID))
            # last round
            self.save_best_results()

            if self._cfg.federate.save_to != '':
                # save the policy
                ckpt = dict()
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
