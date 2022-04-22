import os
import logging

import yaml

import numpy as np
from scipy.special import logsumexp

from federatedscope.core.message import Message
from federatedscope.core.worker import Server
from federatedscope.autotune import Continuous, Discrete, split_raw_config
from federatedscope.autotune.algos import random_search
from federatedscope.core.auxiliaries.utils import formatted_logging, merge_dict

logger = logging.getLogger(__name__)


class FedExServer(Server):
    """Some code snippets are borrowed from the open-sourced FedEx (https://github.com/mkhodak/FedEx)
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
        _, tbd_config = split_raw_config(ss)
        if config.hpo.fedex.flatten_ss:
            self._cfsp = [random_search(tbd_config, config.hpo.fedex.num_arms)]
        else:
            # TODO: cross-producting the grids of all aspects
            # in which case, self._cfsp will be a list with length equal to #aspects
            pass
        sizes = [len(cand_set) for cand_set in self._cfsp]
        # TODO: support other step size
        eta0 = 'auto'
        self._eta0 = [np.sqrt(2.0 * np.log(size)) if eta0 == 'auto' else eta0 for size in sizes]
        self._z = [np.full(size, -np.log(size)) for size in sizes]
        self._theta = [np.exp(z) for z in self._z]
        self._store = [0.0 for _ in sizes]

        super(FedExServer, self).__init__(ID, state, config, data, model, client_num, total_round_num, device, strategy, **kwargs)

    def broadcast_model_para(self,
                             msg_type='model_para',
                             sample_client_num=-1):
        """
        To broadcast the message to all clients or sampled clients
        """
        if sample_client_num > 0:
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

        if self.model_num > 1:
            model_para = [model.state_dict() for model in self.models]
        else:
            model_para = self.model.state_dict()

        # sample the hyper-parameter config specific to the clients
        
        for rcv_idx in receiver:
            cfg_idx = [np.random.choice(len(theta), p=theta) for theta in self._theta]
            sampled_cfg = [sps[i] for i, sps in zip(cfg_idx, self._cfsp)]
            content = {'model_param': model_para, "arms": cfg_idx, 'hyperparam': sampled_cfg}
            self.comm_manager.send(
                Message(msg_type=msg_type,
                        sender=self.ID,
                        receiver=[rcv_idx],
                        state=self.state,
                        content=content))
        if self._cfg.federate.online_aggr:
            for idx in range(self.model_num):
                self.aggregators[idx].reset()

    def callback_funcs_model_para(self, message: Message):
        round, sender, content = message.state, message.sender, message.content
        # For a new round
        if round not in self.msg_buffer['train'].keys():
            self.msg_buffer['train'][round] = dict()

        self.msg_buffer['train'][round][sender] = list(content)

        if self._cfg.federate.online_aggr:
            self.aggregator.inc(tuple(content[0:2]))
        self.check_and_move_on()

    def update_policy(self, feedbacks):
        """Update the policy. This implementation is borrowed from the open-sourced FedEx (https://github.com/mkhodak/FedEx/blob/150fac03857a3239429734d59d319da71191872e/hyper.py#L151)
        Arguments:
            feedbacks (list): each element is a tuple in the form (sample_size, arms, loss)
        """

        index = [tp[1] for tp in feedbacks]
        weight = np.asarray([tp[0] for tp in feedbacks], dtype=np.float64)
        weight /= np.sum(weight)
        if self._cfg.hpo.fedex.diff:
            # TODO: let the client provide before loss and after loss together
            before, after = None, None
        else:
            after = [tp[2] for tp in feedbacks]
        # TODO: enable baseline for variance reduction
        baseline = .0

        for i, (z, theta) in enumerate(zip(self._z, self._theta)):
            grad = np.zeros(len(z))
            for idx, s, w in zip(index, after-before if self._cfg.hpo.fedex.diff else after, weight):
                grad[idx[i]] += w * (s - baseline) / theta[idx[i]]
            if self._cfg.hpo.fedex.sched == 'adaptive':
                self._store[i] += norm(grad, float('inf')) ** 2
                denom = np.sqrt(self._store[i])
            elif self._cfg.hpo.fedex.sched == 'aggressive':
                denom = 1.0 if np.all(grad == 0.0) else norm(grad, float('inf'))
            elif self._cfg.hpo.fedex.sched == 'auto':
                self._store[i] += 1.0
                denom = np.sqrt(self._store[i])
            elif self._cfg.hpo.fedex.sched == 'constant':
                denom = 1.0
            elif self._cfg.hpo.fedex.sched == 'scale':
                denom = 1.0 / np.sqrt(2.0 * np.log(len(grad))) if len(grad) > 1 else float('inf')
            else:
                raise NotImplementedError
            eta = self._eta0[i] / denom
            z -= eta * grad
            z -= logsumexp(z)
            self._theta[i] = np.exp(z)

    def check_and_move_on(self, check_eval_result=False):
        """
        To check the message_buffer, when enough messages are receiving, trigger some events (such as perform aggregation, evaluation, and move to the next training round)
        """

        if check_eval_result:
            # all clients are participating in evaluation
            minimal_number = self.client_num
        else:
            # sampled clients are participating in training
            minimal_number = self.sample_client_num

        if self.check_buffer(self.state, minimal_number, check_eval_result):

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
                            msg_list.append(tuple(train_msg_buffer[client_id][0:2]))
                        else:
                            train_data_size, model_para_multiple = train_msg_buffer[
                                client_id][0:2]
                            msg_list.append((train_data_size,
                                             model_para_multiple[model_idx]))

                        if model_idx == 0:
                            # temporarily, we consider training loss
                            # TODO: use validation loss and sample size
                            mab_feedbacks.append((train_msg_buffer[client_id][0], train_msg_buffer[client_id][2], train_msg_buffer[client_id][3]))

                    # Trigger the monitor here (for training)
                    if 'dissim' in self._cfg.eval.monitoring:
                        B_val = calc_blocal_dissim(
                            model.load_state_dict(strict=False), msg_list)
                        formatted_eval_res = formatted_logging(B_val,
                                                               rnd=self.state,
                                                               role='Server #')
                        logger.info(formatted_eval_res)

                    # Aggregate
                    agg_info = {
                        'client_feedback': msg_list,
                        'recover_fun': self.recover_fun
                    }
                    result = aggregator.aggregate(agg_info)
                    model.load_state_dict(result, strict=False)
                    #aggregator.update(result)

                # update the policy
                self.update_policy(mab_feedbacks)

                self.state += 1
                if self.state % self._cfg.eval.freq == 0 and self.state != self.total_round_num:
                    #  Evaluate
                    logger.info(
                        'Server #{:d}: Starting evaluation at round {:d}.'.
                        format(self.ID, self.state))
                    self.eval()

                if self.state < self.total_round_num:
                    # Move to next round of training
                    logger.info(
                        '----------- Starting a new training round (Round #{:d}) -------------'
                        .format(self.state))
                    # Clean the msg_buffer
                    self.msg_buffer['train'][self.state - 1].clear()

                    self.broadcast_model_para(
                        msg_type='model_para',
                        sample_client_num=self.sample_client_num)
                else:
                    # Final Evaluate
                    logger.info(
                        'Server #{:d}: Training is finished! Starting evaluation.'
                        .format(self.ID))
                    self.eval()

            else:  # in the evaluation process
                # Get all the message & aggregate
                formatted_eval_res = self.merge_eval_results_from_all_clients()
                self.history_results = merge_dict(self.history_results,
                                                  formatted_eval_res)
                self.check_and_save()

    def check_and_save(self):
        """
        To save the results and save model after each evaluation
        """
        # early stopping
        should_stop = False

        if "Results_weighted_avg" in self.history_results and \
                self._cfg.eval.best_res_update_round_wise_key in self.history_results['Results_weighted_avg']:
            should_stop = self.early_stopper.track_and_check(
                self.history_results['Results_weighted_avg'][
                    self._cfg.eval.best_res_update_round_wise_key])
        elif "Results_avg" in self.history_results and \
                self._cfg.eval.best_res_update_round_wise_key in self.history_results['Results_avg']:
            should_stop = self.early_stopper.track_and_check(
                self.history_results['Results_avg'][
                    self._cfg.eval.best_res_update_round_wise_key])
        else:
            should_stop = False

        if should_stop:
            self.state = self.total_round_num + 1

        if should_stop or self.state == self.total_round_num:
            logger.info(
                'Server #{:d}: Final evaluation is finished! Starting merging results.'
                .format(self.ID))
            # last round
            # TODO: call save with a specified frequency
            if self._cfg.federate.save_to != '':
                self.aggregator.save_model(self._cfg.federate.save_to,
                                           self.state)
                # save the policy
                with open(os.path.join(self._cfg.outdir, "policy.npy"), 'wb') as ops:
                    np.save(ops, self._z)
            formatted_best_res = formatted_logging(self.best_results,
                                                   rnd="Final",
                                                   role='Server #',
                                                   forms=["raw"])
            with open(os.path.join(self._cfg.outdir, "eval_results.log"),
                      "a") as outfile:
                outfile.write(str(formatted_best_res) + "\n")
            logger.info(formatted_best_res)

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
            #break out the loop for distributed mode
            self.state += 1
