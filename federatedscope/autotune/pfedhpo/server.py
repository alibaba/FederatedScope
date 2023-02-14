import os
import json
import logging
import torch.nn
import yaml
from torch.nn import functional as F

from federatedscope.core.message import Message
from federatedscope.core.workers import Server
from federatedscope.autotune.pfedhpo.utils import *
from federatedscope.autotune.utils import parse_search_space

logger = logging.getLogger(__name__)


class pFedHPOServer(Server):
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
        with open(config.hpo.pfedhpo.ss, 'r') as ips:
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

        super(pFedHPOServer,
              self).__init__(ID, state, config, data, model, client_num,
                             total_round_num, device, strategy, **kwargs)

        os.makedirs(self._cfg.hpo.working_folder, exist_ok=True)
        self.discrete = self._cfg.hpo.pfedhpo.discrete
        # prepare search space and bounds
        self._ss = parse_search_space(self._cfg.hpo.pfedhpo.ss)
        self.dim = len(self._ss)
        self.bounds = np.asarray([(0., 1.) for _ in self._ss])
        self.pbounds = {}

        if not self.discrete:
            for k, v in self._ss.items():
                if not (hasattr(v, 'lower') and hasattr(v, 'upper')):
                    raise ValueError("Unsupported hyper type {}".format(
                        type(v)))
                else:
                    if v.log:
                        l, u = np.log10(v.lower), np.log10(v.upper)
                    else:
                        l, u = v.lower, v.upper
                    self.pbounds[k] = (l, u)
        else:
            for k, v in self._ss.items():
                if not (hasattr(v, 'lower') and hasattr(v, 'upper')):
                    if hasattr(v, 'choices'):
                        self.pbounds[k] = list(v.choices)
                    else:
                        raise ValueError("Unsupported hyper type {}".format(
                            type(v)))
                else:
                    if v.log:
                        l, u = np.log10(v.lower), np.log10(v.upper)
                    else:
                        l, u = v.lower, v.upper
                    N_samp = 10
                    samp = []
                    for i in range(N_samp):
                        samp.append((u - l) / N_samp * i + l)
                    self.pbounds[k] = samp

        # prepare hyper-net
        self.client2idx = None

        if not self.discrete:
            self.var = 0.01
            dist = MultivariateNormal(
                loc=torch.zeros(len(self.pbounds)),
                covariance_matrix=torch.eye(len(self.pbounds)) * self.var)
            self.logprob_max = dist.log_prob(dist.sample() * 0)
        else:
            self.logprob_max = 1.

        encoding_tensor = []
        for i in range(self._cfg.federate.client_num + 1):
            p = os.path.join(self._cfg.hpo.working_folder,
                             'client_%d_encoding.pt' % i)
            if os.path.exists(p):
                t = torch.load(p)
                encoding_tensor.append(t)
        encoding_tensor = torch.stack(encoding_tensor)

        if not self.discrete:
            self.HyperNet = HyperNet(encoding=encoding_tensor,
                                     num_params=len(self.pbounds),
                                     n_clients=client_num,
                                     device=self._cfg.device,
                                     var=self.var).to(self._cfg.device)
        else:
            self.HyperNet = DisHyperNet(
                encoding=encoding_tensor, cands=self.pbounds,
                n_clients=client_num, device=self._cfg.device,)\
                .to(self._cfg.device)

        self.saved_models = [None] * self._cfg.hpo.pfedhpo.\
            target_fl_total_round
        self.opt_params = self.HyperNet.EncNet.parameters()

        self.opt = torch.optim.Adam([
            {
                'params': self.HyperNet.EncNet.parameters(),
                'lr': 0.001,
                'weight_decay': 1e-4
            },
        ])

        with open(
                os.path.join(self._cfg.hpo.working_folder,
                             'anchor_eval_results.json'), 'r') as f:
            self.anchor_res = json.load(f)
        self.anchor_res_smooth = None

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
            self.receiver = self.sampler.sample(size=sample_client_num)
        else:
            # broadcast to all clients
            self.receiver = list(self.comm_manager.neighbors.keys())
            if msg_type == 'model_para':
                self.sampler.change_state(self.receiver, 'working')

        if msg_type == 'model_para':
            # random sample start round and load saved global model
            self.start_round = np.random.randint(
                1, self._cfg.hpo.pfedhpo.target_fl_total_round)
            logger.info('==> Sampled start round: %d' % self.start_round)
            ckpt_path = os.path.join(
                self._cfg.hpo.working_folder,
                'temp_model_round_%d.pt' % self.start_round)
            if self.model_num > 1:
                raise NotImplementedError
            else:
                self.model.load_state_dict(torch.load(ckpt_path))

        model_para = self.model.state_dict()

        # generate hyper-params for all clients
        if not self.client2idx:
            client2idx = {}
            _all_clients = list(self.comm_manager.neighbors.keys())
            for i, k in zip(range(len(_all_clients)), _all_clients):
                client2idx[k] = i
            self.client2idx = client2idx

        if not self.discrete:
            var_max = 2.0
            var_min = 0.1
            var = var_max + (var_min - var_max) / (
                0.5 * self.total_round_num) * self.state
            if var < 0.1:
                var = 0.1
            self.HyperNet.var = var
            param_raw, self.logprob, self.entropy = self.HyperNet()
            xs = param_raw.detach().cpu().numpy()
        else:
            logits, self.enc_loss = self.HyperNet()
            # self.logprob = [None] * len(self.receiver)
            self.logprob = [None] * len(self.client2idx)
            self.p_idx = {}
            for k in self.pbounds.keys():
                # self.p_idx[k] = [None] * len(self.receiver)
                self.p_idx[k] = [None] * len(self.client2idx)

        # sample the hyper-parameter config specific to the clients
        self.sampled = False
        for rcv_idx in self.receiver:
            if not self.discrete:
                sampled_cfg = map_value_to_param(xs[self.client2idx[rcv_idx]],
                                                 self.pbounds, self._ss)
            else:
                client_logprob = 0.
                sampled_cfg = {}

                for i, (k, v) in zip(range(len(self.pbounds)),
                                     self.pbounds.items()):
                    probs = logits[i][self.client2idx[rcv_idx]]
                    m = torch.distributions.Categorical(probs)

                    idx = m.sample()
                    p = v[idx.item()]
                    if hasattr(self._ss[k], 'log') and self._ss[k].log:
                        p = 10**p
                    if 'int' in str(type(self._ss[k])).lower():
                        sampled_cfg[k] = int(p)
                    else:
                        sampled_cfg[k] = float(p)

                    log_prob = m.log_prob(idx)
                    client_logprob += log_prob
                    self.p_idx[k][self.client2idx[rcv_idx]] = torch.argmax(
                        probs)

                self.logprob[self.client2idx[rcv_idx]] = client_logprob / len(
                    self.pbounds)

            content = {'model_param': model_para, 'hyper_param': sampled_cfg}
            self.comm_manager.send(
                Message(msg_type=msg_type,
                        sender=self.ID,
                        receiver=[rcv_idx],
                        state=self.state,
                        content=content))
        if self._cfg.federate.online_aggr:
            try:
                for idx in range(self.model_num):
                    self.aggregators[idx].reset()
            except:
                pass

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
            try:
                self.aggregator.inc(tuple(content[0:2]))
            except:
                pass

        return self.check_and_move_on()

    def update_policy(self):
        key1 = 'Results_weighted_avg'
        key2 = 'val_acc'

        if 'twitter' in str(self._cfg.data.type).lower():
            anchor_res_start = \
                self.anchor_res['Results_raw']['test_acc'][self.start_round-1]
            res_end = \
                self.history_results['Results_weighted_avg']['test_acc'][-1]
        else:
            anchor_res_start = self.anchor_res[key1][key2][self.start_round -
                                                           1]
            res_end = self.history_results[key1][key2][-1]

        if not self.discrete:
            reward = np.maximum(0, res_end - anchor_res_start)
            losses = -reward * self.logprob

        else:
            reward = np.maximum(0, res_end - anchor_res_start) \
                     * anchor_res_start
            self.logprob = torch.stack(self.logprob, dim=-1)
            losses = F.relu(-reward * self.logprob * 100)

        self.opt.zero_grad()
        loss = losses.mean()
        loss.backward()
        nn.utils.clip_grad_norm_(self.opt_params, max_norm=10, norm_type=2)
        self.opt.step()

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
                mab_feedbacks = dict()
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
                            mab_feedbacks[client_id] = train_msg_buffer[
                                client_id][2]

                    # Trigger the monitor here (for training)
                    if 'dissim' in self._cfg.eval.monitoring:
                        from federatedscope.core.auxiliaries.utils import \
                            calc_blocal_dissim
                        # TODO: fix load_state_dict
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
                self.fb = mab_feedbacks

                self.state += 1
                #  Evaluate
                logger.info(
                    'Server: Starting evaluation at begin of round {:d}.'.
                    format(self.state))
                self.eval()

            else:  # in the evaluation process
                # Get all the message & aggregate
                logger.info('-' * 30)
                formatted_eval_res = self.merge_eval_results_from_all_clients()
                self.history_results = merge_dict(self.history_results,
                                                  formatted_eval_res)
                self.check_and_save()

                if self.state < self.total_round_num:
                    if len(self.history_results) > 0:
                        logger.info('=' * 10 + ' updating hypernet at round ' +
                                    str(self.state) + ' ' + '=' * 10)
                        self.update_policy()

                    # Move to next round of training
                    logger.info(
                        f'----------- Starting a new training round (Round '
                        f'#{self.state}) -------------')
                    logger.info(self._cfg.device)
                    # Clean the msg_buffer
                    self.msg_buffer['train'][self.state - 1].clear()

                    self.broadcast_model_para(
                        msg_type='model_para',
                        sample_client_num=self.sample_client_num)

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

        _path = os.path.join(self._cfg.hpo.working_folder,
                             'hyperNet_encoding.pt')
        hyper_enc = {
            'hyperNet': self.HyperNet.state_dict(),
        }
        torch.save(hyper_enc, _path)

        if should_stop or self.state == self.total_round_num:
            logger.info('Server: Final evaluation is finished! Starting '
                        'merging results.')
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
                                                          )] + "_pfedhpo.yaml"
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
