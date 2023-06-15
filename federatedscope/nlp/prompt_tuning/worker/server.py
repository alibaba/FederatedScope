import os
import sys
import logging
import pickle
import numpy as np
import torch
import torch.distributed as dist
import json
from lm_eval import evaluator
from federatedscope.core.message import Message
from federatedscope.core.auxiliaries.model_builder import get_model
from federatedscope.core.auxiliaries.trainer_builder import get_trainer
from federatedscope.core.auxiliaries.sampler_builder import get_sampler
from federatedscope.core.auxiliaries.utils import merge_dict_of_results
from federatedscope.core.workers import Server
from federatedscope.nlp.prompt_tuning.dataset.utils import setup_tokenizer
from federatedscope.nlp.prompt_tuning.trainer.utils import merge_param_dict
from federatedscope.nlp.prompt_tuning.model.model import LMEvalModel

logger = logging.getLogger(__name__)
LM_EVAL_TASK_NAME_MAPPING = {'web_questions': 'webqs'}


class PLServer(Server):
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
        super(PLServer, self).__init__(ID=ID,
                                       state=state,
                                       config=config,
                                       data=data,
                                       model=model,
                                       client_num=client_num,
                                       total_round_num=total_round_num,
                                       device=device,
                                       strategy=strategy,
                                       unseen_clients_id=unseen_clients_id,
                                       **kwargs)

        self.local_rank = dist.get_rank() if self._cfg.use_ddp else 0
        if self._cfg.federate.pl_init_kd:
            self.init_cfg = self._cfg.clone()
            self.init_cfg.defrost()
            self.init_cfg.merge_from_file(self._cfg.federate.pl_kd_cfg_file)
            self.init_cfg.model.client_freeze_param = []
            self.init_cfg.freeze()
            self.init_model = get_model(self.init_cfg.model,
                                        local_data=None,
                                        backend=self.init_cfg.backend,
                                        role='client')
            self.init_trainer = get_trainer(model=self.init_model,
                                            data=data,
                                            device=device,
                                            config=self.init_cfg,
                                            monitor=self._monitor)
            self.init_trainer.ctx.teacher_model = get_model(
                self.init_cfg.model,
                local_data=None,
                backend=self.init_cfg.backend,
                role='server')

        if self._cfg.federate.make_global_eval:
            self.global_cfg = self._cfg.clone()
            self.global_cfg.defrost()
            self.global_cfg.merge_from_file(
                self._cfg.federate.pl_global_cfg_file)
            self.global_cfg.freeze()
            self.trainer = get_trainer(
                model=self.model,
                data=self.data,
                device=self.device,
                config=self.global_cfg,
                only_for_eval=not self._cfg.federate.make_global_train,
                monitor=self._monitor)
            self.trainers = [self.trainer]

            self.client_model = get_model(self._cfg.model,
                                          local_data=None,
                                          backend=self._cfg.backend,
                                          role='client')
            self.client_trainer = get_trainer(model=self.client_model,
                                              data=data,
                                              device=device,
                                              config=self._cfg,
                                              monitor=self._monitor)
            self.trainer.ctx.teacher_model = self.client_model
            self.client_trainer.ctx.teacher_model = self.model
            self.best_val_res = np.inf

    def _perform_federated_aggregation(self):
        train_msg_buffer = dict(
            sorted(self.msg_buffer['train'][self.state].items(),
                   key=lambda x: x[0]))
        msg_list = list()
        for client_id in train_msg_buffer:
            msg_list.append(train_msg_buffer[client_id])

        # Aggregate
        aggregated_num = len(msg_list)
        agg_info = {
            'client_feedback': [[x['sample_size'], x['model_para']]
                                for x in msg_list],
            'recover_fun': self.recover_fun,
        }
        avg_model = self.aggregator.aggregate(agg_info)
        if self._cfg.federate.make_global_eval:  # When server and
            # client have different personalized params, params of
            # avg_model that are also in server's personalized params
            # should be filtered out first to avoid overwriting.
            self.trainer.update(avg_model)
            self.client_trainer.update(avg_model)
        else:
            merged_param = merge_param_dict(self.model.state_dict().copy(),
                                            avg_model)
            self.model.load_state_dict(merged_param, strict=False)

        return aggregated_num

    def broadcast_model_para(self,
                             msg_type='model_para',
                             sample_client_num=-1,
                             filter_unseen_clients=True):
        if filter_unseen_clients:
            self.sampler.change_state(self.unseen_clients_id, 'unseen')

        if sample_client_num > 0:
            receiver = self.sampler.sample(size=sample_client_num)
        else:
            receiver = list(self.comm_manager.neighbors.keys())
            if msg_type == 'model_para':
                self.sampler.change_state(receiver, 'working')

        skip_broadcast = self._cfg.federate.method in ['local', 'global']
        model_para = {}
        if not skip_broadcast:
            if self._cfg.federate.pl_init_kd and self.state == 0:
                logger.info('Conducting initial KD training')

                if self.init_cfg.federate.pl_alter_train:
                    self.init_trainer.update_alter_stage('model')
                    self.init_trainer.train()
                    self.init_trainer.update_alter_stage('prompt')
                train_metrics = self.init_trainer.train()[-1]
                formatted_train_res = self._monitor.format_eval_res(
                    train_metrics,
                    rnd=self.state,
                    role='Client #',
                    return_raw=self.init_cfg.federate.make_global_eval)
                logger.info(formatted_train_res)
                model_para = self.init_trainer.get_model_para()
            else:
                if self._cfg.federate.make_global_eval:
                    if self._cfg.federate.pl_ret_avg_model:
                        model_para = self.client_trainer.get_model_para()
                    else:  # avoid undesired param overwriting due to
                        # different personalized params between server
                        # and client
                        model_para = self.trainer.get_model_para()
                else:
                    model_para = self.model.state_dict()

        self.comm_manager.send(
            Message(msg_type=msg_type,
                    sender=self.ID,
                    receiver=receiver,
                    state=self.state,
                    content={'model_para': model_para}))

        if filter_unseen_clients:
            self.sampler.change_state(self.unseen_clients_id, 'seen')

    def check_and_move_on(self,
                          check_eval_result=False,
                          min_received_num=None):
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
            if not check_eval_result:  # in the training process
                # Receiving enough feedback in the training process
                aggregated_num = self._perform_federated_aggregation()
                if aggregated_num == -1:
                    return move_on_flag

                self.state += 1
                if self.state % self._cfg.eval.freq == 0 and \
                        self.state < self.total_round_num:
                    #  Evaluate
                    logger.info(
                        'Server #{:d}: Starting evaluation at the end of '
                        'round {:d}.'.format(self.ID, self.state))
                    self.eval()

                if self.state < self.total_round_num:
                    # Move to next round of training
                    logger.info(
                        '----------- Starting a new training round (Round '
                        '#{:d}/{:d}) -------------'.format(
                            self.state + 1,
                            self._cfg.federate.total_round_num))
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

    def merge_eval_results_from_all_clients(self, final_round=False):
        state = self.state if not final_round else self.state - 1
        eval_msg_buffer = self.msg_buffer['eval'][state]

        if 'group_avg' in self._cfg.eval.report:
            metrics_all_clients = eval_msg_buffer
        else:
            metrics_all_clients = dict()
            for each_client in eval_msg_buffer:
                client_eval_results = eval_msg_buffer[each_client]
                for key in client_eval_results.keys():
                    res = client_eval_results[key]
                    if isinstance(res, dict):
                        for k, v in res.items():
                            cur_key = key + '_' + k
                            if key not in metrics_all_clients:
                                metrics_all_clients[cur_key] = list()
                            metrics_all_clients[cur_key].append(float(v))
                    else:
                        if key not in metrics_all_clients:
                            metrics_all_clients[key] = list()
                        metrics_all_clients[key].append(float(res))

        formatted_logs = self._monitor.format_eval_res(
            metrics_all_clients,
            rnd=self.state + 1,
            role='Server #',
            forms=self._cfg.eval.report)
        logger.info(formatted_logs)
        self._monitor.save_formatted_results(formatted_logs)

        return formatted_logs

    def trigger_for_start(self):
        if self.check_client_join_in():
            if self._cfg.federate.use_ss:
                self.broadcast_client_address()

            # get sampler
            if 'client_resource' in self._cfg.federate.join_in_info:
                client_resource = [
                    self.join_in_info[client_index]['client_resource']
                    for client_index in np.arange(1, self.client_num + 1)
                ]
            else:
                if self._cfg.backend == 'torch':
                    model_size = sys.getsizeof(pickle.dumps(
                        self.model)) / 1024.0 * 8.
                else:
                    # TODO: calculate model size for TF Model
                    model_size = 1.0
                    logger.warning(f'The calculation of model size in backend:'
                                   f'{self._cfg.backend} is not provided.')

                client_resource = [
                    model_size / float(x['communication']) +
                    float(x['computation']) / 1000.
                    for x in self.client_resource_info
                ] if self.client_resource_info is not None else None

            if self.sampler is None:
                self.sampler = get_sampler(
                    sample_strategy=self._cfg.federate.sampler,
                    client_num=self.client_num,
                    client_info=client_resource)

            # change the deadline if the asyn.aggregator is `time up`
            if self._cfg.asyn.use and self._cfg.asyn.aggregator == 'time_up':
                self.deadline_for_cur_round = self.cur_timestamp + \
                                               self._cfg.asyn.time_budget

            logger.info('----------- Starting training (Round '
                        '#{:d}/{:d}) -------------'.format(
                            self.state + 1,
                            self._cfg.federate.total_round_num))
            self.broadcast_model_para(msg_type='model_para',
                                      sample_client_num=self.sample_client_num)

    def eval(self):
        if self._cfg.federate.make_global_eval:
            # Perform training in server
            if self._cfg.federate.make_global_train:
                # server
                train_metrics = self.trainer.train()[-1]
                formatted_train_res = self._monitor.format_eval_res(
                    train_metrics,
                    rnd=self.state,
                    role='Server #',
                    return_raw=self._cfg.federate.make_global_eval)
                logger.info(formatted_train_res)

            # Evaluate on val dataset
            save_path = os.path.join(self._cfg.federate.pl_save_to,
                                     'best_model.pt')
            if not self._cfg.federate.skip_local_train:
                val_metrics = self.trainer.evaluate(
                    target_data_split_name='val')
                formatted_val_res = self._monitor.format_eval_res(
                    val_metrics,
                    rnd=self.state,
                    role='Server #',
                    return_raw=self._cfg.federate.make_global_eval)
                logger.info(formatted_val_res)

                cur_val_res = val_metrics['val_avg_loss']
                if cur_val_res < self.best_val_res:
                    self.best_val_res = cur_val_res
                    if self._cfg.federate.pl_save_to and self.local_rank == 0:
                        logger.info(f'Best val res {cur_val_res} obtained. '
                                    f'Model saved to {save_path}.')
                        ckpt = {
                            'round': self.state,
                            'val_res': cur_val_res,
                            'model': self.trainer.get_model_para(),
                            'client_model': self.client_trainer.
                            get_model_para(),
                        }
                        os.makedirs(self._cfg.federate.pl_save_to,
                                    exist_ok=True)
                        torch.save(ckpt, save_path)

                self.history_results = merge_dict_of_results(
                    self.history_results, formatted_val_res)
                self._monitor.save_formatted_results(formatted_val_res)

            # Evaluate on test dataset
            if self.state == self.total_round_num:
                if self._cfg.federate.ckpt_path and \
                        os.path.exists(self._cfg.federate.ckpt_path):
                    best_ckpt = torch.load(self._cfg.federate.ckpt_path,
                                           map_location='cpu')
                    self.trainer.update(best_ckpt['model'])
                    logger.info(f"Loaded best model obtained in round "
                                f"{best_ckpt['round']} "
                                f"({best_ckpt['val_res']}).")

                elif os.path.exists(save_path):
                    best_ckpt = torch.load(save_path, map_location='cpu')
                    self.trainer.update(best_ckpt['model'])
                    logger.info(f"Loaded best model obtained in round "
                                f"{best_ckpt['round']} "
                                f"({best_ckpt['val_res']}).")

                tokenizer = setup_tokenizer(self._cfg)
                self.model.to(self.device)
                lm_eval_model = LMEvalModel(self.model, tokenizer, self.device)
                test_results = evaluator.simple_evaluate(
                    model=lm_eval_model,
                    tasks=[
                        LM_EVAL_TASK_NAME_MAPPING.get(
                            self._cfg.data.dataset_name,
                            self._cfg.data.dataset_name)
                    ],
                    batch_size=128,
                    no_cache=True,
                )
                self.model.to('cpu')
                del test_results['config']['model']
                logger.info(evaluator.make_table(test_results))
                with open(os.path.join(self._cfg.outdir, 'test_results.json'),
                          'w') as f:
                    json.dump(test_results, f, indent=2)

            self.check_and_save()
        else:
            # Preform evaluation in clients
            self.broadcast_model_para(msg_type='evaluate')
