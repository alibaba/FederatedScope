import copy
import json
import logging
import os
import gzip
import shutil
import datetime
from collections import defaultdict

import numpy as np

from federatedscope.core.auxiliaries.logging import logline_2_wandb_dict
from federatedscope.core.monitors.metric_calculator import MetricCalculator

try:
    import torch
except ImportError:
    torch = None

logger = logging.getLogger(__name__)

global_all_monitors = [
]  # used in standalone mode, to merge sys metric results for all workers


class Monitor(object):
    """
        Provide the monitoring functionalities such as formatting the
        evaluation results into diverse metrics.
        Besides the prediction related performance, the monitor also can
        track efficiency related metrics for a worker
    """
    SUPPORTED_FORMS = ['weighted_avg', 'avg', 'fairness', 'raw']

    def __init__(self, cfg, monitored_object=None):
        self.log_res_best = {}
        self.outdir = cfg.outdir
        self.use_wandb = cfg.wandb.use
        self.wandb_online_track = cfg.wandb.online_track if cfg.wandb.use \
            else False
        # self.use_tensorboard = cfg.use_tensorboard

        self.monitored_object = monitored_object
        self.metric_calculator = MetricCalculator(cfg.eval.metrics)

        # Obtain the whether the larger the better
        self.round_wise_update_key = cfg.eval.best_res_update_round_wise_key
        for mode in ['train', 'val', 'test']:
            if mode in self.round_wise_update_key:
                update_key = self.round_wise_update_key.split(f'{mode}_')[1]
        assert update_key in self.metric_calculator.eval_metric, \
            f'{update_key} not found in metrics.'
        self.the_larger_the_better = self.metric_calculator.eval_metric[
            update_key][1]

        # =======  efficiency indicators of the worker to be monitored =======
        # leveraged the flops counter provided by [fvcore](
        # https://github.com/facebookresearch/fvcore)
        self.total_model_size = 0  # model size used in the worker, in terms
        # of number of parameters
        self.flops_per_sample = 0  # average flops for forwarding each data
        # sample
        self.flop_count = 0  # used to calculated the running mean for
        # flops_per_sample
        self.total_flops = 0  # total computation flops to convergence until
        # current fl round
        self.total_upload_bytes = 0  # total upload space cost in bytes
        # until current fl round
        self.total_download_bytes = 0  # total download space cost in bytes
        # until current fl round
        self.fl_begin_wall_time = datetime.datetime.now()
        self.fl_end_wall_time = 0
        # for the metrics whose names includes "convergence", 0 indicates
        # the worker does not converge yet
        # Note:
        # 1) the convergence wall time is prone to fluctuations due to
        # possible resource competition during FL courses
        # 2) the global/local indicates whether the early stopping triggered
        # with global-aggregation/local-training
        self.global_convergence_round = 0  # total fl rounds to convergence
        self.global_convergence_wall_time = 0
        self.local_convergence_round = 0  # total fl rounds to convergence
        self.local_convergence_wall_time = 0

        if self.wandb_online_track:
            global_all_monitors.append(self)
        if self.use_wandb:
            try:
                import wandb
            except ImportError:
                logger.error(
                    "cfg.wandb.use=True but not install the wandb package")
                exit()

    def eval(self, ctx):
        results = self.metric_calculator.eval(ctx)
        return results

    def global_converged(self):
        self.global_convergence_wall_time = datetime.datetime.now(
        ) - self.fl_begin_wall_time
        self.global_convergence_round = self.monitored_object.state

    def local_converged(self):
        self.local_convergence_wall_time = datetime.datetime.now(
        ) - self.fl_begin_wall_time
        self.local_convergence_round = self.monitored_object.state

    def finish_fl(self):
        self.fl_end_wall_time = datetime.datetime.now(
        ) - self.fl_begin_wall_time

        system_metrics = self.get_sys_metrics()
        sys_metric_f_name = os.path.join(self.outdir, "system_metrics.log")
        with open(sys_metric_f_name, "a") as f:
            f.write(json.dumps(system_metrics) + "\n")

    def get_sys_metrics(self, verbose=True):
        system_metrics = {
            "id": self.monitored_object.ID,
            "fl_end_time_minutes": self.fl_end_wall_time.total_seconds() /
            60 if isinstance(self.fl_end_wall_time, datetime.timedelta) else 0,
            "total_model_size": self.total_model_size,
            "total_flops": self.total_flops,
            "total_upload_bytes": self.total_upload_bytes,
            "total_download_bytes": self.total_download_bytes,
            "global_convergence_round": self.global_convergence_round,
            "local_convergence_round": self.local_convergence_round,
            "global_convergence_time_minutes": self.
            global_convergence_wall_time.total_seconds() / 60 if isinstance(
                self.global_convergence_wall_time, datetime.timedelta) else 0,
            "local_convergence_time_minutes": self.local_convergence_wall_time.
            total_seconds() / 60 if isinstance(
                self.local_convergence_wall_time, datetime.timedelta) else 0,
        }
        if verbose:
            logger.info(
                f"In worker #{self.monitored_object.ID}, the system-related "
                f"metrics are: {str(system_metrics)}")
        return system_metrics

    def merge_system_metrics_simulation_mode(self,
                                             file_io=True,
                                             from_global_monitors=False):
        """
            average the system metrics recorded in "system_metrics.json" by
            all workers
        :return:
        """

        all_sys_metrics = defaultdict(list)
        avg_sys_metrics = defaultdict()
        std_sys_metrics = defaultdict()

        if file_io:
            sys_metric_f_name = os.path.join(self.outdir, "system_metrics.log")
            if not os.path.exists(sys_metric_f_name):
                logger.warning(
                    "You have not tracked the workers' system metrics in "
                    "$outdir$/system_metrics.log, "
                    "we will skip the merging. Plz check whether you do not "
                    "want to call monitor.finish_fl()")
                return
            with open(sys_metric_f_name, "r") as f:
                for line in f:
                    res = json.loads(line)
                    if all_sys_metrics is None:
                        all_sys_metrics = res
                        all_sys_metrics["id"] = "all"
                    else:
                        for k, v in res.items():
                            all_sys_metrics[k].append(v)
            id_to_be_merged = all_sys_metrics["id"]
            if len(id_to_be_merged) != len(set(id_to_be_merged)):
                logger.warning(
                    f"The sys_metric_file ({sys_metric_f_name}) contains "
                    f"duplicated tracked sys-results with these ids: "
                    f"f{id_to_be_merged} "
                    f"We will skip the merging as the merge is invalid. "
                    f"Plz check whether you specify the 'outdir' "
                    f"as the same as the one of another older experiment.")
                return
        elif from_global_monitors:
            for monitor in global_all_monitors:
                res = monitor.get_sys_metrics(verbose=False)
                if all_sys_metrics is None:
                    all_sys_metrics = res
                    all_sys_metrics["id"] = "all"
                else:
                    for k, v in res.items():
                        all_sys_metrics[k].append(v)
        else:
            raise ValueError("file_io or from_monitors should be True: "
                             f"but got file_io={file_io}, from_monitors"
                             f"={from_global_monitors}")

        for k, v in all_sys_metrics.items():
            if k == "id":
                avg_sys_metrics[k] = "sys_avg"
                std_sys_metrics[k] = "sys_std"
            else:
                v = np.array(v).astype("float")
                mean_res = np.mean(v)
                std_res = np.std(v)
                if "flops" in k or "bytes" in k or "size" in k:
                    mean_res = self.convert_size(mean_res)
                    std_res = self.convert_size(std_res)
                avg_sys_metrics[f"sys_avg/{k}"] = mean_res
                std_sys_metrics[f"sys_std/{k}"] = std_res

        logger.info(
            f"After merging the system metrics from all works, we got avg:"
            f" {avg_sys_metrics}")
        logger.info(
            f"After merging the system metrics from all works, we got std:"
            f" {std_sys_metrics}")
        if file_io:
            with open(sys_metric_f_name, "a") as f:
                f.write(json.dumps(avg_sys_metrics) + "\n")
                f.write(json.dumps(std_sys_metrics) + "\n")

        if self.use_wandb and self.wandb_online_track:
            try:
                import wandb
                # wandb.log(avg_sys_metrics)
                # wandb.log(std_sys_metrics)
                for k, v in avg_sys_metrics.items():
                    wandb.summary[k] = v
                for k, v in std_sys_metrics.items():
                    wandb.summary[k] = v
            except ImportError:
                logger.error(
                    "cfg.wandb.use=True but not install the wandb package")
                exit()

    def save_formatted_results(self,
                               formatted_res,
                               save_file_name="eval_results.log"):
        line = str(formatted_res) + "\n"
        if save_file_name != "":
            with open(os.path.join(self.outdir, save_file_name),
                      "a") as outfile:
                outfile.write(line)
        if self.use_wandb and self.wandb_online_track:
            try:
                import wandb
                exp_stop_normal = False
                exp_stop_normal, log_res = logline_2_wandb_dict(
                    exp_stop_normal, line, self.log_res_best, raw_out=False)
                wandb.log(log_res)
            except ImportError:
                logger.error(
                    "cfg.wandb.use=True but not install the wandb package")
                exit()

    def finish_fed_runner(self, fl_mode=None):
        self.compress_raw_res_file()
        if fl_mode == "standalone":
            self.merge_system_metrics_simulation_mode()

        if self.use_wandb and not self.wandb_online_track:
            try:
                import wandb
            except ImportError:
                logger.error(
                    "cfg.wandb.use=True but not install the wandb package")
                exit()

            from federatedscope.core.auxiliaries.logging import \
                logfile_2_wandb_dict
            with open(os.path.join(self.outdir, "eval_results.log"),
                      "r") as exp_log_f:
                # track the prediction related performance
                all_log_res, exp_stop_normal, last_line, log_res_best = \
                    logfile_2_wandb_dict(exp_log_f, raw_out=False)
                for log_res in all_log_res:
                    wandb.log(log_res)
                wandb.log(log_res_best)

                # track the system related performance
                sys_metric_f_name = os.path.join(self.outdir,
                                                 "system_metrics.log")
                with open(sys_metric_f_name, "r") as f:
                    for line in f:
                        res = json.loads(line)
                        if res["id"] in ["sys_avg", "sys_std"]:
                            # wandb.log(res)
                            for k, v in res.items():
                                wandb.summary[k] = v

    def compress_raw_res_file(self):
        old_f_name = os.path.join(self.outdir, "eval_results.raw")
        if os.path.exists(old_f_name):
            logger.info(
                "We will compress the file eval_results.raw into a .gz file, "
                "and delete the old one")
            with open(old_f_name, 'rb') as f_in:
                with gzip.open(old_f_name + ".gz", 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(old_f_name)

    def format_eval_res(self,
                        results,
                        rnd,
                        role=-1,
                        forms=None,
                        return_raw=False):
        """
        format the evaluation results from trainer.ctx.eval_results

        Args:
            results (dict): a dict to store the evaluation results {metric:
            value}
            rnd (int|string): FL round
            role (int|string): the output role
            forms (list): format type
            return_raw (bool): return either raw results, or other results

        Returns:
            round_formatted_results (dict): a formatted results with
            different forms and roles,
            e.g.,
            {
            'Role': 'Server #',
            'Round': 200,
            'Results_weighted_avg': {
                'test_avg_loss': 0.58, 'test_acc': 0.67, 'test_correct':
                3356, 'test_loss': 2892, 'test_total': 5000
                },
            'Results_avg': {
                'test_avg_loss': 0.57, 'test_acc': 0.67, 'test_correct':
                3356, 'test_loss': 2892, 'test_total': 5000
                },
            'Results_fairness': {
             'test_total': 33.99, 'test_correct': 27.185,
             'test_avg_loss_std': 0.433551,
             'test_avg_loss_bottom_decile': 0.356503,
             'test_avg_loss_top_decile': 1.212492,
             'test_avg_loss_min': 0.198317, 'test_avg_loss_max': 3.603567,
             'test_avg_loss_bottom10%': 0.276681, 'test_avg_loss_top10%':
             1.686649,
             'test_avg_loss_cos1': 0.867932, 'test_avg_loss_entropy': 5.164172,
             'test_loss_std': 13.686828, 'test_loss_bottom_decile': 11.822035,
             'test_loss_top_decile': 39.727236, 'test_loss_min': 7.337724,
             'test_loss_max': 100.899873, 'test_loss_bottom10%': 9.618685,
             'test_loss_top10%': 54.96769, 'test_loss_cos1': 0.880356,
             'test_loss_entropy': 5.175803, 'test_acc_std': 0.123823,
             'test_acc_bottom_decile': 0.676471, 'test_acc_top_decile':
             0.916667,
             'test_acc_min': 0.071429, 'test_acc_max': 0.972973,
             'test_acc_bottom10%': 0.527482, 'test_acc_top10%': 0.94486,
             'test_acc_cos1': 0.988134, 'test_acc_entropy': 5.283755
                },
            }
        """
        if forms is None:
            forms = ['weighted_avg', 'avg', 'fairness', 'raw']
        round_formatted_results = {'Role': role, 'Round': rnd}
        round_formatted_results_raw = {'Role': role, 'Round': rnd}
        for form in forms:
            new_results = copy.deepcopy(results)
            if not role.lower().startswith('server') or form == 'raw':
                round_formatted_results_raw['Results_raw'] = new_results
            elif form not in Monitor.SUPPORTED_FORMS:
                continue
            else:
                for key in results.keys():
                    dataset_name = key.split("_")[0]
                    if f'{dataset_name}_total' not in results:
                        raise ValueError(
                            "Results to be formatted should be include the "
                            "dataset_num in the dict,"
                            f"with key = {dataset_name}_total")
                    else:
                        dataset_num = np.array(
                            results[f'{dataset_name}_total'])
                        if key in [
                                f'{dataset_name}_total',
                                f'{dataset_name}_correct'
                        ]:
                            new_results[key] = np.mean(new_results[key])

                    if key in [
                            f'{dataset_name}_total', f'{dataset_name}_correct'
                    ]:
                        new_results[key] = np.mean(new_results[key])
                    else:
                        all_res = np.array(copy.copy(results[key]))
                        if form == 'weighted_avg':
                            new_results[key] = np.sum(
                                np.array(new_results[key]) *
                                dataset_num) / np.sum(dataset_num)
                        if form == "avg":
                            new_results[key] = np.mean(new_results[key])
                        if form == "fairness" and all_res.size > 1:
                            # by default, log the std and decile
                            new_results.pop(
                                key, None)  # delete the redundant original one
                            all_res.sort()
                            new_results[f"{key}_std"] = np.std(
                                np.array(all_res))
                            new_results[f"{key}_bottom_decile"] = all_res[
                                all_res.size // 10]
                            new_results[f"{key}_top_decile"] = all_res[
                                all_res.size * 9 // 10]
                            # log more fairness metrics
                            # min and max
                            new_results[f"{key}_min"] = all_res[0]
                            new_results[f"{key}_max"] = all_res[-1]
                            # bottom and top 10%
                            new_results[f"{key}_bottom10%"] = np.mean(
                                all_res[:all_res.size // 10])
                            new_results[f"{key}_top10%"] = np.mean(
                                all_res[all_res.size * 9 // 10:])
                            # cosine similarity between the performance
                            # distribution and 1
                            new_results[f"{key}_cos1"] = np.mean(all_res) / (
                                np.sqrt(np.mean(all_res**2)))
                            # entropy of performance distribution
                            all_res_preprocessed = all_res + 1e-9
                            new_results[f"{key}_entropy"] = np.sum(
                                -all_res_preprocessed /
                                np.sum(all_res_preprocessed) * (np.log(
                                    (all_res_preprocessed) /
                                    np.sum(all_res_preprocessed))))
                round_formatted_results[f'Results_{form}'] = new_results

        with open(os.path.join(self.outdir, "eval_results.raw"),
                  "a") as outfile:
            outfile.write(str(round_formatted_results_raw) + "\n")

        return round_formatted_results_raw if return_raw else \
            round_formatted_results

    def calc_blocal_dissim(self, last_model, local_updated_models):
        '''
        Arguments:
            last_model (dict): the state of last round.
            local_updated_models (list): each element is ooxx.
        Returns:
            b_local_dissimilarity (dict): the measurements proposed in
            "Tian Li, Anit Kumar Sahu, Manzil Zaheer, and et al. Federated
            Optimization in Heterogeneous Networks".
        '''
        # for k, v in last_model.items():
        #    print(k, v)
        # for i, elem in enumerate(local_updated_models):
        #    print(i, elem)
        local_grads = []
        weights = []
        local_gnorms = []
        for tp in local_updated_models:
            weights.append(tp[0])
            grads = dict()
            gnorms = dict()
            for k, v in tp[1].items():
                grad = v - last_model[k]
                grads[k] = grad
                gnorms[k] = torch.sum(grad**2)
            local_grads.append(grads)
            local_gnorms.append(gnorms)
        weights = np.asarray(weights)
        weights = weights / np.sum(weights)
        avg_gnorms = dict()
        global_grads = dict()
        for i in range(len(local_updated_models)):
            gnorms = local_gnorms[i]
            for k, v in gnorms.items():
                if k not in avg_gnorms:
                    avg_gnorms[k] = .0
                avg_gnorms[k] += weights[i] * v
            grads = local_grads[i]
            for k, v in grads.items():
                if k not in global_grads:
                    global_grads[k] = torch.zeros_like(v)
                global_grads[k] += weights[i] * v
        b_local_dissimilarity = dict()
        for k in avg_gnorms:
            b_local_dissimilarity[k] = np.sqrt(
                avg_gnorms[k].item() / torch.sum(global_grads[k]**2).item())
        return b_local_dissimilarity

    def convert_size(self, size_bytes):
        import math
        if size_bytes <= 0:
            return str(size_bytes)
        size_name = ("", "K", "M", "G", "T", "P", "E", "Z", "Y")
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s}{size_name[i]}"

    def track_model_size(self, models):
        """
            calculate the total model size given the models hold by the
            worker/trainer

        :param models: torch.nn.Module or list of torch.nn.Module
        :return:
        """
        if self.total_model_size != 0:
            logger.warning(
                "the total_model_size is not zero. You may have been "
                "calculated the total_model_size before")

        if not hasattr(models, '__iter__'):
            models = [models]
        for model in models:
            assert isinstance(model, torch.nn.Module), \
                f"the `model` should be type torch.nn.Module when " \
                f"calculating its size, but got {type(model)}"
            for name, para in model.named_parameters():
                self.total_model_size += para.numel()

    def track_avg_flops(self, flops, sample_num=1):
        """
            update the average flops for forwarding each data sample,
            for most models and tasks,
            the averaging is not needed as the input shape is fixed

        :param flops: flops/
        :param sample_num:
        :return:
        """

        self.flops_per_sample = (self.flops_per_sample * self.flop_count +
                                 flops) / (self.flop_count + sample_num)
        self.flop_count += 1

    def track_upload_bytes(self, bytes):
        self.total_upload_bytes += bytes

    def track_download_bytes(self, bytes):
        self.total_download_bytes += bytes

    def update_best_result(self, best_results, new_results, results_type):
        """
            update best evaluation results.
            by default, the update is based on validation loss with
            `round_wise_update_key="val_loss" `
        """
        update_best_this_round = False
        if not isinstance(new_results, dict):
            raise ValueError(
                f"update best results require `results` a dict, but got"
                f" {type(new_results)}")
        else:
            if results_type not in best_results:
                best_results[results_type] = dict()
            best_result = best_results[results_type]
            # update different keys separately: the best values can be in
            # different rounds
            if self.round_wise_update_key is None:
                for key in new_results:
                    cur_result = new_results[key]
                    if 'loss' in key or 'std' in key:  # the smaller,
                        # the better
                        if results_type in [
                                "client_best_individual",
                                "unseen_client_best_individual"
                        ]:
                            cur_result = min(cur_result)
                        if key not in best_result or cur_result < best_result[
                                key]:
                            best_result[key] = cur_result
                            update_best_this_round = True

                    elif 'acc' in key:  # the larger, the better
                        if results_type in [
                                "client_best_individual",
                                "unseen_client_best_individual"
                        ]:
                            cur_result = max(cur_result)
                        if key not in best_result or cur_result > best_result[
                                key]:
                            best_result[key] = cur_result
                            update_best_this_round = True
                    else:
                        # unconcerned metric
                        pass
            # update different keys round-wise: if find better
            # round_wise_update_key, update others at the same time
            else:
                found_round_wise_update_key = False
                sorted_keys = []
                for key in new_results:
                    if self.round_wise_update_key in key:
                        sorted_keys.insert(0, key)
                        found_round_wise_update_key = True
                    else:
                        sorted_keys.append(key)
                if not found_round_wise_update_key:
                    raise ValueError(
                        "Your specified eval.best_res_update_round_wise_key "
                        "is not in target results, "
                        "use another key or check the name. \n"
                        f"Got eval.best_res_update_round_wise_key"
                        f"={self.round_wise_update_key}, "
                        f"the keys of results are {list(new_results.keys())}")

                for key in sorted_keys:
                    cur_result = new_results[key]
                    if update_best_this_round or (
                            not self.the_larger_the_better):
                        # The smaller the better
                        if results_type in [
                                "client_best_individual",
                                "unseen_client_best_individual"
                        ]:
                            cur_result = min(cur_result)
                        if update_best_this_round or \
                                key not in best_result or cur_result < \
                                best_result[key]:
                            best_result[key] = cur_result
                            update_best_this_round = True
                    elif update_best_this_round or self.the_larger_the_better:
                        # The larger the better
                        if results_type in [
                                "client_best_individual",
                                "unseen_client_best_individual"
                        ]:
                            cur_result = max(cur_result)
                        if update_best_this_round or \
                                key not in best_result or cur_result > \
                                best_result[key]:
                            best_result[key] = cur_result
                            update_best_this_round = True
                    else:
                        # unconcerned metric
                        pass

        if update_best_this_round:
            line = f"Find new best result: {best_results}"
            logging.info(line)
            if self.use_wandb and self.wandb_online_track:
                try:
                    import wandb
                    exp_stop_normal = False
                    exp_stop_normal, log_res = logline_2_wandb_dict(
                        exp_stop_normal,
                        line,
                        self.log_res_best,
                        raw_out=False)
                    # wandb.log(self.log_res_best)
                    for k, v in self.log_res_best.items():
                        wandb.summary[k] = v
                except ImportError:
                    logger.error(
                        "cfg.wandb.use=True but not install the wandb package")
                    exit()
