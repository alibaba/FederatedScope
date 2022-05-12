import copy
import logging
import os
import gzip
import shutil

import numpy as np

try:
    import torch
except ImportError:
    torch = None

logger = logging.getLogger(__name__)


class Monitor(object):
    """
        provide the monitoring functionalities such as formatting the evaluation results into diverse metrics
    """
    SUPPORTED_FORMS = ['weighted_avg', 'avg', 'fairness', 'raw']

    def __init__(
        self,
        cfg,
    ):
        self.outdir = cfg.outdir
        self.use_wandb = cfg.wandb.use
        # self.use_tensorboard = cfg.use_tensorboard

    def compress_raw_res_file(self):
        old_f_name = os.path.join(self.outdir, "eval_results.raw")
        if os.path.exists(old_f_name):
            logger.info(
                "We will compress the file eval_results.raw into a .gz file, and delete the old one"
            )
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
            results (dict): a dict to store the evaluation results {metric: value}
            rnd (int|string): FL round
            role (int|string): the output role
            forms (list): format type
            return_raw (bool): return either raw results, or other results

        Returns:
            round_formatted_results (dict): a formatted results with different forms and roles,
            e.g.,
            {
            'Role': 'Server #',
            'Round': 200,
            'Results_weighted_avg': {
                'test_avg_loss': 0.58, 'test_acc': 0.67, 'test_correct': 3356, 'test_loss': 2892, 'test_total': 5000
                },
            'Results_avg': {
                'test_avg_loss': 0.57, 'test_acc': 0.67, 'test_correct': 3356, 'test_loss': 2892, 'test_total': 5000
                },
            'Results_fairness': {
                'test_correct': 3356,      'test_total': 5000,
                'test_avg_loss_std': 0.04, 'test_avg_loss_bottom_decile': 0.52, 'test_avg_loss_top_decile': 0.64,
                'test_acc_std': 0.06,      'test_acc_bottom_decile': 0.60,      'test_acc_top_decile': 0.75,
                'test_loss_std': 214.17,   'test_loss_bottom_decile': 2644.64,  'test_loss_top_decile': 3241.23
                },
            }
        """
        # TODO: better visualization via wandb or tensorboard
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
                            "Results to be formatted should be include the dataset_num in the dict,"
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
                round_formatted_results[f'Results_{form}'] = new_results

        with open(os.path.join(self.outdir, "eval_results.raw"),
                  "a") as outfile:
            outfile.write(str(round_formatted_results_raw) + "\n")

        return round_formatted_results_raw if return_raw else round_formatted_results

    def calc_blocal_dissim(last_model, local_updated_models):
        '''
        Arguments:
            last_model (dict): the state of last round.
            local_updated_models (list): each element is ooxx.
        Returns:
            b_local_dissimilarity (dict): the measurements proposed in
            "Tian Li, Anit Kumar Sahu, Manzil Zaheer, and et al. Federated Optimization in Heterogeneous Networks".
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


def update_best_result(best_results,
                       new_results,
                       results_type,
                       round_wise_update_key="val_loss"):
    """
        update best evaluation results.
        by default, the update is based on validation loss with `round_wise_update_key="val_loss" `
    """
    update_best_this_round = False
    if not isinstance(new_results, dict):
        raise ValueError(
            f"update best results require `results` a dict, but got {type(new_results)}"
        )
    else:
        if results_type not in best_results:
            best_results[results_type] = dict()
        best_result = best_results[results_type]
        # update different keys separately: the best values can be in different rounds
        if round_wise_update_key is None:
            for key in new_results:
                cur_result = new_results[key]
                if 'loss' in key or 'std' in key:  # the smaller, the better
                    if results_type == "client_individual":
                        cur_result = min(cur_result)
                    if key not in best_result or cur_result < best_result[
                            key]:
                        best_result[key] = cur_result
                        update_best_this_round = True

                elif 'acc' in key:  # the larger, the better
                    if results_type == "client_individual":
                        cur_result = max(cur_result)
                    if key not in best_result or cur_result > best_result[
                            key]:
                        best_result[key] = cur_result
                        update_best_this_round = True
                else:
                    # unconcerned metric
                    pass
        # update different keys round-wise: if find better round_wise_update_key, update others at the same time
        else:
            if round_wise_update_key not in [
                    "val_loss", "val_acc", "val_std", "test_loss",
                    "test_acc", "test_std", "test_avg_loss", "loss"
            ]:
                raise NotImplementedError(
                    f"We currently support round_wise_update_key as one of "
                    f"['val_loss', 'val_acc', 'val_std', 'test_loss', 'test_acc', 'test_std'] "
                    f"for round-wise best results update, but got {round_wise_update_key}."
                )

            found_round_wise_update_key = False
            sorted_keys = []
            for key in new_results:
                if round_wise_update_key in key:
                    sorted_keys.insert(0, key)
                    found_round_wise_update_key = True
                else:
                    sorted_keys.append(key)
            if not found_round_wise_update_key:
                raise ValueError(
                    "Your specified eval.best_res_update_round_wise_key is not in target results, "
                    "use another key or check the name. \n"
                    f"Got eval.best_res_update_round_wise_key={round_wise_update_key}, "
                    f"the keys of results are {list(new_results.keys())}")

            for key in sorted_keys:
                cur_result = new_results[key]
                if update_best_this_round or \
                        ('loss' in round_wise_update_key and 'loss' in key) or \
                        ('std' in round_wise_update_key and 'std' in key):
                    # The smaller the better
                    if results_type == "client_individual":
                        cur_result = min(cur_result)
                    if update_best_this_round or \
                            key not in best_result or cur_result < best_result[key]:
                        best_result[key] = cur_result
                        update_best_this_round = True
                elif update_best_this_round or \
                        'acc' in round_wise_update_key and 'acc' in key:
                    # The larger the better
                    if results_type == "client_individual":
                        cur_result = max(cur_result)
                    if update_best_this_round or \
                            key not in best_result or cur_result > best_result[key]:
                        best_result[key] = cur_result
                        update_best_this_round = True
                else:
                    # unconcerned metric
                    pass

    if update_best_this_round:
        logging.info(f"Find new best result: {best_results}")