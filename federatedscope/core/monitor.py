import copy
import os

import numpy as np

try:
    import torch
except ImportError:
    torch = None


class Monitor(object):
    SUPPORTED_FORMS = ['weighted_avg', 'avg', 'fairness', 'raw']

    def __init__(self,
                 cfg,
                 ):
        self.outdir = cfg.outdir
        self.use_wandb = cfg.use_wandb
        # self.use_tensorboard = cfg.use_tensorboard

    def __del__(self):
        self.compress_raw_res_file()

    def compress_raw_res_file(self):
        import gzip
        import shutil
        old_f_name = os.path.join(self.outdir, "eval_results.raw")
        if os.path.exists(old_f_name):
            with open(old_f_name, 'rb') as f_in:
                with gzip.open(old_f_name + ".gz", 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(old_f_name)

    def format_eval_res(self, results, rnd, role=-1, forms=None):
        """
        format the evaluation results from trainer.ctx.eval_results

        Args:
            results (dict): a dict to store the evaluation results {metric: value}
            rnd (int|string): FL round
            role (int|string): the output role
            forms (list): format type

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
                        dataset_num = np.array(results[f'{dataset_name}_total'])
                        if key in [
                            f'{dataset_name}_total', f'{dataset_name}_correct'
                        ]:
                            new_results[key] = np.mean(new_results[key])

                    if key in [f'{dataset_name}_total', f'{dataset_name}_correct']:
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
                            new_results[f"{key}_std"] = np.std(np.array(all_res))
                            new_results[f"{key}_bottom_decile"] = all_res[
                                all_res.size // 10]
                            new_results[f"{key}_top_decile"] = all_res[all_res.size
                                                                       * 9 // 10]
                round_formatted_results[f'Results_{form}'] = new_results

        with open(os.path.join(self.outdir, "eval_results.raw"),
                  "a") as outfile:
            outfile.write(str(round_formatted_results_raw) + "\n")

        return round_formatted_results

    def calc_blocal_dissim(last_model, local_updated_models):
        '''
        Arguments:
            last_model (dict): the state of last round.
            local_updated_models (list): each element is ooxx.
        Returns:
            b_local_dissimilarity (dict): the measurements.
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
                gnorms[k] = torch.sum(grad ** 2)
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
                avg_gnorms[k].item() / torch.sum(global_grads[k] ** 2).item())
        return b_local_dissimilarity
