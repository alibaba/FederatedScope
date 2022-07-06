import copy
import json
import os

import numpy as np
import wandb
from collections import OrderedDict

import yaml

api = wandb.Api()

name_project = "daoyuan/pFL-bench"

filters_each_line_main_table = OrderedDict(
    # {dataset_name: filter}
    [
        # ("all",
        # None,
        # ),
        # ("FEMNIST-all",
        #  {"$and":
        #      [
        #          {"config.data.type": "femnist"},
        #      ]
        #  }
        #  ),
        ("FEMNIST-s02", {
            "$and": [
                {
                    "config.data.type": "femnist"
                },
                {
                    "config.federate.sample_client_rate": 0.2
                },
                {
                    "state": "finished"
                },
            ]
        }),
        # ("cifar10-alpha05",
        #  {"$and":
        #      [
        #          {"config.data.type": "CIFAR10@torchvision"},
        #          {"config.data.splitter_args": [{"alpha": 0.5}]},
        #      ]
        #  }
        #  ),
        ("sst2", {
            "$and": [
                {
                    "config.data.type": "sst2@huggingface_datasets"
                },
            ]
        }),
        ("pubmed", {
            "$and": [
                {
                    "config.data.type": "pubmed"
                },
            ]
        }),
    ])

filters_each_line_all_cifar10 = OrderedDict(
    # {dataset_name: filter}
    [
        ("cifar10-alpha5", {
            "$and": [
                {
                    "config.data.type": "CIFAR10@torchvision"
                },
                {
                    "config.data.splitter_args": [{
                        "alpha": 5
                    }]
                },
            ]
        }),
        ("cifar10-alpha05", {
            "$and": [
                {
                    "config.data.type": "CIFAR10@torchvision"
                },
                {
                    "config.data.splitter_args": [{
                        "alpha": 0.5
                    }]
                },
            ]
        }),
        ("cifar10-alpha01", {
            "$and": [
                {
                    "config.data.type": "CIFAR10@torchvision"
                },
                {
                    "config.data.splitter_args": [{
                        "alpha": 0.1
                    }]
                },
            ]
        }),
    ])

filters_each_line_femnist_all_s = OrderedDict(
    # {dataset_name: filter}
    [
        ("FEMNIST-s02", {
            "$and": [
                {
                    "config.data.type": "femnist"
                },
                {
                    "config.federate.sample_client_rate": 0.2
                },
                {
                    "state": "finished"
                },
            ]
        }),
        ("FEMNIST-s01", {
            "$and": [
                {
                    "config.data.type": "femnist"
                },
                {
                    "config.federate.sample_client_rate": 0.1
                },
                {
                    "state": "finished"
                },
            ]
        }),
        ("FEMNIST-s005", {
            "$and": [
                {
                    "config.data.type": "femnist"
                },
                {
                    "config.federate.sample_client_rate": 0.05
                },
                {
                    "state": "finished"
                },
            ]
        }),
    ])

filters_each_line_all_graph = OrderedDict(
    # {dataset_name: filter}
    [
        ("pubmed", {
            "$and": [
                {
                    "config.data.type": "pubmed"
                },
            ]
        }),
        ("cora", {
            "$and": [
                {
                    "config.data.type": "cora"
                },
            ]
        }),
        ("citeseer", {
            "$and": [
                {
                    "config.data.type": "citeseer"
                },
            ]
        }),
    ])

filters_each_line_all_nlp = OrderedDict(
    # {dataset_name: filter}
    [
        ("cola", {
            "$and": [
                {
                    "config.data.type": "cola@huggingface_datasets"
                },
            ]
        }),
        ("sst2", {
            "$and": [
                {
                    "config.data.type": "sst2@huggingface_datasets"
                },
            ]
        }),
    ])

sweep_name_2_id = dict()
column_names_generalization = [
    "best_client_summarized_weighted_avg/test_acc",
    "best_unseen_client_summarized_weighted_avg_unseen/test_acc",
    "participation_gap"
]
column_names_fair = [
    "best_client_summarized_avg/test_acc",
    "best_client_summarized_fairness/test_acc_std",
    "best_client_summarized_fairness/test_acc_bottom_decile"
]
column_names_efficiency = [
    "sys_avg/total_flops",
    "sys_avg/total_upload_bytes",
    "sys_avg/total_download_bytes",
    "sys_avg/global_convergence_round",
    # "sys_avg/local_convergence_round"
]
column_names_generalization_for_plot = [
    "Acc (Parti.)", "Acc (Un-parti.)", "Generalization Gap"
]
column_name_for_plot = {
    "best_client_summarized_weighted_avg/test_acc": "Acc (Parti.)",
    "total_flops": "Total Flops",
    "communication_bytes": "Communication Bytes",
    "sys_avg/global_convergence_round": "Convergence Round",
}
sorted_method_name_pair = [
    ("global-train", "Global-Train"),
    ("isolated-train", "Isolated"),
    ("fedavg", "FedAvg"),
    ("fedavg-ft", "FedAvg-FT"),
    ("fedopt", "FedOpt"),
    ("fedopt-ft", "FedOpt-FT"),
    ("pfedme", "pFedMe"),
    ("ft-pfedme", "pFedMe-FT"),
    ("fedbn", "FedBN"),
    ("fedbn-ft", "FedBN-FT"),
    ("fedbn-fedopt", "FedBN-FedOPT"),
    ("fedbn-fedopt-ft", "FedBN-FedOPT-FT"),
    ("ditto", "Ditto"),
    ("ditto-ft", "Ditto-FT"),
    ("ditto-fedbn", "Ditto-FedBN"),
    ("ditto-fedbn-ft", "Ditto-FedBN-FT"),
    ("ditto-fedbn-fedopt", "Ditto-FedBN-FedOpt"),
    ("ditto-fedbn-fedopt-ft", "Ditto-FedBN-FedOpt-FT"),
    ("fedem", "FedEM"),
    ("fedem-ft", "FedEM-FT"),
    ("fedbn-fedem", "FedEM-FedBN"),
    ("fedbn-fedem-ft", "FedEM-FedBN-FT"),
    ("fedbn-fedem-fedopt", "FedEM-FedBN-FedOPT"),
    ("fedbn-fedem-fedopt-ft", "FedEM-FedBN-FedOPT-FT"),
]
sorted_keys = OrderedDict(sorted_method_name_pair)
expected_keys = set(list(sorted_keys.keys()))
expected_method_names = list(sorted_keys.values())
expected_datasets_name = [
    "cola", "sst2", "pubmed", "cora", "citeseer", "cifar10-alpha5",
    "cifar10-alpha05", "cifar10-alpha01", "FEMNIST-s02", "FEMNIST-s01",
    "FEMNIST-s005"
]
expected_seed_set = ["1", "2", "3"]
expected_expname_tag = set()

original_method_names = [
    "Global-Train", "Isolated", "FedAvg", "pFedMe", "FedBN", "Ditto", "FedEM"
]

for method_name in expected_method_names:
    for dataset_name in expected_datasets_name:
        for seed in expected_seed_set:
            expected_expname_tag.add(
                f"{method_name}_{dataset_name}_seed{seed}")
        expected_expname_tag.add(f"{method_name}_{dataset_name}_repeat")

from collections import defaultdict

all_missing_scripts = defaultdict(list)

all_res_structed = defaultdict(dict)
for expname_tag in expected_expname_tag:
    for metric in column_names_generalization + column_names_efficiency + column_names_fair:
        if "repeat" in expname_tag:
            all_res_structed[expname_tag][metric] = []
        else:
            all_res_structed[expname_tag][metric] = "-"


def load_best_repeat_res(filter_seed_set=None):
    for expname_tag in expected_expname_tag:
        filter = {
            "$and": [
                {
                    "config.expname_tag": expname_tag
                },
            ]
        }
        filtered_runs = api.runs("pfl-bench-best-repeat", filters=filter)
        method, dataname, seed = expname_tag.split("_")
        finished_run_cnt = 0
        for run in filtered_runs:
            if run.state != "finished":
                print(f"run {run} is not fished")
            else:
                finished_run_cnt += 1
                for metric in column_names_generalization + column_names_efficiency + column_names_fair:
                    try:
                        if method in ["Isolated", "Global-Train"]:
                            skip_generalize = "unseen" in metric or metric == "participation_gap"
                            skip_global_fairness = method == "Global-Train" and "fairness" in metric
                            if skip_generalize or skip_global_fairness:
                                all_res_structed[expname_tag][metric] = "-"
                                continue

                        if metric == "participation_gap":
                            all_res_structed[expname_tag][metric] = all_res_structed[expname_tag][
                                                                        "best_unseen_client_summarized_weighted_avg_unseen/test_acc"] - \
                                                                    all_res_structed[expname_tag][
                                                                        "best_client_summarized_weighted_avg/test_acc"]
                        else:
                            all_res_structed[expname_tag][
                                metric] = run.summary[metric]
                    except KeyError:
                        print("Something wrong")

        print_missing = True
        for seed in filter_seed_set:
            if seed in expname_tag:
                print_missing = False
        if finished_run_cnt == 0 and print_missing:
            print(f"Missing run {expname_tag})")
            yaml_name = f"{method}_{dataname}.yaml"
            if "Global" in method:
                yaml_name = f"\'{yaml_name}\'"
                expname_tag_new = expname_tag.replace("Global Train",
                                                      "Global-Train")
            else:
                expname_tag_new = expname_tag
            seed_num = seed.replace("seed", "")
            all_missing_scripts[seed].append(
                f"python federatedscope/main.py --cfg scripts/personalization_exp_scripts/pfl_bench/yaml_best_runs/{yaml_name} seed {seed_num} expname_tag {expname_tag_new} wandb.name_project pfl-bench-best-repeat"
            )
        elif finished_run_cnt != 1 and print_missing:
            print(f"run_cnt = {finished_run_cnt} for the exp {expname_tag}")

    for seed in all_missing_scripts.keys():
        print(
            f"+================= All MISSING SCRIPTS, seed={seed} =====================+, cnt={len(all_missing_scripts[seed])}"
        )
        for scipt in all_missing_scripts[seed]:
            print(scipt)
        print()


def bytes_to_unit_size(size_bytes):
    import math
    if size_bytes == 0:
        return "0"
    size_name = ("", "K", "M", "G", "T", "P", "E", "Z", "Y")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s}{size_name[i]}"


def unit_size_to_bytes(size_str):
    if not isinstance(size_str, str):
        return size_str
    else:
        try:
            last_unit = size_str[-1]
            size_name = ("", "K", "M", "G", "T", "P", "E", "Z", "Y")
            if last_unit not in size_name:
                return float(size_str)
            else:
                # need transform
                import math
                idx = size_name.index(last_unit)
                p = math.pow(1024, idx)
                return float(size_str[:-1]) * p
        except:
            return size_str


def avg_res_of_seeds():
    # add all res to repeat
    for expname_tag in expected_expname_tag:
        if "repeat" in expname_tag:
            continue
        else:
            for metric in column_names_generalization + column_names_efficiency + column_names_fair:
                if all_res_structed[expname_tag][
                        metric] == "-" and "Global" not in expname_tag and "Isolated" not in expname_tag:
                    print(f"missing {expname_tag} for metric {metric}")
                method, dataname, seed = expname_tag.split("_")
                cur_res = all_res_structed[expname_tag][metric]
                all_res_structed[f"{method}_{dataname}_repeat"][metric].append(
                    cur_res)

    for expname_tag in expected_expname_tag:
        if "repeat" in expname_tag:
            for metric in column_names_generalization + column_names_efficiency + column_names_fair:
                valid_res = [
                    unit_size_to_bytes(v)
                    for v in all_res_structed[expname_tag][metric] if v != "-"
                ]
                if len(valid_res) == 0:
                    all_res_structed[expname_tag][metric] = "-"
                else:
                    res = sum(valid_res) / len(valid_res)
                    if "flops" in metric or "bytes" in metric:
                        res = bytes_to_unit_size(res)
                    all_res_structed[expname_tag][metric] = res


def highlight_tex_res_in_table(res_to_print_matrix_raw,
                               rank_order,
                               need_scale=False,
                               filter_out=None,
                               convergence_case=False):
    res_to_print_matrix = []
    if filter_out is not None:
        # filter out the Global-Train and Isolated
        for line in res_to_print_matrix_raw:
            if line[0] in filter_out:
                continue
            else:
                res_to_print_matrix.append(line)
    else:
        res_to_print_matrix = res_to_print_matrix_raw

    res_np = np.array(res_to_print_matrix)
    row_len, col_len = res_np.shape

    if need_scale:
        vfun = np.vectorize(unit_size_to_bytes)
        res_np = vfun(res_np)

    # select method idx
    method_heads_all = res_np[:, :1]
    selected_method_idx = [
        i for i in range(row_len)
        if method_heads_all[i] in original_method_names
    ]

    raw_i_to_selected_i = {}
    for idx_i, selected_i in enumerate(selected_method_idx):
        raw_i_to_selected_i[selected_i] = idx_i

    # render column by column
    for col_i, col in enumerate(res_np[:, 1:].T):
        # first replace the missing results into numerical res
        if rank_order[col_i] == "+":
            # order == "+" indicates the larger, the better
            col = np.where(col == "-", -999999999, col)
        else:
            col = np.where(col == "-", 9999999999, col)
        if convergence_case:
            col = np.where(col == "0", 9999999999, col)
        col = col.astype("float")
        if rank_order[col_i] == "+":
            col = -col
        col_all = pd.DataFrame(col)
        ind_all_method = col_all.rank(
            method='dense').astype(int)[0].values.tolist()
        col_filter = pd.DataFrame(col[selected_method_idx])
        ind_partial_method_tmp = col_filter.rank(
            method='dense').astype(int)[0].values.tolist()
        for raw_i in range(row_len):
            if ind_all_method[raw_i] == 1:
                res_to_print_matrix[raw_i][
                    col_i +
                    1] = "\\textbf{" + res_to_print_matrix[raw_i][col_i +
                                                                  1] + "}"
            if ind_all_method[raw_i] == 2:
                res_to_print_matrix[raw_i][
                    col_i +
                    1] = "\\underline{" + res_to_print_matrix[raw_i][col_i +
                                                                     1] + "}"
            if raw_i in selected_method_idx and ind_partial_method_tmp[
                    raw_i_to_selected_i[raw_i]] == 1:
                res_to_print_matrix[raw_i][
                    col_i +
                    1] = "\\color{red}{" + res_to_print_matrix[raw_i][col_i +
                                                                      1] + "}"
            if raw_i in selected_method_idx and ind_partial_method_tmp[
                    raw_i_to_selected_i[raw_i]] == 2:
                res_to_print_matrix[raw_i][
                    col_i +
                    1] = "\\color{blue}{" + res_to_print_matrix[raw_i][col_i +
                                                                       1] + "}"

    return res_to_print_matrix


def print_paper_table_from_repeat(filters_each_line_table):
    res_of_each_line_generalization = OrderedDict()
    res_of_each_line_fair = OrderedDict()
    res_of_each_line_efficiency = OrderedDict()
    res_of_each_line_commu_acc_trade = OrderedDict()
    res_of_each_line_conver_acc_trade = OrderedDict()

    for key in expected_method_names:
        res_of_each_line_generalization[key] = []
        res_of_each_line_fair[key] = []
        res_of_each_line_efficiency[key] = []
        for dataset_name in filters_each_line_table:
            expname_tag = f"{key}_{dataset_name}_repeat"
            for metric in column_names_generalization:
                res_of_each_line_generalization[key].append(
                    all_res_structed[expname_tag][metric])
            for metric in column_names_fair:
                res_of_each_line_fair[key].append(
                    all_res_structed[expname_tag][metric])
            for metric in column_names_efficiency:
                res = all_res_structed[expname_tag][metric]
                if "round" in metric:
                    res = "{:.2f}".format(res)
                res_of_each_line_efficiency[key].append(res)

    print("\n=============res_of_each_line [Generalization]===============" +
          ",".join(list(filters_each_line_table.keys())))
    # Acc, Unseen-ACC, Delta
    res_to_print_matrix = []
    for key in expected_method_names:
        res_to_print = [
            "{:.2f}".format(v * 100) if v != "-" else v
            for v in res_of_each_line_generalization[key]
        ]
        res_to_print = [key] + res_to_print
        res_to_print_matrix.append(res_to_print)
        # print("&".join(res_to_print) + "\\\\")

    colum_order_per_data = ["+", "+", "+"]
    # "+" indicates the larger, the better
    rank_order = colum_order_per_data * len(filters_each_line_table)
    res_to_print_matrix = highlight_tex_res_in_table(res_to_print_matrix,
                                                     rank_order=rank_order)
    for res_to_print in res_to_print_matrix:
        print("&".join(res_to_print) + "\\\\")

    print("\n=============res_of_each_line [Fairness]===============" +
          ",".join(list(filters_each_line_table.keys())))
    res_to_print_matrix = []
    for key in expected_method_names:
        res_to_print = [
            "{:.2f}".format(v * 100) if v != "-" else v
            for v in res_of_each_line_fair[key]
        ]
        res_to_print = [key] + res_to_print
        res_to_print_matrix.append(res_to_print)
        # print("&".join(res_to_print) + "\\\\")

    colum_order_per_data = ["+", "-", "+"]
    # "+" indicates the larger, the better
    rank_order = colum_order_per_data * len(filters_each_line_table)
    res_to_print_matrix = highlight_tex_res_in_table(
        res_to_print_matrix,
        rank_order=rank_order,
        filter_out=["Global-Train"])
    for res_to_print in res_to_print_matrix:
        print("&".join(res_to_print) + "\\\\")

    # print("\n=============res_of_each_line [All Efficiency]===============" + ",".join(
    #    list(filters_each_line_table.keys())))
    ## FLOPS, UPLOAD, DOWNLOAD
    # for key in expected_method_names:
    #    res_to_print = [str(v) for v in res_of_each_line_efficiency[key]]
    #    res_to_print = [key] + res_to_print
    #    print("&".join(res_to_print) + "\\\\")

    print(
        "\n=============res_of_each_line [flops, communication, acc]==============="
        + ",".join(list(filters_each_line_table.keys())))
    res_to_print_matrix = []
    for key in expected_method_names:
        res_of_each_line_commu_acc_trade[key] = []
        dataset_num = 2 if "cola" in list(
            filters_each_line_table.keys()) else 3
        for i in range(dataset_num):
            res_of_each_line_commu_acc_trade[key].extend(
                [str(res_of_each_line_efficiency[key][i * 4])] + \
                [str(res_of_each_line_efficiency[key][i * 4 + 1])] + \
                ["{:.2f}".format(v * 100) if v != "-" else v for v in
                 res_of_each_line_generalization[key][i * 3:i * 3 + 1]]
            )

        res_to_print = [str(v) for v in res_of_each_line_commu_acc_trade[key]]
        res_to_print = [key] + res_to_print
        res_to_print_matrix.append(res_to_print)
        # print("&".join(res_to_print)+ "\\\\")

    colum_order_per_data = ["-", "-", "+"]
    # "+" indicates the larger, the better
    rank_order = colum_order_per_data * len(filters_each_line_table)
    res_to_print_matrix = highlight_tex_res_in_table(
        res_to_print_matrix,
        rank_order=rank_order,
        need_scale=True,
        filter_out=["Global-Train", "Isolated"])
    for res_to_print in res_to_print_matrix:
        print("&".join(res_to_print) + "\\\\")

    print(
        "\n=============res_of_each_line [converge_round, acc]==============="
        + ",".join(list(filters_each_line_table.keys())))
    res_to_print_matrix = []
    for key in expected_method_names:
        res_of_each_line_conver_acc_trade[key] = []
        dataset_num = 2 if "cola" in list(
            filters_each_line_table.keys()) else 3
        for i in range(dataset_num):
            res_of_each_line_conver_acc_trade[key].extend(
                [str(res_of_each_line_efficiency[key][i * 4 + 3])] + \
                # [str(res_of_each_line_efficiency[key][i * 4 + 4])] + \
                ["{:.2f}".format(v * 100) if v != "-" else v for v in res_of_each_line_fair[key][i * 3:i * 3 + 1]]
            )

        res_to_print = [str(v) for v in res_of_each_line_conver_acc_trade[key]]
        res_to_print = [key] + res_to_print
        res_to_print_matrix.append(res_to_print)
        # print("&".join(res_to_print) + "\\\\")

    colum_order_per_data = ["-", "+"]
    # "+" indicates the larger, the better
    rank_order = colum_order_per_data * len(filters_each_line_table)
    res_to_print_matrix = highlight_tex_res_in_table(
        res_to_print_matrix,
        rank_order=rank_order,
        filter_out=["Global-Train", "Isolated"],
        convergence_case=True)
    for res_to_print in res_to_print_matrix:
        print("&".join(res_to_print) + "\\\\")


import json

with open('best_res_all_metric.json', 'r') as fp:
    all_res_structed_load = json.load(fp)
    for expname_tag in expected_expname_tag:
        if "repeat" in expname_tag:
            continue
        for metric in column_names_generalization + column_names_efficiency + column_names_fair:
            all_res_structed[expname_tag][metric] = all_res_structed_load[
                expname_tag][metric]

# add all res to a df
import pandas as pd


def load_data_to_pd(use_repeat_res=False):
    all_res_for_pd = []
    for expname_tag in expected_expname_tag:
        if not use_repeat_res:
            if "repeat" in expname_tag:
                continue
        else:
            if not "repeat" in expname_tag:
                continue
        res = expname_tag.split("_")  # method, data, seed
        for metric in column_names_generalization + column_names_fair + column_names_efficiency:
            res.append(all_res_structed[expname_tag][metric])
        s = "-"
        alpha = "-"
        if "FEMNIST-s0" in res[1]:
            s = float(res[1].replace("FEMNIST-s0", "0."))
        if "cifar10-alpha0" in res[1]:
            alpha = float(res[1].replace("cifar10-alpha0", "0."))
        elif "cifar10-alpha" in res[1]:
            alpha = float(res[1].replace("cifar10-alpha", ""))
        res.append(s)
        res.append(alpha)
        total_com_bytes = unit_size_to_bytes(res[-5]) + unit_size_to_bytes(
            res[-4])
        total_flops = unit_size_to_bytes(res[-6])
        res.append(total_com_bytes)
        res.append(total_flops)
        all_res_for_pd.append(res)

    all_res_pd = pd.DataFrame().from_records(
        all_res_for_pd,
        columns=["method", "data", "seed"] + column_names_generalization +
        column_names_fair + column_names_efficiency +
        ["s", "alpha", "communication_bytes", "total_flops"])
    return all_res_pd


def plot_generalization_lines(all_res_pd, data_cate, data_cate_name):
    import seaborn as sns
    from matplotlib import pyplot as plt
    import matplotlib.pylab as pylab

    plt.clf()
    sns.set()
    fig, axes = plt.subplots(1, 3, figsize=(6, 4))
    print(all_res_pd.columns.tolist())

    plot_data = all_res_pd.loc[all_res_pd["data"].isin(data_cate)]

    plot_data = plot_data.loc[plot_data["method"] != "Global-Train"]
    plot_data = plot_data.loc[plot_data["method"] != "Isolated"]
    plot_data = plot_data.loc[plot_data["method"] != "FedOpt"]
    plot_data = plot_data.loc[plot_data["method"] != "FedOpt-FT"]
    filter_out_methods = ["Global-Train", "Isolated", "FedOpt", "FedOpt-FT"]
    for i, metric in enumerate(column_names_generalization):
        plt.clf()
        sns.set()
        fig, axes = plt.subplots(1, 1, figsize=(2, 3))
        x = "data"
        if data_cate_name == "femnist_all":
            x = "s"
        if data_cate_name == "cifar10_all":
            x = "alpha"

        ax = sns.lineplot(
            ax=axes,
            data=plot_data,
            x=x,
            y=metric,
            hue="method",
            style="method",
            markers=True,
            dashes=True,
            hue_order=[
                m for m in expected_method_names if m not in filter_out_methods
            ],
            sort=True,
        )
        ax.set(ylabel=column_names_generalization_for_plot[i])
        plt.gca().invert_xaxis()

        if data_cate_name == "cifar10_all":
            ax.set_xscale('log')

        plt.legend(bbox_to_anchor=(1, 1), loc=2, ncol=2, borderaxespad=0.)
        plt.tight_layout()
        plt.savefig(f"generalization_all_{data_cate_name}_{i}.pdf",
                    bbox_inches='tight',
                    pad_inches=0)

        plt.show()


def plot_tradeoff(all_res_pd, data_cate, data_cate_name, metric_a, metric_b,
                  fig_time):
    import seaborn as sns
    from matplotlib import pyplot as plt
    import matplotlib.pylab as pylab

    plt.clf()
    sns.set()
    print(all_res_pd.columns.tolist())

    plot_data = all_res_pd.loc[all_res_pd["data"].isin(data_cate)]

    plot_data = plot_data.loc[plot_data["method"] != "Global-Train"]
    plot_data = plot_data.loc[plot_data["method"] != "Isolated"]
    plot_data = plot_data.loc[plot_data["method"] != "FedOpt"]
    plot_data = plot_data.loc[plot_data["method"] != "FedOpt-FT"]
    filter_out_methods = ["Global-Train", "Isolated", "FedOpt", "FedOpt-FT"]
    plt.clf()
    sns.set()
    fig, axes = plt.subplots(1, 1, figsize=(2, 3))

    ax = sns.scatterplot(ax=axes,
                         data=plot_data,
                         x=metric_a,
                         y=metric_b,
                         hue="method",
                         style="method",
                         markers=True,
                         hue_order=[
                             m for m in expected_method_names
                             if m not in filter_out_methods
                         ],
                         s=100)
    ax.set(xlabel=column_name_for_plot[metric_a],
           ylabel=column_name_for_plot[metric_b])
    # plt.gca().invert_xaxis()
    if metric_a == "total_flops":
        ax.set_xscale('log')

    if data_cate_name == "cifar10_all":
        ax.set_xscale('log')

    plt.legend(bbox_to_anchor=(1, 1), loc=2, ncol=2, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(f"{fig_time}_{data_cate_name}.pdf",
                bbox_inches='tight',
                pad_inches=0)

    plt.show()


if __name__ == "__main__":
    load_best_repeat_res(["1", "repeat"])
    avg_res_of_seeds()

    print_paper_table_from_repeat(filters_each_line_main_table)
    print_paper_table_from_repeat(filters_each_line_femnist_all_s)
    print_paper_table_from_repeat(filters_each_line_all_cifar10)
    print_paper_table_from_repeat(filters_each_line_all_nlp)
    print_paper_table_from_repeat(filters_each_line_all_graph)

    all_res_pd = load_data_to_pd(use_repeat_res=False)
    all_res_pd_repeat = load_data_to_pd(use_repeat_res=True)


def plot_line_figs():
    plot_generalization_lines(all_res_pd,
                              list(filters_each_line_femnist_all_s.keys()),
                              data_cate_name="femnist_all")
    plot_generalization_lines(all_res_pd,
                              list(filters_each_line_all_cifar10.keys()),
                              data_cate_name="cifar10_all")


def plot_trade_off_figs(filters_each_line_main_table):
    for data_name in list(filters_each_line_main_table.keys()):
        plot_tradeoff(all_res_pd_repeat, [data_name],
                      data_cate_name=data_name,
                      metric_a="communication_bytes",
                      metric_b="best_client_summarized_weighted_avg/test_acc",
                      fig_time="com-acc")

    for data_name in list(filters_each_line_main_table.keys()):
        plot_tradeoff(all_res_pd_repeat, [data_name],
                      data_cate_name=data_name,
                      metric_a="total_flops",
                      metric_b="best_client_summarized_weighted_avg/test_acc",
                      fig_time="flops-acc")

    for data_name in list(filters_each_line_main_table.keys()):
        plot_tradeoff(all_res_pd_repeat, [data_name],
                      data_cate_name=data_name,
                      metric_a="sys_avg/global_convergence_round",
                      metric_b="best_client_summarized_weighted_avg/test_acc",
                      fig_time="round-acc")
