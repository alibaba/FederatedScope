import copy
import json
import os

import wandb
from collections import OrderedDict

import yaml

from scripts.personalization_exp_scripts.pfl_bench.res_analysis_plot.render_paper_res import highlight_tex_res_in_table

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

filters_each_line_all_rec = OrderedDict(
    # {dataset_name: filter}
    [
        ("movielens1m", {
            "$and": [
                {
                    "config.data.type": "HFLMovieLens1M"
                },
            ]
        }),
        ("movielens10m", {
            "$and": [
                {
                    "config.data.type": "VFLMovieLens10M"
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
column_names_generalization_loss = [
    "best_client_summarized_weighted_avg/test_avg_loss",
    "best_unseen_client_summarized_weighted_avg_unseen/test_avg_loss",
    "participation_gap"
]
column_names_fair_loss = [
    "best_client_summarized_avg/test_avg_loss",
    "best_client_summarized_fairness/test_avg_loss_std",
    "best_client_summarized_fairness/test_avg_loss_top_decile"  # reverse
]
column_names_efficiency = [
    "sys_avg/total_flops",
    "sys_avg/total_upload_bytes",
    "sys_avg/total_download_bytes",
    "sys_avg/global_convergence_round",
    # "sys_avg/local_convergence_round"
]
sorted_method_name_pair = [
    ("global-train", "Global-Train"),
    ("isolated-train", "Isolated"),
    ("fedavg", "FedAvg"),
    ("fedavg-ft", "FedAvg-FT"),
    #("fedopt", "FedOpt"),
    #("fedopt-ft", "FedOpt-FT"),
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
sorted_method_name_pair_rec = [
    # for the recommendation dataset, the mf model does not have the bn parameter
    ("global-train", "Global-Train"),
    ("isolated-train", "Isolated"),
    ("fedavg", "FedAvg"),
    ("fedavg-ft", "FedAvg-FT"),
    ("fedavg-fedopt-ft", "FedAvg-FT-FedOpt"),
    ("fedopt", "FedOpt"),
    ("fedopt-ft", "FedOpt-FT"),
    ("pfedme", "pFedMe"),
    ("ft-pfedme", "pFedMe-FT"),
    ("ditto", "Ditto"),
    #("ditto-fedopt", "Ditto-FedOpt"),
    ("ditto-ft", "Ditto-FT"),
    ("ditto-fedopt-ft", "Ditto-FT-FedOpt"),
    ("fedem", "FedEM"),
    ("fedem-ft", "FedEM-FT"),
    ("fedem-fedopt-ft", "FedEM-FT-FedOpt"),
]
sorted_method_name_to_print = OrderedDict(sorted_method_name_pair)
expected_keys = set(list(sorted_method_name_to_print.keys()))
expected_method_names = list(sorted_method_name_to_print.values())

expected_datasets_name = [
    "cola",
    "sst2",
    "pubmed",
    "cora",
    "citeseer",
    "cifar10-alpha5",
    "cifar10-alpha05",
    "cifar10-alpha01",
    "FEMNIST-s02",
    "FEMNIST-s01",
    "FEMNIST-s005",
]
expected_datasets_name_rec = ["HFLMovieLens1M", "VFLMovieLens10M"]
expected_seed_set = ["0"]
expected_expname_tag = set()

for method_name in expected_method_names:
    for dataset_name in expected_datasets_name:
        for seed in expected_seed_set:
            expected_expname_tag.add(
                f"{method_name}_{dataset_name}_seed{seed}")
from collections import defaultdict

all_res_structed = defaultdict(dict)
for expname_tag in expected_expname_tag:
    for metric in column_names_generalization + column_names_efficiency + column_names_fair:
        all_res_structed[expname_tag][metric] = "-"

sorted_method_name_to_print_rec = OrderedDict(sorted_method_name_pair_rec)
expected_keys_rec = set(list(sorted_method_name_to_print_rec.keys()))
expected_method_names_rec = list(sorted_method_name_to_print_rec.values())
expected_expname_tag_rec = set()

for method_name in expected_method_names_rec:
    for dataset_name in expected_datasets_name_rec:
        for seed in expected_seed_set:
            expected_expname_tag_rec.add(
                f"{method_name}_{dataset_name}_seed{seed}")

all_res_structed_rec = defaultdict(dict)
for expname_tag in expected_expname_tag_rec:
    for metric in column_names_generalization_loss + column_names_efficiency + column_names_fair_loss:
        all_res_structed_rec[expname_tag][metric] = "-"


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


def get_best_runs_within_sweep(sweep_id_lists):
    best_run_list = []
    for sweep_id in sweep_id_lists:
        sweep = api.sweep(f"{name_project}/{sweep_id}")
        best_run = sweep.best_run()
        best_run_list.append(best_run)


def get_sweep_filter_by(filter_name, filters_each_line_table):
    filter = filters_each_line_table[filter_name]
    filtered_runs = api.runs(name_project, filters=filter)
    filtered_sweep_ids = set()
    check_run_cnt = 0
    # may hang on
    for run in filtered_runs:
        if run.sweep is not None:
            filtered_sweep_ids.add(run.sweep.id)
        check_run_cnt += 1
    print(f"check_run_cnt is {check_run_cnt}")
    return list(filtered_sweep_ids)


order_acc = '-' + 'summary_metrics.best_client_summarized_weighted_avg/val_acc'
order_loss = '+' + 'summary_metrics.best_client_summarized_weighted_avg/val_avg_loss'


def print_table_datasets_list(filters_each_line_table,
                              order,
                              all_res_structed,
                              column_names_generalization,
                              column_names_efficiency,
                              column_names_fair,
                              expected_keys,
                              sorted_method_name_to_print,
                              percent=True):
    res_of_each_line_generalization = OrderedDict()
    res_of_each_line_fair = OrderedDict()
    res_of_each_line_efficiency = OrderedDict()
    res_of_each_line_commu_acc_trade = OrderedDict()
    res_of_each_line_conver_acc_trade = OrderedDict()
    res_of_all_sweeps = OrderedDict()
    for data_name in filters_each_line_table:
        unseen_keys = copy.copy(expected_keys)
        print(f"======= processing dataset {data_name}")
        sweep_ids = get_sweep_filter_by(data_name, filters_each_line_table)
        for sweep_id in sweep_ids:
            sweep = api.sweep(f"{name_project}/{sweep_id}")
            run_header = sweep.name
            if sweep.order != order:
                print(f"un-expected order for {run_header}")
            best_run = sweep.best_run()
            res_all_generalization = []
            res_all_fair = []
            res_all_efficiency = []
            if best_run.state != "finished":
                print(
                    f"==================Waring: the best_run with id={best_run} has state {best_run.state}. "
                    f"In weep_id={sweep_id}, sweep_name={run_header}")
            else:
                print(f"Finding the best_run with id={best_run}. "
                      f"In sweep_id={sweep_id}, sweep_name={run_header}")

            # for generalization results
            wrong_sweep = False
            if "isolated" in run_header or "global" in run_header:
                try:
                    res = best_run.summary[column_names_generalization[0]]
                    res_all_generalization.append(res)
                except KeyError:
                    print(
                        f"KeyError with key={column_names_generalization[0]}, sweep_id={sweep_id}, sweep_name={run_header}, best_run_id={best_run.id}"
                    )
                    wrong_sweep = True
                if wrong_sweep:
                    continue
                res_all_generalization.append("-")  # un-seen
                res_all_generalization.append("-")  # Gap
            else:
                for column_name in column_names_generalization[0:2]:
                    try:
                        res = best_run.summary[column_name]
                        res_all_generalization.append(res)
                    except KeyError:
                        print(
                            f"KeyError with key={column_name}, sweep_id={sweep_id}, sweep_name={run_header}, best_run_id={best_run.id}"
                        )
                        wrong_sweep = True
                if wrong_sweep:
                    continue
                res_all_generalization.append(res_all_generalization[-1] -
                                              res_all_generalization[-2])
            # -============== for fairness results ======
            for column_name in column_names_fair:
                if "global" in run_header:
                    res_all_fair.append("-")
                    res_all_fair.append("-")
                    res_all_fair.append("-")
                else:
                    try:
                        res = best_run.summary[column_name]
                        res_all_fair.append(res)
                    except KeyError:
                        print(
                            f"KeyError with key={column_name}, sweep_id={sweep_id}, sweep_name={run_header}, best_run_id={best_run.id}"
                        )
                        res_all_fair.append("-")
                        wrong_sweep = True

            # -============== for efficiency results ======
            for column_name in column_names_efficiency:
                try:
                    res = best_run.summary[column_name]
                    contain_unit = False
                    for size_unit in ["K", "M", "G", "T", "P", "E", "Z", "Y"]:
                        if size_unit in str(res):
                            contain_unit = True
                    if not contain_unit:
                        res = bytes_to_unit_size(float(res))

                    res_all_efficiency.append(res)
                except KeyError:
                    print(
                        f"KeyError with key={column_name}, sweep_id={sweep_id}, sweep_name={run_header}, best_run_id={best_run.id}"
                    )
                    wrong_sweep = True
                    res_all_efficiency.append("-")

            run_header = str.lower(run_header)
            run_header = run_header.replace("movielens1m-h", "movielens1m")
            run_header = run_header.replace("movielens10m-v", "movielens10m")

            # fix some run_header error
            # best_run_cfg = json.loads(best_run.json_config)
            best_run_cfg = best_run.config

            def remove_a_key(d, remove_key):
                if isinstance(d, dict):
                    for key in list(d.keys()):
                        if key == remove_key:
                            del d[key]
                        else:
                            remove_a_key(d[key], remove_key)

            remove_a_key(best_run_cfg, "cfg_check_funcs")
            old_run_header = run_header
            if best_run_cfg["trainer"]["finetune"][
                    "before_eval"] is True and "ft" not in run_header:
                run_header = run_header + ",ft"
            if best_run_cfg["trainer"]["finetune"][
                    "before_eval"] is False and "ft" in run_header:
                run_header = run_header.replace(",ft", "")
            if best_run_cfg["fedopt"][
                    "use"] is True and "fedopt" not in run_header:
                run_header = run_header + ",fedopt"
            if best_run_cfg["fedopt"][
                    "use"] is False and "fedopt" in run_header:
                run_header = run_header.replace(",fedopt", "")
            if old_run_header != run_header:
                print(
                    f"processed {old_run_header} to new run header {run_header}"
                )

            if run_header not in res_of_all_sweeps:
                res_of_all_sweeps[run_header] = res_all_generalization
                sweep_name_2_id[run_header] = sweep_id
            else:
                print(
                    f"processed duplicated sweep with name {run_header}, plz check it with id {sweep_id}. "
                    f"The first appeared sweep has id {sweep_name_2_id[run_header]}"
                )

                while run_header + "_dup" in res_of_all_sweeps:
                    run_header = run_header + "_dup"
                run_header = run_header + "dup"
                print(f"processed to new run header {run_header}")
                res_of_all_sweeps[run_header] = res_all_generalization

            # pre-process the expname, step 1. split by ",", get the method header atomic elements
            run_header = run_header.replace("-", ",")
            run_header = run_header.replace("+", ",")
            split_res = run_header.split(",")
            filter_split_res = []
            for sub in split_res:
                # filter the dataset name
                if "femnist" in sub or "cifar" in sub or "cora" in sub or "cola" in sub or "pubmed" in sub or "citeseer" in sub or "sst2" in sub \
                        or "s02" in sub or "s005" in sub or "s01" in sub \
                        or "alpha5" in sub or "alpha0.5" in sub or "alpha0.1" in sub \
                        or "movielen" in sub:
                    pass
                else:
                    filter_split_res.append(sub)
            # pre-process the expname, step 2. combining the method header elements with "-"
            method_header = "-".join(sorted(filter_split_res))
            if method_header in unseen_keys:
                unseen_keys.remove(method_header)

            # save all res into the structured dict
            cur_seed = best_run_cfg["seed"]
            #`if method_header in ["fedopt", "fedopt-ft"]:
            #`    continue
            exp_name_current = f"{sorted_method_name_to_print[method_header]}_{data_name}_seed{cur_seed}"
            for i, metric in enumerate(column_names_generalization):
                all_res_structed[exp_name_current][
                    metric] = res_all_generalization[i]
            for i, metric in enumerate(column_names_efficiency):
                all_res_structed[exp_name_current][
                    metric] = res_all_efficiency[i]
            for i, metric in enumerate(column_names_fair):
                all_res_structed[exp_name_current][metric] = res_all_fair[i]

            # save config
            parent_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "..")
            best_cfg_dir = os.path.join(parent_dir, "yaml_best_runs")
            os.makedirs(best_cfg_dir, exist_ok=True)
            yaml_f_name = f"{sorted_method_name_to_print[method_header]}_{data_name}.yaml"
            with open(os.path.join(best_cfg_dir, yaml_f_name), 'w') as yml_f:
                yaml.dump(best_run_cfg, yml_f, allow_unicode=True)

            if method_header not in res_of_each_line_generalization:
                res_of_each_line_generalization[
                    method_header] = res_all_generalization
                res_of_each_line_fair[method_header] = res_all_fair
                res_of_each_line_efficiency[method_header] = res_all_efficiency
            else:
                res_of_each_line_generalization[method_header].extend(
                    res_all_generalization)
                res_of_each_line_fair[method_header].extend(res_all_fair)
                res_of_each_line_efficiency[method_header].extend(
                    res_all_efficiency)

        for missing_header in unseen_keys:
            print(
                f"the header is missing {missing_header} in dataset {data_name}"
            )
            if missing_header not in res_of_each_line_generalization:
                res_of_each_line_generalization[missing_header] = ["-"] * 3
                res_of_each_line_fair[missing_header] = ["-"] * 3
                res_of_each_line_efficiency[missing_header] = ["-"] * 4
            else:
                res_of_each_line_generalization[missing_header].extend(["-"] *
                                                                       3)
                res_of_each_line_fair[missing_header].extend(["-"] * 3)
                res_of_each_line_efficiency[missing_header].extend(["-"] * 4)

    print("\n=============res_of_each_line [Generalization]===============" +
          ",".join(list(filters_each_line_table.keys())))
    # Acc, Unseen-ACC, Delta
    res_to_print_matrix = []
    times_ratio = 100 if percent else 1
    for key in sorted_method_name_to_print:
        res_to_print = [
            "{:.2f}".format(v * times_ratio) if v != "-" else v
            for v in res_of_each_line_generalization[key]
        ]
        res_to_print = [sorted_method_name_to_print[key]] + res_to_print
        #print(",".join(res_to_print))
        res_to_print_matrix.append(res_to_print)

    colum_order_per_data = ["-", "-",
                            "-"]  # for the loss, the smaller the better
    # "+" indicates the larger, the better
    rank_order = colum_order_per_data * len(filters_each_line_table)
    res_to_print_matrix = highlight_tex_res_in_table(res_to_print_matrix,
                                                     rank_order=rank_order)
    for res_to_print in res_to_print_matrix:
        print("&".join(res_to_print) + "\\\\")

    print("\n=============res_of_each_line [Fairness]===============" +
          ",".join(list(filters_each_line_table.keys())))
    res_to_print_matrix = []
    for key in sorted_method_name_to_print:
        res_to_print = [
            "{:.2f}".format(v * times_ratio) if v != "-" else v
            for v in res_of_each_line_fair[key]
        ]
        res_to_print = [sorted_method_name_to_print[key]] + res_to_print
        #print(",".join(res_to_print))
        res_to_print_matrix.append(res_to_print)

    colum_order_per_data = ["-", "-", "-"]
    # "+" indicates the larger, the better
    rank_order = colum_order_per_data * len(filters_each_line_table)
    res_to_print_matrix = highlight_tex_res_in_table(
        res_to_print_matrix,
        rank_order=rank_order,
        filter_out=["Global-Train"])
    for res_to_print in res_to_print_matrix:
        print("&".join(res_to_print) + "\\\\")

    #print("\n=============res_of_each_line [All Efficiency]===============" + ",".join(
    #    list(filters_each_line_table.keys())))
    ## FLOPS, UPLOAD, DOWNLOAD

    #for key in sorted_method_name_to_print:
    #    res_to_print = [str(v) for v in res_of_each_line_efficiency[key]]
    #    res_to_print = [sorted_method_name_to_print[key]] + res_to_print
    #    print(",".join(res_to_print))
    print(
        "\n=============res_of_each_line [flops, communication, acc/loss]==============="
        + ",".join(list(filters_each_line_table.keys())))
    res_to_print_matrix = []

    dataset_num = 2 if "cola" or "movie" in list(
        filters_each_line_table.keys()) else 3
    for key in sorted_method_name_to_print:
        res_of_each_line_commu_acc_trade[key] = []
        for i in range(dataset_num):
            try:
                res_of_each_line_commu_acc_trade[key].extend(
                    [str(res_of_each_line_efficiency[key][i * 4])] + \
                    [str(res_of_each_line_efficiency[key][i * 4 + 1])] + \
                    ["{:.2f}".format(v * times_ratio) if v != "-" else v for v in
                     res_of_each_line_generalization[key][i * 3:i * 3 + 1]]
                )
            except:
                print(f"error with index i={i}")

        res_to_print = [str(v) for v in res_of_each_line_commu_acc_trade[key]]
        res_to_print = [sorted_method_name_to_print[key]] + res_to_print
        #print(",".join(res_to_print))
        res_to_print_matrix.append(res_to_print)

    colum_order_per_data = ["-", "-", "-"]
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
        "\n=============res_of_each_line [converge_round, acc/loss]==============="
        + ",".join(list(filters_each_line_table.keys())))
    res_to_print_matrix = []
    for key in sorted_method_name_to_print:
        res_of_each_line_conver_acc_trade[key] = []
        for i in range(dataset_num):
            res_of_each_line_conver_acc_trade[key].extend(
                [str(res_of_each_line_efficiency[key][i * 4 + 3])] + \
                # [str(res_of_each_line_efficiency[key][i * 4 + 4])] + \
                ["{:.2f}".format(v * times_ratio) if v != "-" else v for v in
                 res_of_each_line_fair[key][i * 3:i * 3 + 1]]
            )

        res_to_print = [str(v) for v in res_of_each_line_conver_acc_trade[key]]
        res_to_print = [sorted_method_name_to_print[key]] + res_to_print
        res_to_print_matrix.append(res_to_print)
        #print(",".join(res_to_print))

    colum_order_per_data = ["-", "-"]
    # "+" indicates the larger, the better
    rank_order = colum_order_per_data * len(filters_each_line_table)
    res_to_print_matrix = highlight_tex_res_in_table(
        res_to_print_matrix,
        rank_order=rank_order,
        filter_out=["Global-Train", "Isolated"],
        convergence_case=True)
    for res_to_print in res_to_print_matrix:
        print("&".join(res_to_print) + "\\\\")


def print_res_non_rec():
    print_table_datasets_list(filters_each_line_main_table, order_acc,
                              all_res_structed, column_names_generalization,
                              column_names_efficiency, column_names_fair,
                              expected_keys, sorted_method_name_to_print)
    print_table_datasets_list(filters_each_line_femnist_all_s, order_acc,
                              all_res_structed, column_names_generalization,
                              column_names_efficiency, column_names_fair,
                              expected_keys, sorted_method_name_to_print)
    print_table_datasets_list(filters_each_line_all_cifar10, order_acc,
                              all_res_structed, column_names_generalization,
                              column_names_efficiency, column_names_fair,
                              expected_keys, sorted_method_name_to_print)
    print_table_datasets_list(filters_each_line_all_nlp, order_acc,
                              all_res_structed, column_names_generalization,
                              column_names_efficiency, column_names_fair,
                              expected_keys, sorted_method_name_to_print)
    print_table_datasets_list(filters_each_line_all_graph, order_acc,
                              all_res_structed, column_names_generalization,
                              column_names_efficiency, column_names_fair,
                              expected_keys, sorted_method_name_to_print)
    for expname_tag in expected_expname_tag:
        for metric in column_names_generalization + column_names_efficiency + column_names_fair:
            if all_res_structed[expname_tag][metric] == "-":
                print(f"Missing {expname_tag} for metric {metric}")

    with open('best_res_all_metric.json', 'w') as fp:
        json.dump(all_res_structed, fp)


def print_res_rec():
    print_table_datasets_list(
        filters_each_line_all_rec,
        order_loss,
        all_res_structed_rec,
        column_names_generalization_loss,
        column_names_efficiency,
        column_names_fair_loss,
        expected_keys=expected_keys_rec,
        sorted_method_name_to_print=sorted_method_name_to_print_rec,
        percent=False)
    #for expname_tag in expected_expname_tag_rec:
    #    for metric in column_names_generalization_loss + column_names_efficiency + col#umn_names_fair_loss:
    #        if all_res_structed_rec[expname_tag][metric] == "-":
    #print(f"Missing {expname_tag} for metric {metric}")

    with open('best_res_all_metric_rec.json', 'w') as fp:
        json.dump(all_res_structed_rec, fp)


print_res_non_rec()
print_res_rec()
