import copy
import json
import os

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
sorted_keys = OrderedDict([
    ("global-train", "Global Train"),
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
])
expected_keys = set(list(sorted_keys.keys()))


def bytes_to_unit_size(size_bytes):
    import math
    if size_bytes == 0:
        return "0"
    size_name = ("", "K", "M", "G", "T", "P", "E", "Z", "Y")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s}{size_name[i]}"


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


def get_runs_filter_by(filter_name, filters_each_line_table):
    filter = filters_each_line_table[filter_name]
    filtered_runs = api.runs(name_project, filters=filter)
    return filtered_runs


order = '-' + 'summary_metrics.best_client_summarized_weighted_avg/val_acc'


def print_table_datasets_list(filters_each_line_table):
    res_of_each_line_generalization = OrderedDict()
    res_of_each_line_fair = OrderedDict()
    res_of_each_line_efficiency = OrderedDict()
    res_of_each_line_commu_acc_trade = OrderedDict()
    res_of_each_line_conver_acc_trade = OrderedDict()
    res_of_all_sweeps = OrderedDict()
    for data_name in filters_each_line_table:
        unseen_keys = copy.copy(expected_keys)
        print(f"======= processing dataset {data_name}")
        runs_ids = get_sweep_filter_by(data_name, filters_each_line_table)
        for best_run in runs_ids:
            res_all_generalization = []
            res_all_fair = []
            res_all_efficiency = []
            if best_run.state != "finished":
                print(
                    f"==================Waring: the best_run with id={best_run} has state {best_run.state}. "
                )

            def remove_a_key(d, remove_key):
                if isinstance(d, dict):
                    for key in list(d.keys()):
                        if key == remove_key:
                            del d[key]
                        else:
                            remove_a_key(d[key], remove_key)

            remove_a_key(best_run_cfg, "cfg_check_funcs")
            best_run_cfg = best_run.config
            run_header = best_run_cfg.expname_tag
            run_header = run_header.split("_")[0]

            # for generalization results
            if "isolated" in run_header.lower(
            ) or "global" in run_header.lower():
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

            old_run_header = run_header
            if best_run_cfg["trainer"]["finetune"][
                    "before_eval"] is True and "ft" not in run_header:
                run_header = run_header + ",ft"
            elif best_run_cfg["fedopt"][
                    "use"] is True and "fedopt" not in run_header:
                run_header = run_header + ",fedopt"
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

            run_header = run_header.replace("-", ",")
            run_header = run_header.replace("+", ",")
            split_res = run_header.split(",")
            filter_split_res = []
            for sub in split_res:
                if "femnist" in sub or "cifar" in sub or "cora" in sub or "cola" in sub or "pubmed" in sub or "citeseer" in sub or "sst2" in sub \
                        or "s02" in sub or "s005" in sub or "s01" in sub \
                        or "alpha5" in sub or "alpha0.5" in sub or "alpha0.1" in sub:
                    pass
                else:
                    filter_split_res.append(sub)
            method_header = "-".join(sorted(filter_split_res))
            if method_header in unseen_keys:
                unseen_keys.remove(method_header)

            # save config
            parent_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "..")
            best_cfg_dir = os.path.join(parent_dir, "yaml_best_rums")
            os.makedirs(best_cfg_dir, exist_ok=True)
            yaml_f_name = f"best_{sorted_keys[method_header]}_on_{data_name}.yaml"
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
    for key in sorted_keys:
        res_to_print = [
            "{:.2f}".format(v * 100) if v != "-" else v
            for v in res_of_each_line_generalization[key]
        ]
        res_to_print = [sorted_keys[key]] + res_to_print
        print(",".join(res_to_print))
    print("\n=============res_of_each_line [Fairness]===============" +
          ",".join(list(filters_each_line_table.keys())))
    for key in sorted_keys:
        res_to_print = [
            "{:.2f}".format(v * 100) if v != "-" else v
            for v in res_of_each_line_fair[key]
        ]
        res_to_print = [sorted_keys[key]] + res_to_print
        print(",".join(res_to_print))
    print("\n=============res_of_each_line [All Efficiency]===============" +
          ",".join(list(filters_each_line_table.keys())))
    # FLOPS, UPLOAD, DOWNLOAD
    for key in sorted_keys:
        res_to_print = [str(v) for v in res_of_each_line_efficiency[key]]
        res_to_print = [sorted_keys[key]] + res_to_print
        print(",".join(res_to_print))
    print(
        "\n=============res_of_each_line [flops, communication, acc]==============="
        + ",".join(list(filters_each_line_table.keys())))
    for key in sorted_keys:
        res_of_each_line_commu_acc_trade[key] = []
        dataset_num = 2 if "cola" in list(
            filters_each_line_table.keys()) else 3
        for i in range(dataset_num):
            res_of_each_line_commu_acc_trade[key].extend(
                [str(res_of_each_line_efficiency[key][i * 4])] + \
                [str(res_of_each_line_efficiency[key][i * 4 + 1])] + \
                ["{:.2f}".format(v * 100) if v != "-" else v for v in res_of_each_line_fair[key][i * 3:i * 3 + 1]]
            )

        res_to_print = [str(v) for v in res_of_each_line_commu_acc_trade[key]]
        res_to_print = [sorted_keys[key]] + res_to_print
        print(",".join(res_to_print))
    print(
        "\n=============res_of_each_line [converge_round, acc]==============="
        + ",".join(list(filters_each_line_table.keys())))
    for key in sorted_keys:
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
        res_to_print = [sorted_keys[key]] + res_to_print
        print(",".join(res_to_print))
    # print("\n=============res_of_all_sweeps [Generalization]===============")
    # for key in sorted(res_of_all_sweeps.keys()):
    #     res_to_print = ["{:.2f}".format(v * 100) if v != "-" else v for v in res_of_all_sweeps[key]]
    #     res_to_print = [key] + res_to_print
    #     print(",".join(res_to_print))
    #


def generate_repeat_scripts(best_cfg_path, seed_sets=None):
    file_cnt = 0
    if seed_sets is None:
        seed_sets = [2, 3]
    from os import listdir
    from os.path import isfile, join
    onlyfiles = [
        f for f in listdir(best_cfg_path) if isfile(join(best_cfg_path, f))
    ]
    for file_name in onlyfiles:
        exp_name = file_name
        exp_name = exp_name.replace(".yaml", "")
        method, data = exp_name.split("_")
        for seed in seed_sets:
            print(
                f"python federatedscope/main.py --cfg scripts/personalization_exp_scripts/pfl_bench/yaml_best_runs/{file_name} seed {seed} expname_tag {exp_name}_seed{seed} wandb.name_project pfl-bench-best-repeat"
            )
            file_cnt += 1
            if file_cnt % 10 == 0:
                print(
                    f"Seed={seed}, totally generated {file_cnt} run scripts\n\n"
                )

    print(f"Seed={seed_sets}, totally generated {file_cnt} run scripts")
    print(
        f"=============================== END ===============================")


def generate_res_table():
    print_table_datasets_list(filters_each_line_main_table)
    print_table_datasets_list(filters_each_line_femnist_all_s)
    print_table_datasets_list(filters_each_line_all_cifar10)
    print_table_datasets_list(filters_each_line_all_nlp)
    print_table_datasets_list(filters_each_line_all_graph)


seed_sets = [2, 3]
for seed in seed_sets:
    generate_repeat_scripts(
        "/mnt/daoyuanchen.cdy/FederatedScope/scripts/personalization_exp_scripts/pfl_bench/yaml_best_runs",
        seed_sets=[seed])
