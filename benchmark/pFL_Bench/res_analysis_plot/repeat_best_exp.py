
import wandb
from collections import OrderedDict

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
        ("FEMNIST-s02",
         {"$and":
             [
                 {"config.data.type": "femnist"},
                 {"config.federate.sample_client_rate": 0.2},
                 {"state": "finished"},
             ]
         }
         ),
        # ("cifar10-alpha05",
        #  {"$and":
        #      [
        #          {"config.data.type": "CIFAR10@torchvision"},
        #          {"config.data.splitter_args": [{"alpha": 0.5}]},
        #      ]
        #  }
        #  ),
        ("sst2",
         {"$and":
             [
                 {"config.data.type": "sst2@huggingface_datasets"},
             ]
         }
         ),
        ("pubmed",
         {"$and":
             [
                 {"config.data.type": "pubmed"},
             ]
         }
         ),
    ]
)

filters_each_line_all_cifar10 = OrderedDict(
    # {dataset_name: filter}
    [
        ("cifar10-alpha5",
         {"$and":
             [
                 {"config.data.type": "CIFAR10@torchvision"},
                 {"config.data.splitter_args": [{"alpha": 5}]},
             ]
         }
         ),
        ("cifar10-alpha05",
         {"$and":
             [
                 {"config.data.type": "CIFAR10@torchvision"},
                 {"config.data.splitter_args": [{"alpha": 0.5}]},
             ]
         }
         ),
        ("cifar10-alpha01",
         {"$and":
             [
                 {"config.data.type": "CIFAR10@torchvision"},
                 {"config.data.splitter_args": [{"alpha": 0.1}]},
             ]
         }
         ),
    ]
)

filters_each_line_femnist_all_s = OrderedDict(
    # {dataset_name: filter}
    [
        ("FEMNIST-s02",
         {"$and":
             [
                 {"config.data.type": "femnist"},
                 {"config.federate.sample_client_rate": 0.2},
                 {"state": "finished"},
             ]
         }
         ),
        ("FEMNIST-s01",
         {"$and":
             [
                 {"config.data.type": "femnist"},
                 {"config.federate.sample_client_rate": 0.1},
                 {"state": "finished"},
             ]
         }
         ),
        ("FEMNIST-s005",
         {"$and":
             [
                 {"config.data.type": "femnist"},
                 {"config.federate.sample_client_rate": 0.05},
                 {"state": "finished"},
             ]
         }
         ),

    ]
)

filters_each_line_all_graph = OrderedDict(
    # {dataset_name: filter}
    [
        ("pubmed",
         {"$and":
             [
                 {"config.data.type": "pubmed"},
             ]
         }
         ),
        ("cora",
         {"$and":
             [
                 {"config.data.type": "cora"},
             ]
         }
         ),
        ("citeseer",
         {"$and":
             [
                 {"config.data.type": "citeseer"},
             ]
         }
         ),
    ]
)

filters_each_line_all_nlp = OrderedDict(
    # {dataset_name: filter}
    [
        ("cola",
         {"$and":
             [
                 {"config.data.type": "cola@huggingface_datasets"},
             ]
         }
         ),
        ("sst2",
         {"$and":
             [
                 {"config.data.type": "sst2@huggingface_datasets"},
             ]
         }
         ),
    ]
)


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
sorted_keys = OrderedDict(
    [("global-train", "Global Train"),
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
)
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


def get_runs_filter_by(filter_name, filters_each_line_table):
    filter = filters_each_line_table[filter_name]
    filtered_runs = api.runs(name_project, filters=filter)
    return filtered_runs

order = '-' + 'summary_metrics.best_client_summarized_weighted_avg/val_acc'

def generate_repeat_scripts(best_cfg_path, seed_sets=None):
    file_cnt = 0
    if seed_sets is None:
        seed_sets = [2, 3]
    from os import listdir
    from os.path import isfile, join
    onlyfiles = [f for f in listdir(best_cfg_path) if isfile(join(best_cfg_path, f))]
    for file_name in onlyfiles:
        exp_name = file_name
        exp_name = exp_name.replace(".yaml", "")
        method, data = exp_name.split("_")
        for seed in seed_sets:
            print(f"python federatedscope/main.py --cfg scripts/personalization_exp_scripts/pfl_bench/yaml_best_runs/{file_name} seed {seed} expname_tag {exp_name}_seed{seed} wandb.name_project pfl-bench-best-repeat")
            file_cnt += 1
            if file_cnt % 10 == 0:
                print(f"Seed={seed}, totally generated {file_cnt} run scripts\n\n")

    print(f"Seed={seed_sets}, totally generated {file_cnt} run scripts")
    print(f"=============================== END ===============================")

seed_sets = [2, 3]
for seed in seed_sets:
    generate_repeat_scripts("/mnt/daoyuanchen.cdy/FederatedScope/scripts/personalization_exp_scripts/pfl_bench/yaml_best_runs", seed_sets=[seed])