import argparse
import json
import logging
import yaml
import re


parser = argparse.ArgumentParser(description='FederatedScope result parsing')
parser.add_argument('--exp_dir',
                    help='path of exp results',
                    required=True,
                    type=str)
parser.add_argument('--project',
                    help='project name used in wandb',
                    required=True,
                    type=str)
parser.add_argument('--user',
                    help='wandb user name',
                    required=True,
                    type=str)
parser.add_argument('--need_contain_str_in_exp_dir',
                    help='whether need contain some strings in exp_dir',
                    required=False,
                    default="",
                    type=str)
parser.add_argument('--filter_str_in_exp_dir',
                    help='whether filter exp_dir that contain some strings',
                    required=False,
                    default="",
                    type=str)
args = parser.parse_args()


def contain_key_sub_str(need_contain_str_in_exp_dir, abs_path):
    for required_str in need_contain_str_in_exp_dir:
        if required_str in abs_path:
            return True
    return False


def main():
    try:
        import wandb
    except ImportError:
        logging.warning("not found wandb, plz use another parse tool")
        exit()

    from pathlib import Path
    # e.g. 'exp_out/FedEM/lr_0.01_10_bs64_on_synthetic/config.yaml'
    #      'exp_out/FedEM/lr_0.01_10_bs64_on_synthetic/exp_print.log'
    cfg_file_list = []
    exp_log_f_list = []
    for path in Path(args.exp_dir).rglob('*config.yaml'):
        abs_path = str(path)
        print(abs_path)
        if args.need_contain_str_in_exp_dir != "" and \
                not contain_key_sub_str(
                    args.need_contain_str_in_exp_dir.split(","),
                    abs_path
                ):
            continue
        if args.filter_str_in_exp_dir != "" and \
                    contain_key_sub_str(
                        args.filter_str_in_exp_dir.split(","),
                        abs_path
                    ):
            continue
        cfg_file_list.append(abs_path)
        exp_log_f_list.append(abs_path.replace("config.yaml", "exp_print.log"))

    print("will parse these exps: \n")
    print(cfg_file_list)

    for exp_i, cfg_f_name in enumerate(cfg_file_list):
        with open(exp_log_f_list[exp_i], 'r') as exp_log_f:
            log_res_best = {}
            exp_stop_normal = False
            all_log_res = []
            print(f'Process {exp_log_f_list[exp_i]}')
            last_line = None
            for line in exp_log_f:
                last_line = line
                if " Find new best result" in line:
                    # e.g.,
                    # 2022-03-22 10:48:42,562 (server:459) INFO: Find new best result for client_individual.test_acc with value 0.5911787974683544
                    parse_res = line.split("INFO: ")[1].split("with value")
                    best_key, best_val = parse_res[-2], parse_res[-1]
                    # client_individual.test_acc -> client_individual/test_acc
                    best_key = best_key.replace("Find new best result for", "").replace(".", "/")
                    log_res_best[best_key.strip()] = float(best_val.strip())

                if "'Role': 'Server #'" in line:
                    res = line.split("INFO: ")[1].replace("\'", "\"")
                    res = json.loads(s=res)
                    if res['Role'] == 'Server #':
                        cur_round = res['Round']
                    res.pop('Role')
                    if cur_round != "Final" and 'Results_raw' in res:
                        res.pop('Results_raw')

                    log_res = {}
                    for key, val in res.items():
                        if not isinstance(val, dict):
                            log_res[key] = val
                        else:
                            if cur_round != "Final":
                                for key_inner, val_inner in val.items():
                                    assert not isinstance(val_inner, dict), "Un-expected log format"
                                    log_res[f"{key}/{key_inner}"] = val_inner

                            else:
                                exp_stop_normal = True
                            #     log_res_best = {}
                            #     for best_res_type, val_dict in val.items():
                            #         for key_inner, val_inner in val_dict.items():
                            #             assert not isinstance(val_inner, dict), "Un-expected log format"
                            #             log_res_best[f"{best_res_type}/{key_inner}"] = val_inner
                    # if log_res_best is not None and "Results_weighted_avg/val_loss" in log_res and \
                    #         log_res_best["client_summarized_weighted_avg/val_loss"] > \
                    #         log_res["Results_weighted_avg/val_loss"]:
                    #     print("Missing the results of last round, update best results")
                    #     for key, val in log_res.items():
                    #         log_res_best[key.replace("Results", "client_summarized")] = val
                    all_log_res.append(log_res)

            exp_stop_normal = True if " Find new best result" in last_line else exp_stop_normal
            if exp_stop_normal:
                with open(cfg_f_name, 'r') as stream:
                    try:
                        parsed_yaml = yaml.safe_load(stream)
                        # print(parsed_yaml)
                    except yaml.YAMLError as exc:
                        print(exc)

                print(cfg_f_name)
                path_split = cfg_f_name.split("/")

                # e.g., xxx/FedBN/convnet2_0.01_3_bs64_on_femnist/sub_exp_03-14_19:16:51/config.yaml
                if re.match("sub_exp_\d{2}-\d{2}_\d{2}:\d{2}:\d{2}", path_split[-2]):
                    print(path_split[-2])
                    print("will merger the sub_exp_xxx & exp_name")
                    method_name, exp_name = path_split[-4], path_split[-3] + "--" + path_split[-2]
                    dataset_name = path_split[-3].split("_")[-1]
                else:
                    # e.g., xxx/FedBN/convnet2_0.01_3_bs64_on_femnist/config.yaml
                    method_name, exp_name = path_split[-3], path_split[-2]
                    dataset_name = exp_name.split("_")[-1]

                wandb.init(project=args.project, entity=args.user, config=parsed_yaml,
                           group=dataset_name, job_type=method_name, name=method_name + "-" + exp_name,
                           notes=f"{method_name}, {exp_name}",
                           reinit=True)
                wandb.log(log_res_best)
                for log_res in all_log_res:
                    wandb.log(log_res)
            print(cfg_f_name)
            print(log_res_best)
            print("\n")
        wandb.finish()


if __name__ == "__main__":
    main()
