import argparse
import logging
import yaml
import re

from federatedscope.core.auxiliaries.utils import logfile_2_wandb_dict

parser = argparse.ArgumentParser(description='FederatedScope result parsing')
parser.add_argument('--exp_dir',
                    help='path of exp results',
                    required=True,
                    type=str)
parser.add_argument('--project',
                    help='project name used in wandb',
                    required=True,
                    type=str)
parser.add_argument('--user', help='wandb user name', required=True, type=str)
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
            print(f'Process {exp_log_f_list[exp_i]}')
            all_log_res, exp_stop_normal, last_line, log_res_best = logfile_2_wandb_dict(
                exp_log_f)

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
                if re.match("sub_exp_\d{2}-\d{2}_\d{2}:\d{2}:\d{2}",
                            path_split[-2]):
                    print(path_split[-2])
                    print("will merger the sub_exp_xxx & exp_name")
                    method_name, exp_name = path_split[
                        -4], path_split[-3] + "--" + path_split[-2]
                    dataset_name = path_split[-3].split("_")[-1]
                else:
                    # e.g., xxx/FedBN/convnet2_0.01_3_bs64_on_femnist/config.yaml
                    method_name, exp_name = path_split[-3], path_split[-2]
                    dataset_name = exp_name.split("_")[-1]

                wandb.init(project=args.project,
                           entity=args.user,
                           config=parsed_yaml,
                           group=dataset_name,
                           job_type=method_name,
                           name=method_name + "-" + exp_name,
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
