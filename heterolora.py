from tqdm import tqdm
from scipy.stats import norm
from transformers import  AutoModelForCausalLM, AutoTokenizer
from peft import (
    LoraConfig,
    get_peft_model,
)
from fed_utils import (FedAvg, client_selection, seed_torch, GeneralClient, distribute_weight_fast, modify_adapter,
                       load_weight_SLoRA, truncate)

import datasets
from datasets import load_dataset
from utils.prompter import Prompter
import socket
import copy

datasets.utils.logging.set_verbosity_error()

import numpy as np
import random
import os
import torch
import logging
import argparse
os.environ["WANDB_MODE"]="disabled"


def read_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('--global_model', default='data_juicer', type=str, help='ifle path to the LLaMA model')
    parser.add_argument('--data_path', default='./data', type=str,
                        help='file path to data')
    parser.add_argument('--cache_dir', default=None, type=str,
                        help='file path for caching data')
    parser.add_argument('--output_dir', default=None, type=str,
                        help='output directory to store model and experiment result')
    parser.add_argument('--session_name', default='test', type=str,
                        help='name for your experiment')
    parser.add_argument('--seed', default=42, type=int,
                        help='random seed')
    parser.add_argument('--resume_epoch', default=None, type=int,
                        help='continue training from an existing experiment, specifying which comm round to resumt')

    ## FL parameters
    parser.add_argument('--aggregation', default='homo', type=str,
                        help='aggregation method',
                        choices=['homo', 'random', 'heavy_tail', 'heavy_tail_strong', 'normal'])
    parser.add_argument('--baseline', default='fedavg', type=str,
                        help='type of FL baselines to choose', choices=['fedavg', 'slora', 'fedit'])
    parser.add_argument('--client_selection_frac', default=0.05, type=float,
                        help='ratio of how many clients participate in each round')
    parser.add_argument('--num_clients', default=1613, type=int,
                        help='total number of clients')
    parser.add_argument('--num_communication_rounds', default=50, type=int,
                        help='total number of communication rounds')
    parser.add_argument('--R_1', default=5, type=int,
                        help='Parameter for SLoRA. Total number of rounds for stage 1 sparse finetuning.')
    parser.add_argument('--early_stop', default=True, type=bool,
                        help='Early stop for FL training. If True, will apply early stop.')
    parser.add_argument('--patience', default=3, type=int,
                        help='Early stop patience.')
    ## Local training parameters
    parser.add_argument('--local_batch_size', default=4, type=int,
                        help='local_batch_size')
    parser.add_argument('--local_micro_batch_size', default=2, type=int,
                        help='local_micro_batch_size')
    parser.add_argument('--local_num_epochs', default=1, type=int,
                        help='local epochs for local client training')
    parser.add_argument('--local_learning_rate', default=1e-6, type=float,
                        help='local training rate for local client training')
    parser.add_argument('--cutoff_len', default=512, type=int,
                        help='cut off len for tokenizing text')
    parser.add_argument('--warmup', default=0, type=int,
                        help='warm up steps for local training')
    parser.add_argument('--lr_decay', default=True, type=bool,
                        help='Learning rate decay. If true, will divide learning rate by 2 after 15-th comm round')
    parser.add_argument('--train_on_inputs', default=True, type=bool,
                        help='Whether training on input text')
    parser.add_argument('--group_by_length', default=False, type=bool,
                        help='')
    parser.add_argument('--prompt_template_name', default='alpaca', type=str,
                        help='template to generate prompt')

    ## LoRA Parameters
    parser.add_argument('--lora_r', default=8, type=int,
                        help='LoRA rank')
    parser.add_argument('--lora_alpha', default=16, type=int,
                        help='LoRA alpha')
    parser.add_argument('--lora_dropout', default=0.05, type=float,
                        help='LoRA dropout')
    parser.add_argument('--lora_target_modules', default=['q_proj', 'v_proj', 'k_proj', 'o_proj',
                                                          'gate_proj', 'down_proj', 'up_proj'
                                                          ], type=list,
                        help='lora_target_modules')
    parser.add_argument('--handle_alpha', default=True, type=bool,
                        help='heteroLoRA parameter, scale LoRA weights based on alpha')
    parser.add_argument('--lambd', default=0.001, type=float,
                        help='heteroLoRA parameter, regularization scale')

    args = parser.parse_args()
    return args


# setting up model and tokenizer
def model_and_tokenizer(global_model, device_map='auto'):
    """
        setting up model and tokenizer
    """
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

    model = AutoModelForCausalLM.from_pretrained(
        global_model,
        device_map=device_map, trust_remote_code=True)

    tokenizer = AutoTokenizer.from_pretrained(global_model)
    tokenizer.pad_token_id = (
        0
    )
    tokenizer.padding_side = "left"
    return model, tokenizer


# get each client's unique LoRA configuration based on the "aggregation" parameter
def get_peft(config_types, num_clients, strategy=None):
    """
        get each client's unique LoRA configuration based on the "aggregation" parameter
    """
    if strategy == 'homo':
        return
    else:
        # random select lora type for clients
        if strategy == 'random':
            config_local = {'alpha': 16, 'lora_dropout': 0.05}
            for i in range(num_clients):
                type = 'Type_' + str(np.random.randint(0, 4))
                config_local['Client_' + str(i)] = config_types[type]
        elif strategy == 'heavy_tail':
            config_local = {'alpha': 16, 'lora_dropout': 0.05}
            for i in range(num_clients):
                rand_num = random.random()  # Generate a random float between 0 and 1
                if rand_num < 0.80:
                    type = 'Type_0'
                elif rand_num < 0.90:
                    type = 'Type_1'
                elif rand_num < 0.95:
                    type = 'Type_2'
                else:
                    type = 'Type_3'
                config_local['Client_' + str(i)] = config_types[type]
        elif strategy == 'heavy_tail_strong':
            config_local = {'alpha': 16, 'lora_dropout': 0.05}
            for i in range(num_clients):
                rand_num = random.random()  # Generate a random float between 0 and 1
                if rand_num < 0.80:
                    type = 'Type_1'
                elif rand_num < 0.90:
                    type = 'Type_2'
                elif rand_num < 0.95:
                    type = 'Type_3'
                else:
                    type = 'Type_0'
                config_local['Client_' + str(i)] = config_types[type]
        elif strategy == 'normal':
            config_local = {'alpha': 16, 'lora_dropout': 0.05}
            positions = np.array([0, 3, 2, 1])
            mu = 1.5
            sigma = 0.7
            probabilities = norm.pdf(positions, mu, sigma)
            probabilities /= probabilities.sum()
            for i in range(num_clients):
                selected_var = np.random.choice(positions, p=probabilities)
                type = 'Type_' + str(selected_var)
                config_local['Client_' + str(i)] = config_types[type]

        return config_local


def local_client_load_weight(args, model, epoch, global_params=None):
    """
    load local client's weight
    """
    if args.aggregation == 'homo':
        if not (args.baseline == 'slora' and args.R_1 == epoch):
            _ = model.load_state_dict(global_params, strict=False)
    else:
        if args.baseline == 'slora':
            if epoch < args.R_1:
                local_weight = global_params
            elif epoch > args.R_1:
                local_weight = {}
                for name, param in model.named_parameters():
                    if name in global_params.keys():
                        rank = min(param.data.shape[0], param.data.shape[1])
                        if args.handle_alpha:
                            merge_rate = 16 / rank
                        else:
                            merge_rate = 1
                        if 'lora_A' in name:
                            local_weight[name] = copy.deepcopy(global_params[name][:rank, :]) / np.sqrt(merge_rate)
                        elif 'lora_B' in name:
                            local_weight[name] = copy.deepcopy(global_params[name][:, :rank]) / np.sqrt(merge_rate)
            _ = model.load_state_dict(local_weight, strict=False)
        else:
            local_weight = {}
            for name, param in model.named_parameters():
                if name in global_params.keys():
                    rank = min(param.data.shape[0], param.data.shape[1])
                    if args.handle_alpha:
                        merge_rate = 16 / rank
                    else:
                        merge_rate = 1
                    if 'lora_A' in name:
                        local_weight[name] = copy.deepcopy(global_params[name][:rank, :]) / np.sqrt(merge_rate)
                    elif 'lora_B' in name:
                        local_weight[name] = copy.deepcopy(global_params[name][:, :rank]) / np.sqrt(merge_rate)
            _ = model.load_state_dict(local_weight, strict=False)


def local_client_modify_layer(args, epoch, config_local, model, client_id):
    """
    Modify local client's LoRA layers based on local config
    """
    if args.aggregation != 'homo':
        if args.baseline == 'slora' and epoch >= args.R_1:
            local_lora_config = config_local['Client_' + str(client_id)]
            modify_adapter(model, 'local', modify_module_rank=local_lora_config,
                           lora_alpha=config_local['alpha'], lora_dropout=config_local['lora_dropout'],
                           init_lora_weights=True)
        else:
            local_lora_config = config_local['Client_' + str(client_id)]
            modify_adapter(model, 'local', modify_module_rank=local_lora_config,
                           lora_alpha=config_local['alpha'], lora_dropout=config_local['lora_dropout'],
                           init_lora_weights=True)


def resume(args, data_path, output_dir):
    """
    resume experiment from an existing study
    """
    selected_clients_set = client_selection(args.num_clients, args.client_selection_frac,
                                            seed=args.seed, other_info=args.resume_epoch - 1)
    local_dataset_len_dict = []
    for client_id in tqdm(selected_clients_set):
        train_path = data_path + '/local_training_' + str(client_id) + '.json'
        train_data = load_dataset("json", data_files=train_path, cache_dir=args.cache_dir)
        local_dataset_len_dict[client_id] = len(train_data['train'])
    if args.baseline == 'slora':
        if args.resume_epoch - 1 < args.R_1 or args.aggregation == 'fedavg':
            global_params = FedAvg(selected_clients_set,
                                   output_dir,
                                   local_dataset_len_dict,
                                   args.resume_epoch - 1,
                                   )
        else:
            global_params = truncate(selected_clients_set,
                                     output_dir,
                                     local_dataset_len_dict,
                                     args.resume_epoch - 1,
                                     )
    else:
        if args.aggregation == 'homo':
            global_params = FedAvg(selected_clients_set,
                                   output_dir,
                                   local_dataset_len_dict,
                                   args.resume_epoch - 1,
                                   )
        else:
            global_params = truncate(selected_clients_set,
                                     output_dir,
                                     local_dataset_len_dict,
                                     args.resume_epoch - 1,
                                     )
    return global_params


def get_density(args, config_local, client_id, config):
    """
    get sparsity for slora sparse finetuning stage 1
    """
    if args.aggregation == 'homo':
        density = 0.0012
    else:
        if config_local['Client_' + str(client_id)] == config['Type_0']:
            density = 0.0012
        if config_local['Client_' + str(client_id)] == config['Type_1']:
            density = 0.1222
        if config_local['Client_' + str(client_id)] == config['Type_2']:
            density = 0.0822
        if config_local['Client_' + str(client_id)] == config['Type_3']:
            density = 0.0246
    return density


# training for FL setting
def FL_training(model, tokenizer, prompter, data_path, output_dir, args, config_local, config=None, config_types=None):
    logging.info("The process of federated instruction-tuning has started..")
    previously_selected_clients_set = set()
    output_dir = os.path.join(output_dir, str(args.num_clients))

    local_dataset_len_dict = dict()
    best_rouge_L = 0
    patience = args.patience
    current_count = 0
    if args.resume_epoch:
        global_params = resume(args, data_path, output_dir)
        start_epoch = args.resume_epoch
    else:
        start_epoch = 0

    optim = 'sgd' if args.baseline == 'fedavg' else 'adamw_torch'
    for epoch in tqdm(range(start_epoch, args.num_communication_rounds)):
        local_train_results = 0
        local_eval_results = 0
        local_eval_rouge_1 = 0
        local_eval_rouge_L = 0
        total_data_num = 0
        logging.info("\In Epoch " + str(epoch))
        logging.info("\nConducting the client selection")

        # select participating clients
        selected_clients_set = client_selection(args.num_clients, args.client_selection_frac,
                                                seed=args.seed, other_info=epoch)
        if epoch == 15 and args.lr_decay == True:
            args.local_learning_rate = args.local_learning_rate / 2

        if args.baseline == 'slora' and epoch == args.R_1:
            model, tokenizer = model_and_tokenizer(global_model=args.global_model, device_map='auto')
            get_peft_model(model, config, adapter_name='local')

        # training for each client
        for k, client_id in enumerate(selected_clients_set):
            train_path = data_path + '/local_training_' + str(client_id) + '.json'
            train_data = load_dataset("json", data_files=train_path, cache_dir=args.cache_dir)
            local_dataset_len_dict[client_id] = len(train_data['train'])

            total_data_num += local_dataset_len_dict[client_id]

            if args.baseline == 'slora' and args.R_1 == epoch:
                local_weight = load_weight_SLoRA(global_params, model)
                _ = model.load_state_dict(local_weight, strict=False)

            # if not fedavg, modify client LoRA config based on previous assignment

            if epoch > 0:
                local_client_load_weight(args, model, epoch, global_params=global_params)

            client = GeneralClient(client_id, model, tokenizer, prompter, data_path, output_dir,
                                   cache_dir=args.cache_dir,
                                   hetero_lora=True, optim=optim)

            logging.info("\nPreparing the local dataset and trainer for Client_{}".format(client_id))
            client.preprare_local_dataset()

            local_eval_result = client.test(epoch, args.local_micro_batch_size)
            local_eval_results += float(local_eval_result['eval_loss']) * local_dataset_len_dict[client_id]
            local_eval_rouge_1 += float(local_eval_result['eval_rouge1']) * local_dataset_len_dict[client_id]
            local_eval_rouge_L += float(local_eval_result['eval_rougeL']) * local_dataset_len_dict[client_id]

            logging.info("Initiating the local training of Client_{}".format(client_id))

            if args.baseline == 'slora' and epoch < args.R_1:
                density = get_density(args, config_local, client_id, config_types)
                sparse = True
                client.get_sparse(model, args.local_learning_rate, args.local_micro_batch_size, args.warmup, density)
            else:
                sparse = False

            client.build_local_trainer(tokenizer,
                                       args.local_micro_batch_size,
                                       args.local_batch_size // args.local_micro_batch_size,
                                       args.local_num_epochs,
                                       args.local_learning_rate,
                                       args.group_by_length,
                                       args.warmup,
                                       lambd = args.lambd)
            client.initiate_local_training(sparse)

            logging.info("Local training starts ... ")
            local_train_result = client.train()
            local_train_results += float(local_train_result['eval_loss']) * local_dataset_len_dict[client_id]

            logging.info("\nTerminating the local training of Client_{}".format(client_id))
            model, local_dataset_len_dict, previously_selected_clients_set, last_client_id = client.terminate_local_training(
                epoch, local_dataset_len_dict, previously_selected_clients_set)
            del client

            logging.info("Collecting the weights of clients and performing aggregation")
        if args.aggregation == 'homo':
            global_params = FedAvg(selected_clients_set,
                                   output_dir,
                                   local_dataset_len_dict,
                                   epoch,
                                   )
            torch.save(global_params, os.path.join(output_dir, "adapter_model.bin"))
        else:
            global_params = truncate(selected_clients_set,
                                     output_dir,
                                     local_dataset_len_dict,
                                     epoch,
                                     )
            torch.save(global_params, os.path.join(output_dir, "adapter_model.bin"))
            global_params = distribute_weight_fast(global_params, config_local)

        global_eval_rouge_L = local_eval_rouge_L / total_data_num

        ### early stop
        if args.early_stop:
            if best_rouge_L < global_eval_rouge_L:
                best_rouge_L = global_eval_rouge_L
                best_round = epoch
                current_count = 0
            else:
                current_count += 1
            if current_count > patience:
                logging.info("Best round is", best_round, "with test_rouge_L", best_rouge_L)
                return


def main():
    args = read_options()
    seed_torch(args.seed)
    if not os.path.exists(args.session_name):
        os.makedirs(args.session_name)
    if args.output_dir:
        if not os.path.exists(os.path.join(args.output_dir, args.session_name)):
            os.makedirs(os.path.join(args.output_dir, args.session_name))
        logging.basicConfig(filename=os.path.join(args.output_dir, args.session_name, '../result.log'),
                            level=logging.INFO,
                            format='%(message)s')
    else:
        logging.basicConfig(filename=os.path.join(args.session_name, '../result.log'),
                            level=logging.INFO,
                            format='%(message)s')
    logging.info("Initial training parameters %s", args)
    print(args)

    logging.info(str(socket.gethostbyname(socket.gethostname())))

    data_path = os.path.join(args.data_path, str(args.num_clients))
    if args.output_dir:
        output_dir = os.path.join(args.output_dir, args.session_name, args.aggregation)
    else:
        output_dir = os.path.join(args.session_name, args.aggregation)

    # set up the global model & toknizer
    model, tokenizer = model_and_tokenizer(global_model=args.global_model, device_map='auto')

    prompter = Prompter(args.prompt_template_name)

    config_types = {
        'Type_0': {'q_proj': 8, 'v_proj': 8, 'k_proj': 8, 'o_proj': 8, 'gate_proj': 8, 'down_proj': 8, 'up_proj': 8},
        'Type_1': {'q_proj': 200, 'v_proj': 200, 'k_proj': 200, 'o_proj': 200, 'gate_proj': 200, 'down_proj': 200,
                   'up_proj': 200},
        'Type_2': {'q_proj': 30, 'v_proj': 30, 'k_proj': 30, 'o_proj': 30, 'gate_proj': 200, 'down_proj': 200,
                   'up_proj': 200},
        'Type_3': {'q_proj': 30, 'v_proj': 30, 'k_proj': 30, 'o_proj': 30, 'gate_proj': 30, 'down_proj': 30,
                   'up_proj': 30}, }
    config_local = get_peft(config_types, num_clients=args.num_clients, strategy=args.aggregation)

    logging.info(config_local)

    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    if args.baseline != 'slora':
        model = get_peft_model(model, config, adapter_name='local')
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    FL_training(model, tokenizer, prompter, data_path, output_dir, args, config_local=config_local, config=config, config_types=config_types)


if __name__ == "__main__":
    main()
