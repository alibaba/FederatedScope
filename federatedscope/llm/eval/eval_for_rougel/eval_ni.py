import json
import os
import random

import numpy as np
import transformers
from tqdm import tqdm
from rouge import Rouge

from federatedscope.core.configs.config import global_cfg
from federatedscope.core.cmd_args import parse_args, parse_client_cfg
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.core.data.utils import download_url
from federatedscope.llm.misc.fschat import FSChatBot

transformers.logging.set_verbosity(40)

DEBUG = False

rouge = Rouge()


def rouge_score(hyps, refs):
    try:
        rouge_score = rouge.get_scores(hyps, refs)[0]['rouge-l']['f']
    except ValueError:
        return 0.0
    return rouge_score


def load_data(file_path,
              instruction='instruction',
              input='input',
              output='output',
              category='category'):

    # Format: [{'instruction': ..., 'input': ..., 'output':...}]
    with open(file_path, 'r', encoding="utf-8") as f:
        list_data_dict = json.load(f)

    # Replace key
    new_list_data_dict = []
    list_data_dict = list_data_dict["Instances"]

    num_samples = int(len(list_data_dict) * 0.02)
    chosen_list_data_dict = random.sample(list_data_dict, num_samples)

    for item in chosen_list_data_dict:
        new_item = dict(
            instruction=item[instruction] if instruction in item else None,
            input=item[input] if input in item else None,
            output=item[output][0] if output in item else None,
            category=item[category] if category in item else None)
        new_list_data_dict.append(new_item)
    return new_list_data_dict


def main():
    init_cfg = global_cfg.clone()
    args = parse_args()

    if args.cfg_file:
        init_cfg.merge_from_file(args.cfg_file)
    cfg_opt, client_cfg_opt = parse_client_cfg(args.opts)
    init_cfg.merge_from_list(cfg_opt)

    update_logger(init_cfg, clear_before_add=True)
    setup_seed(init_cfg.seed)

    # load your finetuned model (saved as xxx.ckpt)
    #    in yaml file federate.save_to
    fschatbot = FSChatBot(init_cfg)

    test_tasks_fp = os.path.join(
        init_cfg.data.root,
        "natural-instructions-2.8/splits/xlingual/test_tasks.txt")

    if not os.path.exists(test_tasks_fp):
        download_url(
            'https://github.com/allenai/natural-instructions/archive/refs'
            '/tags/v2.8.zip', init_cfg.data.root)
        print("Please unzip the data, and rerun.")
        return

    test_tasks = []
    with open(test_tasks_fp, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            test_tasks.append(line.strip())

    list_data_dict = []
    for task in test_tasks:
        fp = os.path.join(init_cfg.data.root, "natural-instructions-2.8/tasks",
                          task + ".json")
        list_data_dict.extend(load_data(fp))

    answers = []
    for sample in tqdm(list_data_dict):
        input_text = sample['input']
        generate_kwargs = dict(max_new_tokens=256, top_p=0.95, temperature=0.8)
        model_answer = fschatbot.generate(input_text, generate_kwargs)

        rougel_cor = rouge_score(model_answer, sample['output'])
        answers.append(rougel_cor)
        if DEBUG:
            print(f'Full input_text:\n{input_text}\n\n')
        print(f'Question: {sample["input"]}\n\n'
              f'Answers: {model_answer}\n\n')

        print(f'Num of total question: {len(answers)}, '
              f'Average score: {np.average(answers)}.')


if __name__ == "__main__":
    main()
