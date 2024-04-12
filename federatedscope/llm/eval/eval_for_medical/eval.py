import os

import numpy as np
import transformers
from tqdm import tqdm

from federatedscope.core.configs.config import global_cfg
from federatedscope.core.cmd_args import parse_args, parse_client_cfg
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.core.data.utils import download_url
from federatedscope.llm.dataloader.dataloader import load_jsonl
from federatedscope.llm.misc.fschat import FSChatBot

transformers.logging.set_verbosity(40)

DEBUG = False


def is_correct(model_answer, answer):
    return model_answer == answer


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

    # Get test file
    fp = os.path.join(init_cfg.data.root, "medical_tc_test.jsonl")
    if not os.path.exists(fp):
        download_url(
            'https://federatedscope.oss-cn-beijing.aliyuncs.com/FS-LLM'
            '/medical_tc_test.jsonl', init_cfg.data.root)
        os.rename(os.path.join(init_cfg.data.root, 'test.jsonl'), fp)

    list_data_dict = load_jsonl(fp,
                                instruction='instruction',
                                input='input',
                                output='output',
                                category='output')

    answers = []
    for sample in tqdm(list_data_dict):
        input_text = sample['instruction'] + sample["input"]
        generate_kwargs = dict(max_new_tokens=256, top_p=0.95, temperature=0.8)
        model_answer = fschatbot.generate(input_text, generate_kwargs)

        is_cor = is_correct(model_answer[0], sample['output'][0])
        answers.append(is_cor)
        if DEBUG:
            print(f'Full input_text:\n{input_text}\n\n')
        print(f'Question: {sample["instruction"]}\n\n'
              f'Answers: {sample["output"]}\n\n'
              f'Model Answers: {model_answer}\n\n'
              f'Is correct: {is_cor}\n\n')

        print(f'Num of total question: {len(answers)}, '
              f'correct num: {sum(answers)}, '
              f'correct rate: {float(sum(answers))/len(answers)}.')


if __name__ == "__main__":
    main()
