import os

import numpy as np
import transformers
from tqdm import tqdm
from rouge import Rouge

from federatedscope.core.configs.config import global_cfg
from federatedscope.core.cmd_args import parse_args, parse_client_cfg
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.core.data.utils import download_url
from federatedscope.llm.dataloader.dataloader import load_jsonl
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
    fp = os.path.join(init_cfg.data.root, "databricks-dolly-15k.jsonl")
    if not os.path.exists(fp):
        download_url(
            'https://raw.githubusercontent.com/databrickslabs'
            '/dolly/d000e3030970379aabbf6d291f50ffdd3b715b64'
            '/data/databricks-dolly-15k.jsonl', init_cfg.data.root)
        os.rename(os.path.join(init_cfg.data.root, 'test.jsonl'), fp)

    list_data_dict = load_jsonl(fp,
                                instruction='instruction',
                                input='context',
                                output='response',
                                category='category')

    list_data_dict = [
        x for x in list_data_dict if x["category"] == "summarization"
    ]
    answers = []
    for sample in tqdm(list_data_dict):
        input_text = sample['instruction']
        generate_kwargs = dict(max_new_tokens=256, top_p=0.95, temperature=0.8)
        model_answer = fschatbot.generate(input_text, generate_kwargs)

        rougel_cor = rouge_score(model_answer, sample['output'])
        answers.append(rougel_cor)
        if DEBUG:
            print(f'Full input_text:\n{input_text}\n\n')
        print(f'Question: {sample["instruction"]}\n\n'
              f'Answers: {model_answer}\n\n')

        print(f'Num of total question: {len(answers)}, '
              f'Average score: {np.average(answers)}.')


if __name__ == "__main__":
    main()
