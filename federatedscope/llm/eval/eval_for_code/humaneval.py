import os
import torch
import json
import transformers
from transformers import GenerationConfig
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
NUM_ANSWERS_PER_QUESTION = 5


def clean_answer(code):
    """
    Borrow from: https://github.com/FSoft-AI4Code/CodeCapybara
    """
    def pad_spaces(s, num=4):
        n = 0
        while n < len(s) and s[n] == " ":
            n += 1
        if n != num:
            s = " " * num + s[n:]
        return s

    # 1. remove the special char \u00a0
    code = code.replace('\u00a0', '')
    # # 2. remove everything after "\n\n"
    # code = code.split("\n\n")[0]
    # 3. remove everything after the following stop sequences
    # Reference: https://github.com/openai/human-eval
    for stop_seq in ['\nclass', '\ndef', '\n#', '\nif', '\nprint', '\nassert']:
        code = code.split(stop_seq)[0]
    # 4. pad to four space to avoid `unindent` error
    code = pad_spaces(code, 4)
    return code


@torch.no_grad()
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
    out_file = f'{init_cfg.federate.save_to}_humaneval_answer.jsonl'

    # Get test file
    fp = os.path.join(init_cfg.data.root, 'HumanEval.jsonl.gz')
    if not os.path.exists(fp):
        download_url(
            'https://github.com/openai/human-eval/raw/'
            '463c980b59e818ace59f6f9803cd92c749ceae61/'
            'data/HumanEval.jsonl.gz', init_cfg.data.root)
    list_data_dict = load_jsonl(fp,
                                instruction='prompt',
                                input='entry_point',
                                category='task_id',
                                output='test',
                                is_gzip=True)

    answers = []
    for sample in tqdm(list_data_dict):
        input_text = sample['instruction']
        generation_config = GenerationConfig(
            temperature=0.1,
            top_k=40,
            top_p=0.75,
            do_sample=True,
            num_return_sequences=NUM_ANSWERS_PER_QUESTION,
        )
        generate_kwargs = dict(
            generation_config=generation_config,
            max_new_tokens=128,
        )
        try:
            model_completions = fschatbot.generate(input_text, generate_kwargs)
        except torch.cuda.OutOfMemoryError as error:
            print(error)
            model_completions = ['' for _ in range(NUM_ANSWERS_PER_QUESTION)]

        for i, completion in enumerate(model_completions):
            completion = clean_answer(completion)
            answers.append(
                dict(task_id=sample['category'], completion=completion))
            if DEBUG:
                print(f"task_id: {sample['category']},\n"
                      f"completion {i + 1}:\n{completion}\n\n")

    # Save as samples.jsonl for eval pass@k score
    # Run `evaluate_functional_correctness samples.jsonl`
    with open(out_file, 'w') as f:
        for answer in answers:
            json_str = json.dumps(answer)
            f.write(json_str + '\n')


if __name__ == "__main__":
    main()
