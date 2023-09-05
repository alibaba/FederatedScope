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
from federatedscope.llm.dataloader.dataloader import load_jsonl
from federatedscope.core.data.utils import download_url
from federatedscope.llm.misc.fschat import FSChatBot

transformers.logging.set_verbosity(40)

DEBUG = False
NUM_ANSWERS_PER_QUESTION = 5
LANGUAGES = ['cpp', 'go', 'java', 'js', 'python']
LANGUAGE_TAG = {
    "cpp": "// language: C++",
    "python": "# language: Python",
    "java": "// language: Java",
    "js": "// language: JavaScript",
    "go": "// language: Go",
}


def clean_answer(code, language_type=None):
    """
    Cleans up the generated code.
    Borrow from: https://github.com/THUDM/CodeGeeX/blob/main/codegeex
    /benchmark/utils.py
    """
    code = code.replace('\u00a0', '')
    if language_type.lower() == "python":
        end_words = ["\ndef", "\nclass", "\nif", "\n#", "\nprint", "\nassert"]
        for w in end_words:
            if w in code:
                code = code[:code.rfind(w)]
    elif language_type.lower() == "java":
        main_pos = code.find("public static void main")
        if main_pos != -1:
            code = code[:main_pos] + '}'
        if '}' in code:
            code = code[:code.rfind('}')] + '}'
        if code.count('{') + 1 == code.count('}'):
            code += "\n}"
    elif language_type.lower() == "go":
        end_words = ["\n//", "\nfunc main("]
        for w in end_words:
            if w in code:
                code = code[:code.rfind(w)]
        if '}' in code:
            code = code[:code.rfind('}')] + '}'
    elif language_type.lower() == "cpp":
        if '}' in code:
            code = code[:code.rfind('}')] + '}'
    elif language_type.lower() == "js":
        if '}' in code:
            code = code[:code.rfind('}')] + '}'
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

    for lang in LANGUAGES:
        out_file = \
            f'{init_cfg.federate.save_to}_humanevalx_{lang}_answer.jsonl'

        # Get test file
        fp = os.path.join(init_cfg.data.root, f'humaneval_{lang}.jsonl.gz')
        if not os.path.exists(fp):
            download_url(
                'https://github.com/THUDM/CodeGeeX/raw'
                '/e64e88e40a73358bb4ad60ef24114355e7141880/codegeex'
                f'/benchmark/humaneval-x/{lang}/data/humaneval_'
                f'{lang}.jsonl.gz', init_cfg.data.root)
        list_data_dict = load_jsonl(fp,
                                    instruction='prompt',
                                    category='task_id',
                                    is_gzip=True)

        answers = []
        for sample in tqdm(list_data_dict):
            input_text = LANGUAGE_TAG[lang] + '\n' + sample['instruction']
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
                model_completions = fschatbot.generate(input_text,
                                                       generate_kwargs)
            except torch.cuda.OutOfMemoryError as error:
                print(error)
                model_completions = [
                    '' for _ in range(NUM_ANSWERS_PER_QUESTION)
                ]

            for i, completion in enumerate(model_completions):
                completion = clean_answer(completion, language_type=lang)
                answers.append(
                    dict(task_id=sample['category'], generation=completion))
                if DEBUG:
                    print(f"task_id: {sample['category']},\n"
                          f"generation {i + 1}:\n{completion}\n\n")

        # Save as samples.jsonl for eval pass@k score
        # Run `evaluate_functional_correctness samples.jsonl`
        with open(out_file, 'w') as f:
            for answer in answers:
                json_str = json.dumps(answer)
                f.write(json_str + '\n')


if __name__ == "__main__":
    main()
