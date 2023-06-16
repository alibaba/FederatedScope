# Ref: https://github.com/kojima-takeshi188/zero_shot_cot

import re
import os
import random
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

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

N_SHOT = 8
COT_FLAG = True
DEBUG = False
ANSWER_TRIGGER = "The answer is"


def extract_answer_from_output(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def is_correct(model_answer, answer):
    gt_answer = extract_answer_from_output(answer)
    assert gt_answer != INVALID_ANS
    return model_answer == gt_answer


def create_demo_text(n_shot=8, cot_flag=True):
    question, chain, answer = [], [], []
    question.append("There are 15 trees in the grove. "
                    "Grove workers will plant trees in the grove today. "
                    "After they are done, there will be 21 trees. "
                    "How many trees did the grove workers plant today?")
    chain.append("There are 15 trees originally. "
                 "Then there were 21 trees after some more were planted. "
                 "So there must have been 21 - 15 = 6.")
    answer.append("6")

    question.append(
        "If there are 3 cars in the parking lot and 2 more cars arrive, "
        "how many cars are in the parking lot?")
    chain.append("There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.")
    answer.append("5")

    question.append(
        "Leah had 32 chocolates and her sister had 42. If they ate 35, "
        "how many pieces do they have left in total?")
    chain.append("Originally, Leah had 32 chocolates. "
                 "Her sister had 42. So in total they had 32 + 42 = 74. "
                 "After eating 35, they had 74 - 35 = 39.")
    answer.append("39")

    question.append(
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason "
        "has 12 lollipops. How many lollipops did Jason give to Denny?")
    chain.append(
        "Jason started with 20 lollipops. Then he had 12 after giving some "
        "to Denny. So he gave Denny 20 - 12 = 8.")
    answer.append("8")

    question.append(
        "Shawn has five toys. For Christmas, he got two toys each from his "
        "mom and dad. How many toys does he have now?")
    chain.append(
        "Shawn started with 5 toys. If he got 2 toys each from his mom and "
        "dad, then that is 4 more toys. 5 + 4 = 9.")
    answer.append("9")

    question.append(
        "There were nine computers in the server room. Five more computers "
        "were installed each day, from monday to thursday. "
        "How many computers are now in the server room?")
    chain.append(
        "There were originally 9 computers. For each of 4 days, 5 more "
        "computers were added. So 5 * 4 = 20 computers were added. "
        "9 + 20 is 29.")
    answer.append("29")

    question.append(
        "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On "
        "wednesday, he lost 2 more. "
        "How many golf balls did he have at the end of wednesday?")
    chain.append(
        "Michael started with 58 golf balls. After losing 23 on tuesday, "
        "he had 58 - 23 = 35. After losing 2 more, "
        "he had 35 - 2 = 33 golf balls.")
    answer.append("33")

    question.append("Olivia has $23. She bought five bagels for $3 each. "
                    "How much money does she have left?")
    chain.append("Olivia had 23 dollars. "
                 "5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. "
                 "So she has 23 - 15 dollars left. 23 - 15 is 8.")
    answer.append("8")

    # randomize order of the examples ...
    index_list = list(range(len(question)))
    random.shuffle(index_list)

    # Concatenate demonstration examples ...
    demo_text = ""
    for i in index_list[:n_shot]:
        if cot_flag:
            demo_text += "Q: " + question[i] + "\nA: " + chain[i] + " " + \
                         ANSWER_TRIGGER + " " + answer[i] + ".\n\n"
        else:
            demo_text += "Question: " + question[i] + "\nAnswer: " + \
                         ANSWER_TRIGGER + " " + answer[i] + ".\n\n"
    return demo_text


def build_prompt(input_text, n_shot, cot_flag):
    demo = create_demo_text(n_shot, cot_flag)
    input_text_prompt = demo + "Q: " + input_text + "\n" + "A:"
    return input_text_prompt


def clean_answer(model_pred):
    model_pred = model_pred.lower()
    preds = model_pred.split(ANSWER_TRIGGER.lower())
    answer_flag = True if len(preds) > 1 else False
    if answer_flag:
        # Pick first answer with flag
        pred = preds[1]
    else:
        # Pick last number without flag
        pred = preds[-1]

    pred = pred.replace(",", "")
    pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]

    if len(pred) == 0:
        return INVALID_ANS

    if answer_flag:
        # choose the first element in list
        pred = pred[0]
    else:
        # choose the last element in list
        pred = pred[-1]

    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred[-1] == ".":
        pred = pred[:-1]

    return pred


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
    fp = os.path.join(init_cfg.data.root, 'gsm8k_test.jsonl')
    if not os.path.exists(fp):
        download_url(
            'https://raw.githubusercontent.com/openai/'
            'grade-school-math/2909d34ef28520753df82a2234c357259d254aa8/'
            'grade_school_math/data/test.jsonl', init_cfg.data.root)
        os.rename(os.path.join(init_cfg.data.root, 'test.jsonl'), fp)

    list_data_dict = load_jsonl(fp, instruction='question', output='answer')

    answers = []
    for sample in tqdm(list_data_dict):
        input_text = build_prompt(sample['instruction'], N_SHOT, COT_FLAG)
        generate_kwargs = dict(max_new_tokens=256, top_p=0.95, temperature=0.8)
        model_completion = fschatbot.generate(input_text, generate_kwargs)
        model_answer = clean_answer(model_completion)
        is_cor = is_correct(model_answer, sample['output'])
        answers.append(is_cor)
        if DEBUG:
            print(f'Full input_text:\n{input_text}\n\n')
        print(f'Question: {sample["instruction"]}\n\n'
              f'Answers: {extract_answer_from_output(sample["output"])}\n\n'
              f'Model Answers: {model_answer}\n\n'
              f'Model Completion: {model_completion}\n\n'
              f'Is correct: {is_cor}\n\n')

        print(f'Num of total question: {len(answers)}, '
              f'correct num: {sum(answers)}, '
              f'correct rate: {float(sum(answers))/len(answers)}.')


if __name__ == "__main__":
    main()
