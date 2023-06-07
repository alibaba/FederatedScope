import re
import os
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
COT_PROMPT = ''

QUESTION, ANSWER, CHAIN = [], [], []

QUESTION.append("There are 15 trees in the grove. "
                "Grove workers will plant trees in the grove today. "
                "After they are done, there will be 21 trees. "
                "How many trees did the grove workers plant today?")
CHAIN.append("There are 15 trees originally. "
             "Then there were 21 trees after some more were planted. "
             "So there must have been 21 - 15 = 6.")
ANSWER.append("6")

QUESTION.append(
    "If there are 3 cars in the parking lot and 2 more cars arrive, "
    "how many cars are in the parking lot?")
CHAIN.append("There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.")
ANSWER.append("5")

QUESTION.append(
    "Leah had 32 chocolates and her sister had 42. If they ate 35, "
    "how many pieces do they have left in total?")
CHAIN.append("Originally, Leah had 32 chocolates. "
             "Her sister had 42. So in total they had 32 + 42 = 74. "
             "After eating 35, they had 74 - 35 = 39.")
ANSWER.append("39")

QUESTION.append(
    "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason "
    "has 12 lollipops. How many lollipops did Jason give to Denny?")
CHAIN.append(
    "Jason started with 20 lollipops. Then he had 12 after giving some "
    "to Denny. So he gave Denny 20 - 12 = 8.")
ANSWER.append("8")

QUESTION.append(
    "Shawn has five toys. For Christmas, he got two toys each from his "
    "mom and dad. How many toys does he have now?")
CHAIN.append(
    "Shawn started with 5 toys. If he got 2 toys each from his mom and "
    "dad, then that is 4 more toys. 5 + 4 = 9.")
ANSWER.append("9")

QUESTION.append(
    "There were nine computers in the server room. Five more computers "
    "were installed each day, from monday to thursday. "
    "How many computers are now in the server room?")
CHAIN.append(
    "There were originally 9 computers. For each of 4 days, 5 more "
    "computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29.")
ANSWER.append("29")

QUESTION.append(
    "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On "
    "wednesday, he lost 2 more. "
    "How many golf balls did he have at the end of wednesday?")
CHAIN.append(
    "Michael started with 58 golf balls. After losing 23 on tuesday, "
    "he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls.")
ANSWER.append("33")

QUESTION.append("Olivia has $23. She bought five bagels for $3 each. "
                "How much money does she have left?")
CHAIN.append("Olivia had 23 dollars. "
             "5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. "
             "So she has 23 - 15 dollars left. 23 - 15 is 8.")
ANSWER.append("8")


def extract_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def is_correct(model_completion, answer):
    gt_answer = extract_answer(answer)
    assert gt_answer != INVALID_ANS
    return extract_answer(model_completion) == gt_answer


def build_prompt(input_text, n_shot):
    input_text_prompt = "The following are math questions (with answers).\n\n "
    for i in range(n_shot):
        input_text_prompt += \
            "Q: " + QUESTION[i] + "\nA: " + \
            "The answer is:" + "#### " + ANSWER[i] + ".\n\n"
    input_text_prompt += "Q: " + input_text + "\nA: " + \
                         "The answer is:"
    return input_text_prompt


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
            'https://raw.githubusercontent.com/openai/grade-school'
            '-math/master/grade_school_math/data/test.jsonl',
            init_cfg.data.root)
        os.rename(os.path.join(init_cfg.data.root, 'test.jsonl'), fp)

    list_data_dict = load_jsonl(fp, instruction='question', output='answer')

    answers = []
    for sample in tqdm(list_data_dict):
        input_text = build_prompt(sample['instruction'], N_SHOT)
        model_completion = fschatbot.predict(input_text,
                                             use_history=False,
                                             use_prompt=False)
        is_cor = is_correct(model_completion, sample['output'])
        answers.append(is_cor)
        print(f'Question: {input_text},\n\n'
              f'Answers: {extract_answer(sample["output"])},\n\n'
              f'Model Completion: {model_completion},\n\n'
              f'Is correct: {is_cor}\n\n')

    print(f'Num of total question: {len(answers)}, '
          f'correct num: {sum(answers)}, '
          f'correct rate: {float(sum(answers))/len(answers)}.')


if __name__ == "__main__":
    main()
