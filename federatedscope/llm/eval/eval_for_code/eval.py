import os
import torch
import random
import transformers
import numpy as np
from tqdm import tqdm

from federatedscope.core.configs.config import global_cfg
from federatedscope.core.cmd_args import parse_args, parse_client_cfg
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.core.data.utils import download_url
from federatedscope.llm.dataloader.dataloader import load_json, load_jsonl
from federatedscope.llm.misc.fschat import FSChatBot

transformers.logging.set_verbosity(40)

EVAL_DATA = 'code_search_net'  # code_search_net
N_SHOT = 5
SAMPLES = [{
    "idx": "cosqa-train-0",
    "doc": "python code to write bool value 1",
    "code": "def writeBoolean(self, n):\n        \"\"\"\n"
    "        Writes a Boolean to the stream.\n        "
    "\"\"\"\n        t = TYPE_BOOL_TRUE\n\n        "
    "if n is False:\n            t = TYPE_BOOL_FALSE\n\n"
    "        self.stream.write(t)",
    "label": 0
}, {
    "idx": "cosqa-train-9",
    "doc": "1d array in char datatype in python",
    "code": "def _convert_to_array(array_like, dtype):\n"
    "        \"\"\"\n        "
    "Convert Matrix attributes which are "
    "array-like or buffer to array.\n        "
    "\"\"\"\n        if isinstance(array_like, bytes):\n"
    "            return np.frombuffer(array_like, dtype=dtype)\n"
    "        return np.asarray(array_like, dtype=dtype)",
    "label": 1
}, {
    "idx": "cosqa-train-2",
    "doc": "python colored output to html",
    "code": "def _format_json(data, theme):\n    "
    "\"\"\"Pretty print a dict as a JSON, "
    "with colors if pygments is present.\"\"\"\n    "
    "output = json.dumps(data, indent=2, sort_keys=True)\n\n"
    "    if pygments and sys.stdout.isatty():\n        "
    "style = get_style_by_name(theme)\n        "
    "formatter = Terminal256Formatter(style=style)\n        "
    "return pygments.highlight(output, JsonLexer(), formatter)\n\n"
    "    return output",
    "label": 0
}, {
    "idx": "cosqa-train-18",
    "doc": "python condition non none",
    "code": "def _not(condition=None, **kwargs):\n    \"\"\"\n"
    "    Return the opposite of input condition.\n\n    "
    ":param condition: condition to process.\n\n    "
    ":result: not condition.\n    :rtype: bool\n    "
    "\"\"\"\n\n    result = True\n\n    "
    "if condition is not None:\n        "
    "result = not run(condition, **kwargs)\n\n    "
    "return result",
    "label": 1
}, {
    "idx": "cosqa-train-4",
    "doc": "python column of an array",
    "code": "def _vector_or_scalar(x, type='row'):\n    "
    "\"\"\"Convert an object to either a scalar or "
    "a row or column vector.\"\"\"\n    "
    "if isinstance(x, (list, tuple)):\n        "
    "x = np.array(x)\n    if isinstance(x, np.ndarray):\n"
    "        assert x.ndim == 1\n        "
    "if type == 'column':\n            "
    "x = x[:, None]\n    return x",
    "label": 0
}]


def build_prompt(sample, n_shot):
    input_text_prompt = 'Input: a piece of code and a document\n' \
                        'Output: 0 or 1 score indicating the degree of ' \
                        'matching between the code and the document, ' \
                        'with 0 indicating a mismatch ' \
                        'and 1 indicating a match.\n\n'

    index_list = list(range(len(SAMPLES)))
    random.shuffle(index_list)
    for i in index_list[:n_shot]:
        input_text_prompt += f"Document: {SAMPLES[i]['doc']}\n" \
                             f"Code: {SAMPLES[i]['code']}\n" \
                             f"Score: {SAMPLES[i]['label']}\n\n"
    input_text_prompt += f"Document:{sample['category']}" \
                         f" {sample['instruction']}\n" \
                         f"Code: {sample['input']}\n" \
                         f"Score: "

    return input_text_prompt


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
    tokenizer = fschatbot.tokenizer
    model = fschatbot.model
    device = fschatbot.device

    # Get test file
    if EVAL_DATA == 'cosqa':
        fp = os.path.join(init_cfg.data.root, 'cosqa-dev.json')
        if not os.path.exists(fp):
            download_url(
                'https://github.com/microsoft/CodeXGLUE/raw/'
                'd67dd5c73b9c433307d7df5f9faab2af9f5d1742/'
                'Text-Code/NL-code-search-WebQuery/CoSQA/cosqa-dev.json',
                init_cfg.data.root)
        list_data_dict = load_json(fp,
                                   instruction='doc',
                                   input='code',
                                   output='label')
        for sample in list_data_dict:
            sample['category'] = 'python'
    elif EVAL_DATA == 'code_search_net':
        fp = os.path.join(init_cfg.data.root, 'csn_test.jsonl')
        if not os.path.exists(fp):
            raise FileNotFoundError('Run `python '
                                    'federatedscope/llm/'
                                    'dataset/code_search_net.py` '
                                    'to build test file')
        list_data_dict = load_jsonl(fp,
                                    instruction='instruction',
                                    input='input',
                                    output='output',
                                    category='category')
    else:
        raise ValueError(EVAL_DATA)

    labels, preds, cors = [], [], []
    category = None
    for sample in tqdm(list_data_dict):
        if sample['category'] != category:
            print(f"==============={category}===============\n"
                  f"Num of total question: {len(cors)}\n"
                  f"Average accuracy {np.mean(cors)}\n\n")
            category = sample['category']
            labels, preds, cors = [], [], []

        n_shot = N_SHOT
        input_text = build_prompt(sample, n_shot)
        label = sample['output']

        while len(input_text) > 1024 and n_shot > 0:
            n_shot -= 1
            input_text = build_prompt(sample, n_shot)

        input_ids = \
            tokenizer(input_text, return_tensors="pt",
                      max_length=tokenizer.model_max_length).input_ids.to(
                device)
        logits = model(input_ids=input_ids).logits[0, -1]
        probs = (torch.nn.functional.softmax(
            torch.tensor([
                logits[tokenizer("0").input_ids[-1]],
                logits[tokenizer("1").input_ids[-1]],
            ]).float(),
            dim=0,
        ).detach().cpu().numpy())

        pred = {0: 0, 1: 1}[np.argmax(probs)]

        cor = pred == label

        labels.append(label)
        preds.append(pred)
        cors.append(cor)

    # Print final
    print(f"==============={category}===============\n"
          f"Num of total question: {len(cors)}\n"
          f"Average accuracy {np.mean(cors)}\n\n")


if __name__ == "__main__":
    main()
