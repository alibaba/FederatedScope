# ref: https://github.com/hendrycks/test/blob/master/evaluate_flan.py
import os
import torch
import numpy as np
import pandas as pd
from federatedscope.llm.eval.eval_for_mmlu.categories import \
     subcategories, categories
import json
import transformers

from federatedscope.core.configs.config import global_cfg
from federatedscope.core.cmd_args import parse_args, parse_client_cfg
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.llm.misc.fschat import FSChatBot
from federatedscope.core.data.utils import download_url
import tarfile

transformers.logging.set_verbosity(40)

choices = ["A", "B", "C", "D"]


def format_subject(subject):
    ll = subject.split("_")
    s = ""
    for entry in ll:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice \
        questions (with answers) about {}.\n\n".format(format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


@torch.no_grad()
def eval(subject, model, tokenizer, dev_df, test_df, device):
    cors = []
    all_probs = []
    answers = choices[:test_df.shape[1] - 2]

    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        k = 5
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        input_ids = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=tokenizer.model_max_length,
        ).input_ids.to(device)

        while input_ids.shape[-1] > 1024:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end
            input_ids = tokenizer(prompt,
                                  return_tensors="pt").input_ids.to(device)

        label = test_df.iloc[i, test_df.shape[1] - 1]

        logits = model(input_ids=input_ids).logits[0, -1]

        probs = (torch.nn.functional.softmax(
            torch.tensor([
                logits[tokenizer("A").input_ids[-1]],
                logits[tokenizer("B").input_ids[-1]],
                logits[tokenizer("C").input_ids[-1]],
                logits[tokenizer("D").input_ids[-1]],
            ]).float(),
            dim=0,
        ).detach().cpu().numpy())
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]

        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs


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

    if not os.path.exists("data/mmlu"):
        download_url("https://people.eecs.berkeley.edu/~hendrycks/data.tar",
                     init_cfg.data.root)
        t = tarfile.open("data/data.tar", "r:")
        os.makedirs("data/mmlu/")
        t.extractall(path="data/mmlu/")
        t.close()

    data_dir = os.path.join(init_cfg.data.root, "mmlu/data")
    eval_dir = "eval_result"

    subjects = sorted([
        f.split("_test.csv")[0]
        for f in os.listdir(os.path.join(data_dir, "test")) if "_test.csv" in f
    ])

    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)
    if not os.path.exists(
            os.path.join(eval_dir, "results_{}".format(
                init_cfg.federate.save_to))):
        os.makedirs(
            os.path.join(eval_dir,
                         "results_{}".format(init_cfg.federate.save_to)))

    all_cors = []
    subcat_cors = {
        subcat: []
        for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}

    for subject in subjects:
        dev_df = pd.read_csv(os.path.join(data_dir, "dev",
                                          subject + "_dev.csv"),
                             header=None)[:5]
        test_df = pd.read_csv(os.path.join(data_dir, "test",
                                           subject + "_test.csv"),
                              header=None)

        cors, acc, probs = eval(subject, model, tokenizer, dev_df, test_df,
                                device)
        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

        test_df["{}_correct".format(init_cfg.federate.save_to)] = cors
        for j in range(probs.shape[1]):
            choice = choices[j]
            test_df["{}_choice{}_probs".format(init_cfg.federate.save_to,
                                               choice)] = probs[:, j]
        test_df.to_csv(
            os.path.join(eval_dir,
                         "results_{}".format(init_cfg.federate.save_to),
                         "{}.csv".format(subject)),
            index=None,
        )

    results = {"subcategories": {}, "categories": {}}
    for subcat in subcat_cors:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))

    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        results["categories"][cat] = cat_acc
        print("Average accuracy {:.3f} - {}".format(cat_acc, cat))
    weighted_acc = np.mean(np.concatenate(all_cors))
    results["weighted_accuracy"] = weighted_acc
    print("Average accuracy: {:.3f}".format(weighted_acc))

    results_file = os.path.join(
        eval_dir, "accuracies_{}.json".format(
            init_cfg.federate.save_to.replace("/", "_")))
    with open(results_file, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
