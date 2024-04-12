import os
import torch
import numpy as np
import pandas as pd
import json
import transformers

from federatedscope.core.configs.config import global_cfg
from federatedscope.core.cmd_args import parse_args, parse_client_cfg
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.llm.misc.fschat import FSChatBot
from federatedscope.core.data.utils import download_url

# import torch._dynamo
# torch._dynamo.config.suppress_errors = True

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
def eval(subject, model, tokenizer, test_df, device):
    cors = []
    all_probs = []

    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        prompt = format_example(test_df, i, include_answer=False)

        input_ids = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=tokenizer.model_max_length,
        ).input_ids.to(device)

        while input_ids.shape[-1] > 1024:
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

    if not os.path.exists("data/FinEval"):
        download_url(
            "https://federatedscope.oss-cn-beijing.aliyuncs.com/FS"
            "-LLM/FinEval.zip", init_cfg.data.root)
        print("Please unzip the file and rerun")
        return

    data_dir = os.path.join(init_cfg.data.root, "FinEval")
    eval_dir = "finance_eval_result"

    subjects = sorted([
        f.split("_dev.csv")[0]
        for f in os.listdir(os.path.join(data_dir, "dev")) if "_dev.csv" in f
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

    for subject in subjects:
        test_df = pd.read_csv(os.path.join(data_dir, "dev",
                                           subject + "_dev.csv"),
                              header=None)
        test_df = test_df.iloc[:, 1:7]

        cors, acc, probs = eval(subject, model, tokenizer, test_df, device)
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
