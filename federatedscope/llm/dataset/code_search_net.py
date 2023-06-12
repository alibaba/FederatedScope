import os
import json
import random

from tqdm import tqdm
from subprocess import call
from federatedscope.llm.dataloader.dataloader import load_jsonl

CSN_FILE_NUM_DICT = {
    'python': {
        'train': 14,
        'val': 1,
        'test': 1,
    },
    'javascript': {
        'train': 5,
        'val': 1,
        'test': 1,
    },
    'java': {
        'train': 16,
        'val': 1,
        'test': 1,
    },
    'ruby': {
        'train': 2,
        'val': 1,
        'test': 1,
    },
    'php': {
        'train': 18,
        'val': 1,
        'test': 1,
    },
    'go': {
        'train': 11,
        'val': 1,
        'test': 1,
    },
}


def generate_eval_files(destination_dir='data'):
    list_data_dict = []
    for language in tqdm(CSN_FILE_NUM_DICT.keys()):
        sub_list_data_dict = []
        for file_index in range(CSN_FILE_NUM_DICT[language]['test']):
            fp = \
                os.path.join(destination_dir, language,
                             'final', 'jsonl', 'test',
                             f'{language}_test_{file_index}.jsonl.gz')
            tmp_list_data_dict = load_jsonl(
                fp,
                instruction='docstring',
                input='code',
                category='language',
                is_gzip=True,
            )
            sub_list_data_dict += tmp_list_data_dict

        # Clear docstring in code
        for sample in sub_list_data_dict:
            if sample['instruction'] in sample['input']:
                sample['input'] = sample['input'].replace(
                    sample['instruction'], "")

        # Build negative samples
        random.shuffle(sub_list_data_dict)
        num_half = len(sub_list_data_dict) // 2
        neg_data_list = sub_list_data_dict[:num_half]
        pos_data_list = sub_list_data_dict[num_half:]

        for i, neg in enumerate(neg_data_list):
            neg['input'] = random.choice(pos_data_list)['input']
            neg['output'] = 0

        for pos in pos_data_list:
            pos['output'] = 1

        sub_list_data_dict = neg_data_list + pos_data_list
        random.shuffle(sub_list_data_dict)

        list_data_dict += sub_list_data_dict

    # Save as a jsonl file
    with open(os.path.join(destination_dir, "csn_test.jsonl"), "w") as f:
        for d in list_data_dict:
            json.dump(d, f)
            f.write("\n")

    return list_data_dict


def download_csn(destination_dir='data'):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    for language in CSN_FILE_NUM_DICT.keys():
        if os.path.exists(os.path.join(destination_dir, f'{language}.zip')):
            continue
        call([
            'wget', 'https://huggingface.co/datasets'
            '/code_search_net/resolve/main/data/{}.zip'.format(language), '-P',
            destination_dir, '-O', '{}.zip'.format(language)
        ])
        call(['unzip', '{}.zip'.format(language)])


if __name__ == '__main__':
    download_csn('data')
    generate_eval_files('data')
