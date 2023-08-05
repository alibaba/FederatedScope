# ---------------------------------------------------------------------- #
# A preprocess script for JSON file all_data.json to json with images
# To get raw all_data.json, see:
# https://github.com/TalwalkarLab/leaf/tree/master/data/celeba
# ---------------------------------------------------------------------- #

import json
import math
import numpy as np
import os
import sys
import copy
from PIL import Image

from tqdm import tqdm

MAX_USERS = 100
size = (84, 84)


def name2json(name):
    file_path = os.path.join('raw', 'img_align_celeba', name)
    img = Image.open(file_path)
    gray = img.convert('RGB')
    gray.thumbnail(size, Image.ANTIALIAS)
    gray = gray.resize(size)
    arr = np.asarray(gray).copy().astype(np.uint8)
    vec = arr.flatten()
    vec = vec.tolist()
    return vec


if __name__ == '__main__':
    file = 'all_data/all_data.json'

    with open(file, 'r') as f:
        raw_data = json.load(f)

    data = copy.deepcopy(raw_data)
    for idx, user in enumerate(tqdm(raw_data['user_data'])):
        img_names = raw_data['user_data'][user]['x']
        data['user_data'][user]['x'] = []
        for name in img_names:
            js = name2json(name)
            data['user_data'][user]['x'].append(js)

    # Save to several json files

    cnt = 0
    file_id = 0
    all_data = {'users': [], 'num_samples': [], 'user_data': {}}

    for idx, user in enumerate(tqdm(data['user_data'])):
        all_data['users'].append(data['users'][idx])
        all_data['num_samples'].append(data['num_samples'][idx])
        all_data['user_data'][user] = data['user_data'][user]
        cnt += 1

        if cnt == MAX_USERS or idx == len(data['user_data']) - 1:
            file_name = f'all_data_{file_id}.json'
            file_path = os.path.join('new_all_data', file_name)
            with open(file_path, 'w') as outfile:
                json.dump(all_data, outfile)
            file_id += 1
            cnt = 0
            all_data = {'users': [], 'num_samples': [], 'user_data': {}}
