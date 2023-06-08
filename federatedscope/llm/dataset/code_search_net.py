import os

from subprocess import call

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


def main():
    destination_dir = os.path.abspath('data')
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    os.chdir(destination_dir)

    for language in CSN_FILE_NUM_DICT.keys():
        call([
            'wget', 'https://huggingface.co/datasets'
            '/code_search_net/resolve/main/data/{}.zip'.format(language), '-P',
            destination_dir, '-O', '{}.zip'.format(language)
        ])
        call(['unzip', '{}.zip'.format(language)])


if __name__ == '__main__':
    main()
