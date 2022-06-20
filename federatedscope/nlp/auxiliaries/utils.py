import time
import sys
from transformers.models.bert import BertTokenizerFast


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        self.val = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                    (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()


def setup_tokenizer(config):
    bos_token, eos_token, eoq_token = '[unused0]', '[unused1]', '[unused2]'
    tokenizer = BertTokenizerFast.from_pretrained(
        config.model.bert_type,
        additional_special_tokens=[bos_token, eos_token, eoq_token],
        skip_special_tokens=True,
    )
    return tokenizer
