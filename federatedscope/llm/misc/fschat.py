import torch
import transformers

transformers.logging.set_verbosity(40)

from federatedscope.core.configs.config import global_cfg
from federatedscope.core.cmd_args import parse_args, parse_client_cfg
from federatedscope.llm.dataloader.dataloader import get_tokenizer
from federatedscope.llm.model.model_builder import get_llm
from federatedscope.llm.dataset.llm_dataset import PROMPT_DICT
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger


class FSChatBot(object):
    def __init__(self, config):
        model_name, _ = config.model.type.split('@')
        self.tokenizer, _ = get_tokenizer(model_name, config.data.root,
                                          config.llm.tok_len)
        self.model = get_llm(config)

        try:
            ckpt = torch.load(config.federate.save_to, map_location='cpu')
            if 'model' and 'cur_round' in ckpt:
                self.model.load_state_dict(ckpt['model'])
            else:
                self.model.load_state_dict(ckpt)
        except Exception as error:
            print(f"{error}, will use raw model.")

        if config.train.is_enable_half:
            self.model.half().to(f'cuda:{config.device}')
        else:
            self.model.to(f'cuda:{config.device}')
        self.model = self.model.eval()

        self.max_history_len = config.llm.chat.max_history_len
        self.max_len = config.llm.chat.max_len
        self.history = []

    def _build_prompt(self, input_text):
        source = {'instruction': input_text}
        return PROMPT_DICT['prompt_no_input'].format_map(source)

    def predict(self, input_text, use_history=True, use_prompt=True):
        if use_prompt:
            input_text = self._build_prompt(input_text)
        text_ids = self.tokenizer.encode(input_text, add_special_tokens=False)
        self.history.append(text_ids)
        input_ids = []
        if use_history:
            for history_ctx in self.history[-self.max_history_len:]:
                input_ids.extend(history_ctx)
        else:
            input_ids.extend(text_ids)
        input_ids = torch.tensor(input_ids).long()
        input_ids = input_ids.unsqueeze(0).cuda()
        response = self.model.generate(input_ids,
                                       max_length=self.max_len,
                                       num_beams=4,
                                       no_repeat_ngram_size=2,
                                       early_stopping=True,
                                       temperature=0.0)

        self.history.append(response[0].tolist())
        response_tokens = \
            self.tokenizer.decode(response[0][input_ids.shape[1]:],
                                  skip_special_tokens=True)
        return response_tokens

    def clear(self):
        self.history = []


def main():
    init_cfg = global_cfg.clone()
    args = parse_args()
    if args.cfg_file:
        init_cfg.merge_from_file(args.cfg_file)
    cfg_opt, client_cfg_opt = parse_client_cfg(args.opts)
    init_cfg.merge_from_list(cfg_opt)

    update_logger(init_cfg, clear_before_add=True)
    setup_seed(init_cfg.seed)

    chat_bot = FSChatBot(init_cfg)
    welcome = "Welcome to FSChatBot，" \
              "`clear` to clear history，" \
              "`quit` to end chat."
    print(welcome)
    while True:
        input_text = input("\nUser:")
        if input_text.strip() == "quit":
            break
        if input_text.strip() == "clear":
            chat_bot.clear()
            print(welcome)
            continue
        print(f'\nFSBot: {chat_bot.predict(input_text)}')


if __name__ == "__main__":
    main()
