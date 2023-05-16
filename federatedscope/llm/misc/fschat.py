import torch
import torch.nn.functional as F

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
            self.model.load_state_dict(ckpt)
        except Exception as error:
            print(f"{error}, will use raw model.")

        self.model.half().cuda()
        self.model = self.model.eval()

        self.max_history_len = config.llm.chat.max_history_len
        self.max_len = config.llm.chat.max_len
        self.history = []

    def _build_prompt(self, input_text):
        source = {'instruction': input_text}
        return PROMPT_DICT['prompt_no_input'].format_map(source)

    def _format_output(self, response_tokens):
        return "".join(response_tokens).replace('Ġ', ' ').replace('Ċ', ' ')

    def predict(self, input_text, use_history=True):
        input_text = self._build_prompt(input_text)
        text_ids = self.tokenizer.encode(input_text, add_special_tokens=False)
        self.history.append(text_ids)
        input_ids = [self.tokenizer.bos_token_id]
        if use_history:
            for history_id, history_utr in enumerate(
                    self.history[-self.max_history_len:]):
                input_ids.extend(history_utr)
                input_ids.append(self.tokenizer.eos_token_id)
        else:
            input_ids.extend(text_ids)
            input_ids.append(self.tokenizer.eos_token_id)
        input_ids = torch.tensor(input_ids).long()
        input_ids = input_ids.unsqueeze(0).cuda()
        response = []
        for _ in range(self.max_len):
            outputs = self.model(input_ids=input_ids)
            logits = outputs.logits
            next_token_logits = logits[0, -1, :]
            next_token_logits[self.tokenizer.convert_tokens_to_ids(
                '[UNK]')] = -float('Inf')
            next_token = torch.multinomial(F.softmax(next_token_logits,
                                                     dim=-1),
                                           num_samples=1)
            if next_token == self.tokenizer.sep_token_id:
                break
            response.append(next_token.item())
            input_ids = torch.cat((input_ids, next_token.unsqueeze(0)), dim=1)
        self.history.append(response)
        response_tokens = self.tokenizer.convert_ids_to_tokens(response)
        return self._format_output(response_tokens)

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
        print(chat_bot.predict(input_text))


if __name__ == "__main__":
    main()
