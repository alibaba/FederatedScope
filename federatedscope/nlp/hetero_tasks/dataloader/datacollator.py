import math
import numpy as np
import torch
from numpy.random import permutation, poisson


class DataCollatorForMLM(object):
    def __init__(self, tokenizer, mlm_probability=0.15):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability

    def __call__(self, examples):
        """ Prepare masked tokens inputs/labels for masked language
        modeling: 80% MASK, 10% random, 10% original. """
        examples = {
            k: torch.stack([x[k] for x in examples])
            for k in examples[0].keys()
        }
        token_ids = examples['token_ids']
        attention_mask = examples['attention_mask']
        labels = token_ids.clone()

        # We sample a few tokens in each sequence for masked-LM training
        # (with probability self.mlm_probability defaults to 0.15 in
        # Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(
                val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask,
                                                     dtype=torch.bool),
                                        value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with
        # tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(
            labels.shape, 0.8)).bool() & masked_indices
        token_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = \
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & \
            masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer),
                                     labels.shape,
                                     dtype=torch.long)
        token_ids[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input
        # tokens unchanged
        return {
            'token_ids': token_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'example_indices': examples['example_indices']
        }


class DataCollatorForDenoisingReconstrcution(object):
    """Data collator used denoising language modeling task in BART.
    The implementation is based on
    https://github.com/pytorch/fairseq/blob/
    1bba712622b8ae4efb3eb793a8a40da386fe11d0/fairseq/data/denoising_dataset.py.
    The default paramters is based on BART paper
    https://arxiv.org/abs/1910.13461.
    """
    def __init__(self,
                 tokenizer,
                 mask_ratio=0.3,
                 poisson_lambda=3.0,
                 permutate_sentence_ratio=1.0):
        self.tokenizer = tokenizer
        self.mask_ratio = mask_ratio
        self.poisson_lambda = poisson_lambda
        self.permutate_sentence_ratio = permutate_sentence_ratio

    def __call__(self, examples):
        examples = {
            k: torch.stack([x[k] for x in examples])
            for k in examples[0].keys()
        }
        token_ids = examples['token_ids'].numpy()
        attention_mask = examples['attention_mask'].numpy()
        labels = token_ids.copy()

        do_permutate = False
        if self.permutate_sentence_ratio > 0.0:
            permute_sent = self.permutate_sentences(token_ids[:, 1:])
            for i, s in enumerate(permute_sent):
                token_ids[i, 1:] = s
            do_permutate = True

        if self.mask_ratio:
            token_ids, _ = self.add_whole_word_mask(token_ids, do_permutate)
            num_non_padding = np.sum(token_ids != self.tokenizer.pad_token_id,
                                     axis=-1)
            for i in range(len(attention_mask)):
                attention_mask[i][num_non_padding[i]:] = 0

        token_ids = torch.from_numpy(token_ids)
        attention_mask = torch.from_numpy(attention_mask)
        labels = torch.from_numpy(labels)
        return {
            'token_ids': token_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'example_indices': examples['example_indices']
        }

    def permutate_sentences(self, inputs):
        results = inputs.copy()

        for i in range(inputs.shape[0]):
            full_stops = (inputs[i] == self.tokenizer.eoq_token_id) | (
                inputs[i] == self.tokenizer.eos_token_id)
            full_stops = full_stops[None, :]
            sentence_ends = np.argwhere(full_stops[:, 1:] *
                                        ~full_stops[:, :-1])
            if len(sentence_ends) == 0:
                continue

            sentence_ends[:, 1] += 2
            num_sentences = np.unique(sentence_ends[:, 0],
                                      return_counts=True)[1]
            num_to_permute = np.ceil(
                (num_sentences * 2 * self.permutate_sentence_ratio) /
                2.0).astype(int)
            sentence_ends = np.split(
                sentence_ends[:, 1],
                np.unique(sentence_ends[:, 0], return_index=True)[1][1:])

            substitutions = np.random.permutation(
                num_sentences[0])[:num_to_permute[0]]
            ordering = np.arange(0, num_sentences[0])
            ordering[substitutions] = substitutions[np.random.permutation(
                num_to_permute[0])]

            index = 0
            for j in ordering:
                sentence = inputs[i, (
                    sentence_ends[0][j -
                                     1] if j > 0 else 0):sentence_ends[0][j]]
                results[i, index:index + sentence.shape[0]] = sentence
                index += sentence.shape[0]

        num_non_padding = np.sum(results != self.tokenizer.pad_token_id,
                                 axis=-1)
        eos_indices = np.where(results == self.tokenizer.eos_token_id)[1]
        for i, (idx1, idx2) in enumerate(zip(eos_indices, num_non_padding)):
            results[i][idx1] = self.tokenizer.eoq_token_id
            results[i][idx2 - 1] = self.tokenizer.eos_token_id

        return results

    def add_whole_word_mask(self, inputs, do_permutate):
        labels = inputs.copy()
        inputs = inputs.copy()

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(
                val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        special_tokens_mask = np.array(special_tokens_mask, dtype=bool)

        # determine how many tokens we need to mask in total
        is_token = ~(labels == self.tokenizer.pad_token_id) & \
                   ~special_tokens_mask
        num_to_mask = int(
            math.ceil(is_token.astype(float).sum() * self.mask_ratio))
        if num_to_mask == 0:
            return inputs, labels

        # generate a sufficient number of span lengths
        lengths = poisson(lam=self.poisson_lambda, size=(num_to_mask, ))
        while np.cumsum(lengths, 0)[-1] < num_to_mask:
            lengths = np.concatenate([
                lengths,
                poisson(lam=self.poisson_lambda, size=(num_to_mask, ))
            ])

        # remove all spans of length 0
        # Note that BART inserts additional mask tokens where length == 0,
        # which we do not implement for now as it adds additional complexity
        lengths = lengths[lengths > 0]

        # trim to about num_to_mask tokens
        idx = np.argmin(np.abs(np.cumsum(lengths, 0) - num_to_mask)) + 1
        lengths = lengths[:idx + 1]

        # select span start indices
        token_indices = np.argwhere(is_token == 1)
        span_starts = permutation(token_indices.shape[0])[:lengths.shape[0]]

        # prepare mask
        masked_indices = np.array(token_indices[span_starts])
        mask = np.full_like(labels, fill_value=False)

        # mask span start indices
        for mi in masked_indices:
            mask[tuple(mi)] = True
        lengths -= 1

        # fill up spans
        max_index = labels.shape[1] - 1
        remaining = (lengths > 0) & (masked_indices[:, 1] < max_index)
        while np.any(remaining):
            masked_indices[remaining, 1] += 1
            for mi in masked_indices:
                mask[tuple(mi)] = True
            lengths -= 1
            remaining = (lengths > 0) & (masked_indices[:, 1] < max_index)

        # place the mask tokens
        mask[np.where(special_tokens_mask)] = False
        inputs[np.where(mask)] = self.tokenizer.mask_token_id

        if not do_permutate:
            labels[np.where(mask)] = -100
        else:
            labels[np.where(special_tokens_mask)] = -100

        # remove mask tokens that are not starts of spans
        to_remove = (mask == 1) & np.roll((mask == 1), 1, 1)
        new_inputs = np.full_like(labels,
                                  fill_value=self.tokenizer.pad_token_id)

        # splits = list(map(lambda x: x.reshape(-1),  np.split(inputs_copy,
        # indices_or_sections=2, axis=0))
        for i, example in enumerate(
                np.split(inputs,
                         indices_or_sections=new_inputs.shape[0],
                         axis=0)):
            new_example = example[0][~to_remove[i]]
            new_inputs[i, 0:new_example.shape[0]] = new_example

        # batching now fixed
        return new_inputs, labels


class DataCollator(object):
    def __init__(self,
                 tokenizer,
                 mlm_probability=0.15,
                 mask_ratio=0.3,
                 poisson_lambda=3.0,
                 permutate_sentence_ratio=1.0):
        self.mlm_collator = DataCollatorForMLM(tokenizer, mlm_probability)
        self.denoise_collator = DataCollatorForDenoisingReconstrcution(
            tokenizer, mask_ratio, poisson_lambda, permutate_sentence_ratio)

    def __call__(self, examples):
        mlm_results = self.mlm_collator(examples)
        denoise_results = self.denoise_collator(examples)
        return {'mlm': mlm_results, 'denoise': denoise_results}
