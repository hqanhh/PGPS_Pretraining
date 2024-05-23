import torch
import json
from transformers import T5Tokenizer
from datasets.utils import *

class SrcLang:

    def __init__(self, vocab_path):
        self.tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-base')
        self.class_tag = ['[PAD]', '[GEN]', '[POINT]', '[NUM]', '[ARG]', '[ANGID]']
        self.sect_tag = ['[PAD]', '[PROB]', '[COND]', '[STRU]']
    
    def indexes_from_sentence(self, sentence, id_type='text'):
        if id_type == 'text':
            return self.tokenizer(sentence, padding='max_length', truncation=True, return_tensors="pt").input_ids
        elif id_type == 'class_tag':
            return [self.class_tag.index(word) for word in sentence]
        elif id_type == 'sect_tag':
            return [self.sect_tag.index(word) for word in sentence]

    def sentence_from_indexes(self, indexes):
        return self.tokenizer.decode(indexes, skip_special_tokens=True)

class TgtLang:

    def __init__(self, vocab_path):
        self.tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-base')
    
    def indexes_from_sentence(self, sentence, var_values=None, arg_values=None):
        inputs = self.tokenizer(sentence, padding='max_length', truncation=True, return_tensors="pt")
        return inputs.input_ids

    def sentence_from_indexes(self, indexes, change_dict={}):
        return self.tokenizer.decode(indexes, skip_special_tokens=True)

class SN:
    def __init__(self):
        self.token = []  # str list
        self.sect_tag = []  # [PROB]/[COND]/[STRU]
        self.class_tag = []  # [GEN]/[NUM]/[ARG]/[POINT]/[ANGID]

def get_raw_pairs(dataset_path):
    raw_pairs = []
    with open(dataset_path, 'r') as fp:
        content_all = json.load(fp)

    for key, content in content_all.items():
        text = content['text']
        stru_seqs = content['parsing_stru_seqs']
        sem_seqs = content['parsing_sem_seqs']
        text_data, stru_data, sem_data = SN(), SN(), SN()
        text_data.token = get_token(text)
        stru_data.token = [get_token(item) + [','] for item in stru_seqs]
        sem_data.token = [get_token(item) + [','] for item in sem_seqs]
        text_data.sect_tag = []
        stru_data.sect_tag = [['[STRU]'] * len(item) for item in stru_data.token]
        sem_data.sect_tag = [['[COND]'] * len(item) for item in sem_data.token]
        split_text(text_data)
        text_data.class_tag = ['[GEN]'] * len(text_data.token)
        stru_data.class_tag = [['[GEN]'] * len(item) for item in stru_data.token]
        sem_data.class_tag = [['[GEN]'] * len(item) for item in sem_data.token]
        get_point_angleID_tag(text_data, stru_data, sem_data)
        get_num_arg_tag(text_data, sem_data)
        expression = content['expression'].split(' ')
        remove_sem_dup(text_data, sem_data, expression)

        content['text'] = text_data
        content['parsing_stru_seqs'] = stru_data
        content['parsing_sem_seqs'] = sem_data
        content['expression'] = expression
        content['id'] = key

        raw_pairs.append(content)

    return raw_pairs

class Collater:

    def __init__(self, args, src_lang):
        self.args = args
        self.src_lang = src_lang

    def __call__(self, batch_data, padding_id=0):
        text_tokens, text_sect_tags, text_class_tags = list(zip(*batch_data))

        len_text = [len(seq_tag) for seq_tag in text_class_tags]
        max_len_text = max(len_text)

        for k in range(len(text_tokens)):
            for j in range(len(text_tokens[k])):
                text_tokens[k][j] += [padding_id] * (max_len_text - len(text_tokens[k][j]))
        text_sect_tags = [seq_tag + [padding_id] * (max_len_text - len(seq_tag)) for seq_tag in text_sect_tags]
        text_class_tags = [seq_tag + [padding_id] * (max_len_text - len(seq_tag)) for seq_tag in text_class_tags]

        text_tokens = torch.LongTensor(text_tokens)
        text_sect_tags = torch.LongTensor(text_sect_tags)
        text_class_tags = torch.LongTensor(text_class_tags)
        len_text = torch.LongTensor(len_text)

        text_tokens, labels = self.get_mask_tokens(text_tokens, text_class_tags, self.src_lang, self.args.mlm_probability)

        text_dict = {
            'token': text_tokens,
            'sect_tag': text_sect_tags,
            'class_tag': text_class_tags,
            'len': len_text,
            'labels': labels
        }

        return text_dict

    def get_mask_tokens(self, text_tokens, text_class_tags, src_lang, mlm_probability):
        prob_replace_mask = 0.8
        prob_replace_rand = 0.1
        prob_keep_ori = 0.1

        labels = text_tokens[:, 0].clone()
        inputs = text_tokens[:, 0].clone()
        probability_matrix = torch.full(labels.shape, mlm_probability)

        special_tokens_mask = text_class_tags == src_lang.class_tag.index('[PAD]')
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        indices_replaced = torch.bernoulli(torch.full(labels.shape, prob_replace_mask)).bool() & masked_indices
        inputs[indices_replaced] = src_lang.tokenizer.convert_tokens_to_ids(src_lang.tokenizer.mask_token)

        current_prob = prob_replace_rand / (1 - prob_replace_mask)
        indices_random = torch.bernoulli(torch.full(labels.shape, current_prob)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(src_lang.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        text_tokens = torch.stack((inputs, text_tokens[:, 1]), dim=1)

        return text_tokens, labels
