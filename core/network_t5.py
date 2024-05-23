import torch
import torch.nn as nn
from transformers import T5Tokenizer, T5ForConditionalGeneration

class Network(nn.Module):
    
    def __init__(self, cfg, src_lang, tgt_lang):
        super(Network, self).__init__()
        self.cfg = cfg
        self.tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-base')
        self.model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-base')

    def forward(self, input_texts, is_train=True):
        '''
            input_texts: List of input text sequences
        '''
        inputs = self.tokenizer(input_texts, return_tensors='pt', padding=True, truncation=True)
        outputs = self.model(**inputs)
        return outputs.logits

    def freeze_module(self, module):
        self.cfg.logger.info("Freezing module of "+" .......")
        for p in module.parameters():
            p.requires_grad = False

    def load_model(self, model_path):
        pretrain_dict = torch.load(model_path, map_location="cuda")
        self.model.load_state_dict(pretrain_dict)
        return pretrain_dict
