import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import T5ForConditionalGeneration, AdamW, get_linear_schedule_with_warmup
from utils.utils import *
from core.worker import train
from core.network import Network
from datasets.preprocessing_t5 import SrcLang, TgtLang, get_raw_pairs, Collater

# Utility functions
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

def get_language(args):
    src_lang = SrcLang(args.vocab_src_path)
    tgt_lang = TgtLang(args.vocab_tgt_path)
    return src_lang, tgt_lang

class CustomDataset(Dataset):
    def __init__(self, raw_pairs, src_lang, tgt_lang):
        self.raw_pairs = raw_pairs
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def __len__(self):
        return len(self.raw_pairs)

    def __getitem__(self, idx):
        sample = self.raw_pairs[idx]
        src_input = self.src_lang.indexes_from_sentence(sample['text'], id_type='text')
        tgt_input = self.tgt_lang.indexes_from_sentence(sample['expression'])
        return src_input, tgt_input

def get_dataloader(args):
    raw_pairs = get_raw_pairs(args.dataset_dir)
    train_size = int(0.8 * len(raw_pairs))
    val_size = len(raw_pairs) - train_size
    train_dataset = CustomDataset(raw_pairs[:train_size], args.src_lang, args.tgt_lang)
    val_dataset = CustomDataset(raw_pairs[train_size:], args.src_lang, args.tgt_lang)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=Collater(args, args.src_lang))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=Collater(args, args.src_lang))

    return train_loader, None, val_loader

# Training script
def main_worker(args):
    set_seed(args)
    src_lang, tgt_lang = get_language(args)
    train_loader, train_sampler, val_loader = get_dataloader(args)
    model = Network(args, src_lang, tgt_lang).cuda()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_loader) * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    model = torch.nn.parallel.DistributedDataParallel(
        model, 
        device_ids=[args.local_rank], 
        output_device=args.local_rank, 
        find_unused_parameters=True
    )

    model.train()

    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_loader):
            inputs = batch['token'].to(args.device)
            labels = batch['labels'].to(args.device)

            model.zero_grad()
            outputs = model(inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % args.logging_steps == 0:
                print(f"Epoch {epoch}, Step {step}, Loss {loss.item()}")

    print("Training completed.")
