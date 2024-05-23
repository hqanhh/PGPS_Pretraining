import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import T5ForConditionalGeneration, AdamW, get_linear_schedule_with_warmup
from utils.utils import *
from core.worker import train
from core.network import Network
from datasets.preprocessing_t5 import SrcLang, TgtLang, get_raw_pairs, Collater
from datasets import get_dataloader

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

# Training script
def main_worker(args):
    set_seed(args)
    src_lang, tgt_lang = get_language(args)
    train_loader, train_sampler, val_loader, src_lang, tgt_lang = get_dataloader(args)
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
