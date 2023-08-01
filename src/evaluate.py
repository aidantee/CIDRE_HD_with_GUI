import argparse
import logging
import time

from torch import Tensor
from tqdm import tqdm

from corpus.cdr_corpus import CDRCorpus
from config.cdr_config import CDRConfig
from dataset.utils import get_cdr_dataset, concat_dataset
from dataset.collator import Collator
from torch.utils.data import DataLoader

from model.cdr_model import GraphStateLSTM
from model.trainer import Trainer
import torch
import random
import numpy as np
import re
from datetime import datetime
import os
import json

from utils.metrics import compute_results


def evaluate(model, dataloader: DataLoader, threshold: float = 0.5, device='cuda'):
    model.eval()
    predict_list = []
    target_list = []
    elapsed_times = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            if device == "cuda":
                start = time.time()
                batch = [elem.cuda() if isinstance(elem, Tensor) else elem for elem in batch]
                inputs = batch[:-2]
                ner_labels = batch[-2]
                labels = batch[-1]
                ner_logits, re_logits = model(inputs)
                re_logits = torch.softmax(re_logits, dim=-1)
                predict_classes = [0 if float(sample[0]) > threshold else 1 for sample in re_logits]
                target_classes = labels.cpu().data.numpy().tolist()
                predict_list.extend(predict_classes)
                target_list.extend(target_classes)
                end = time.time()
                elapsed_times.append(end - start)
    re_precision, re_recall, re_f1, _ = compute_results(predict_list, target_list)
    print(re_precision, re_recall, re_f1)
    return re_precision, re_recall, re_f1


if __name__ == "__main__":
    config_path = "./config.json"
    predict_threshold = 0.7
    model_ckpt_path = 'checkpoints/cdr_2023_07_28_23_20_14/model.pth'
    config = CDRConfig.from_json(config_path)
    corpus = CDRCorpus(config)
    corpus.load_all_vocabs(config.data.saved_data_path)
    device = 'cuda'
    model = GraphStateLSTM(
        len(corpus.rel_vocab),
        len(corpus.pos_vocab),
        len(corpus.char_vocab),
        len(corpus.word_vocab),
        config.model,
        device=device
    )
    model_state_dict = torch.load(model_ckpt_path)['model']
    print(model.load_state_dict(model_state_dict))
    if device == 'cuda':
        model.to('cuda')
    collator = Collator(corpus.word_vocab, corpus.pos_vocab,
                        corpus.char_vocab, corpus.rel_vocab)
    test_dataset = get_cdr_dataset(corpus, config.data.saved_data_path, "test")
    test_loader = DataLoader(
        test_dataset, batch_size=config.train.batch_size, shuffle=False, collate_fn=collator.collate
    )
    evaluate(model, test_loader, predict_threshold, device)
