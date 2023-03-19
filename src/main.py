import argparse
from utils.lr_scheduler import get_cosine_warm_up_lr_scheduler
from corpus.cdr_corpus import CDRCorpus
from dataset.cdr_dataset import CDRDataset
from config.cdr_config import CDRConfig
from dataset.utils import get_cdr_dataset, concat_dataset
from dataset.collator import Collator
from torch.utils.data import DataLoader
from model.trainer import Trainer
import torch
import random
import numpy as np


def seed_all(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config.json")
    parser.add_argument("--seed", default=22, type=int)
    parser.add_argument("--concat", action="store_true")
    args = parser.parse_args()
    seed_all(args.seed)
    config = CDRConfig.from_json(args.config)
    corpus = CDRCorpus(config)
    corpus.load_all_vocabs(config.data.saved_data_path)
    train_dataset = get_cdr_dataset(corpus, config.data.saved_data_path, "train")
    dev_dataset = get_cdr_dataset(corpus, config.data.saved_data_path, "dev")
    test_dataset = get_cdr_dataset(corpus, config.data.saved_data_path, "test")
    device = "cuda"
    # device = "cpu"
    trainer = Trainer(corpus, config, device)
    if args.concat:
        collator = Collator(corpus.word_vocab, corpus.pos_vocab, corpus.char_vocab, corpus.rel_vocab)
        train_dataset = concat_dataset([train_dataset, dev_dataset])
        train_loader = DataLoader(
            train_dataset, batch_size=config.train.batch_size, shuffle=True, collate_fn=collator.collate
        )
        test_loader = DataLoader(
            test_dataset, batch_size=config.train.batch_size, shuffle=False, collate_fn=collator.collate
        )
        trainer.train(train_loader)
        re_loss, ner_loss, ner_f1, re_precision, re_recall, re_f1 = trainer.evaluate(test_loader)
        print(f"Re loss: {re_loss}\nNer loss: {ner_loss}\nNer f1: {ner_f1}\n"
              f"Re precision: {re_precision}\nRe recall: {re_recall}\nRe f1: {re_f1}")
        with open("result.txt", "a") as outfile:
            outfile.write("\n\nRESULT")
            outfile.write(f"Re loss: {re_loss}\nNer loss: {ner_loss}\nNer f1: {ner_f1}\n"
                          f"Re precision: {re_precision}\nRe recall: {re_recall}\nRe f1: {re_f1}" + "\n")
    else:
        collator = Collator(corpus.word_vocab, corpus.pos_vocab, corpus.char_vocab, corpus.rel_vocab)
        train_loader = DataLoader(
            train_dataset, batch_size=config.train.batch_size, shuffle=True, collate_fn=collator.collate
        )
        dev_loader = DataLoader(
            dev_dataset, batch_size=config.train.batch_size, shuffle=True, collate_fn=collator.collate
        )
        trainer.train(train_loader, dev_loader)
