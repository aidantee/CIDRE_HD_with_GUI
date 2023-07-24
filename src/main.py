import argparse
import logging

from corpus.cdr_corpus import CDRCorpus
from config.cdr_config import CDRConfig
from dataset.utils import get_cdr_dataset, concat_dataset
from dataset.collator import Collator
from torch.utils.data import DataLoader
from model.trainer import Trainer
import torch
import random
import numpy as np
import re
from datetime import datetime
import os
import json


def get_logger(filename: str):
    logger = logging.getLogger("TRAIN")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(name)s %(filename)s %(funcName)s %(levelname)s %(module)s %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(filename=filename)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def seed_all(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)


def setup_experiment_dir(config: CDRConfig):
    experiment_name = re.sub(r"\s", "_", config.experiment)
    experiment_subdir = f"{experiment_name}_{datetime.today().strftime('%Y_%m_%d_%H_%M_%S')}"
    experiment_dir = os.path.join(config.experiment_dir, experiment_subdir)
    os.makedirs(experiment_dir)
    with open(f"{experiment_dir}/config.json", "w") as outfile:
        json.dump(json.loads(str(config)), outfile, indent=4, ensure_ascii=False)
    return experiment_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config.json")
    parser.add_argument("--seed", default=22, type=int)
    parser.add_argument("--concat", action="store_true")
    args = parser.parse_args()
    seed_all(args.seed)
    config = CDRConfig.from_json(args.config)

    experiment_dir = setup_experiment_dir(config)
    logger = get_logger(os.path.join(experiment_dir, "log.txt"))
    logger.info(config)
    corpus = CDRCorpus(config)
    corpus.load_all_vocabs(config.data.saved_data_path)
    train_dataset = get_cdr_dataset(corpus, config.data.saved_data_path, "train")
    dev_dataset = get_cdr_dataset(corpus, config.data.saved_data_path, "dev")
    test_dataset = get_cdr_dataset(corpus, config.data.saved_data_path, "test")
    device = "cuda"
    # device = "cpu"
    trainer = Trainer(corpus, config, device, experiment_dir, logger)
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
        logger.info(f"Re loss: {re_loss}\nNer loss: {ner_loss}\nNer f1: {ner_f1}\n"
                    f"Re precision: {re_precision}\nRe recall: {re_recall}\nRe f1: {re_f1}")
    else:
        collator = Collator(corpus.word_vocab, corpus.pos_vocab, corpus.char_vocab, corpus.rel_vocab)
        train_loader = DataLoader(
            train_dataset, batch_size=config.train.batch_size, shuffle=True, collate_fn=collator.collate
        )
        dev_loader = DataLoader(
            dev_dataset, batch_size=config.train.batch_size, shuffle=True, collate_fn=collator.collate
        )
        trainer.train(train_loader, dev_loader)
    trainer.save_model()
