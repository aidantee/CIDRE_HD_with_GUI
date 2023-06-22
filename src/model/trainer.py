import datetime
import os.path
import time
from typing import Optional

import numpy as np
import torch
from torch import optim, nn, Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from corpus.cdr_corpus import CDRCorpus
from config.cdr_config import CDRConfig
from model.cdr_model import GraphEncoder, GraphStateLSTM
from utils.metrics import compute_rel_f1, compute_NER_f1_macro, decode_ner, compute_results

WORD2VEC_PATH = "./data/word_embedding.pt"


class Trainer:
    def __init__(self, corpus: CDRCorpus, config: CDRConfig, device: str, experiment_dir: str, logger,
                 pos_weight: float = 1):
        self.logger = logger
        self.experiment_dir = experiment_dir
        self.corpus = corpus
        self.config = config
        self.model = GraphStateLSTM(
            len(corpus.rel_vocab),
            len(corpus.pos_vocab),
            len(corpus.char_vocab),
            len(corpus.word_vocab),
            config.model,
            device=device
        )
        if device == "cuda":
            self.word_embedding_weight = torch.load(WORD2VEC_PATH, map_location=torch.device("cuda"))
        else:
            self.word_embedding_weight = torch.load(WORD2VEC_PATH, map_location=torch.device("cpu"))
        self.model.encoder.word_embedding.from_pretrained(self.word_embedding_weight, freeze=True)
        num_param = sum([param.numel() for param in self.model.parameters()])
        self.logger.info(f"Num model param {num_param}")
        self.device = device
        self.weight_label = torch.Tensor([1, pos_weight])
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.train.optimizer.lr, weight_decay=0.001)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.5)
        if device == "cuda":
            self.model = self.model.cuda()
            self.weight_label = self.weight_label.cuda()
        self.re_criterion = nn.CrossEntropyLoss(weight=self.weight_label)
        self.ner_criterion = nn.CrossEntropyLoss()
        # self.model_name = f"model_modeltype_{str(config.model_type)}_char_{str(config.use_char)}_pos_{str(config.use_pos)}_attn_{str(config.use_attn)}_ner_{str(config.use_ner)}_state_{str(config.use_state)}_distance_{str(config.distance_thresh)}.pth"
        self.log_interval = 100

    def evaluate(self, dataloader: DataLoader):
        self.model.eval()
        re_losses = []
        ner_losses = []
        predict_list = []
        target_list = []
        ner_target_list = []
        ner_pred_list = []
        elapsed_times = []
        with torch.no_grad():
            self.logger.info("Start evaluate")
            for batch in tqdm(dataloader):
                if self.device == "cuda":
                    start = time.time()
                    batch = [elem.cuda() if isinstance(elem, Tensor) else elem for elem in batch]
                    inputs = batch[:-2]
                    ner_labels = batch[-2]
                    labels = batch[-1]
                    ner_logits, re_logits = self.model(inputs)
                    re_loss = self.re_criterion(re_logits, labels)
                    ner_loss = self.ner_criterion(torch.permute(ner_logits, (0, 2, 1)), ner_labels)
                    total_loss = re_loss + ner_loss
                    predict_classes = torch.argmax(re_logits, dim=-1).cpu().data.numpy().tolist()
                    target_classes = labels.cpu().data.numpy().tolist()
                    predict_list.extend(predict_classes)
                    target_list.extend(target_classes)

                    ner_pred_classes = torch.argmax(ner_logits, dim=-1).cpu().data.numpy().tolist()
                    ner_target_classes = ner_labels.cpu().data.numpy().tolist()

                    ner_pred_classes = decode_ner(ner_pred_classes)
                    ner_target_classes = decode_ner(ner_target_classes)

                    ner_target_list.extend(ner_target_classes)
                    ner_pred_list.extend(ner_pred_classes)

                    re_losses.append(re_loss.item())
                    ner_losses.append(ner_loss.item())
                    end = time.time()
                    elapsed_times.append(end - start)

        elapsed_times = np.array(elapsed_times)
        self.logger.info(f"Inference time: \n"
                         f"Min: {np.min(elapsed_times)}\n"
                         f"Max: {np.max(elapsed_times)}\n"
                         f"Mean: {np.mean(elapsed_times)}\n"
                         f"Median: {np.median(elapsed_times)}\n")
        ner_f1 = compute_NER_f1_macro(ner_pred_list, ner_target_list)
        re_precision, re_recall, re_f1, _ = compute_results(predict_list, target_list)
        return np.mean(re_losses), np.mean(ner_losses), ner_f1, re_precision, re_recall, re_f1

    def train(self, train_loader: DataLoader, dev_loader: Optional[DataLoader] = None):
        train_step = 0
        for i in range(self.config.train.num_epochs):
            train_re_loss = []
            train_ner_loss = []
            train_interval_re_loss = []
            train_interval_ner_loss = []
            for batch in tqdm(train_loader):
                self.model.train()
                if self.device == "cuda":
                    batch = [elem.cuda() if isinstance(elem, Tensor) else elem for elem in batch]
                self.model.zero_grad()
                inputs = batch[:-2]
                ner_labels = batch[-2]
                labels = batch[-1]

                ner_logits, re_logits = self.model(inputs)
                re_loss = self.re_criterion(re_logits, labels)
                ner_loss = self.ner_criterion(torch.permute(ner_logits, (0, 2, 1)), ner_labels)

                if self.config.model.use_ner:
                    total_loss = re_loss + ner_loss
                else:
                    total_loss = re_loss
                total_loss.backward()

                nn.utils.clip_grad_norm(self.model.parameters(), self.config.train.gradient_clipping)
                self.optimizer.step()
                self.optimizer.zero_grad()
                train_interval_re_loss.append(re_loss.item())
                train_interval_ner_loss.append(ner_loss.item())
                train_re_loss.append(re_loss.item())
                train_ner_loss.append(ner_loss.item())

                train_step += 1
                if train_step % self.log_interval == 0:
                    self.logger.info(f"Epoch {i} step {train_step}")
                    self.logger.info(f"Re loss: {np.mean(train_interval_re_loss)}")
                    self.logger.info(f"Ner loss: {np.mean(train_interval_ner_loss)}")
                    train_interval_re_loss = []
                    train_interval_ner_loss = []

            self.scheduler.step()
            self.logger.info(f"Finish epoch {i}")
            self.logger.info(f"Re loss: {np.mean(train_re_loss)}")
            self.logger.info(f"Ner loss: {np.mean(train_ner_loss)}")
            train_re_loss = []
            train_ner_loss = []
            if dev_loader is not None:
                re_loss, ner_loss, ner_f1, re_precision, re_recall, re_f1 = self.evaluate(dev_loader)
                self.logger.info(f"Re loss: {re_loss}\nNer loss: {ner_loss}\nNer f1: {ner_f1}\n"
                                 f"Re precision: {re_precision}\nRe recall: {re_recall}\nRe f1: {re_f1}")

    def save_model(self):
        ckpt = {
            "model": self.model.state_dict()
        }
        save_path = f"{self.experiment_dir}/model.pth"
        torch.save(ckpt, f=save_path)
