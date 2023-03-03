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


class Trainer:
    def __init__(self, corpus: CDRCorpus, config: CDRConfig, device: str, pos_weight: float = 1):
        self.corpus = corpus
        self.config = config
        self.encoder = GraphEncoder(
            time_step=config.time_step,
            word_vocab_size=len(corpus.word_vocab),
            edge_vocab_size=len(corpus.rel_vocab),
            pos_vocab_size=len(corpus.pos_vocab),
            char_vocab_size=len(corpus.char_vocab),
            contextual_word_embedding_dim=config.elmo_hidden_size,
            word_embedding_dim=config.word_embedding_dim,
            edge_embedding_dim=config.rel_embedding_dim,
            pos_embedding_dim=config.pos_embedding_dim,
            combined_embedding_dim=config.combined_embedding_dim,
            in_attn_heads=config.in_attn_heads,
            out_attn_heads=config.out_attn_heads,
            use_attn=config.use_attn,
            drop_out=config.drop_out,
            hidden_size=config.encoder_hidden_size,
            lstm_hidden_size=config.lstm_hidden_size,
            use_char=config.use_char,
            use_pos=config.use_pos,
            use_word=config.use_word,
            use_state=config.use_state,
            device=device
        )
        self.model = GraphStateLSTM(
            relation_classes=config.relation_classes,
            ner_classes=config.ner_classes,
            encoder=self.encoder,
            entity_hidden_size=config.entity_hidden_size,
            max_distance=config.max_distance,
            distance_embedding_dim=config.distance_embedding_dim,
            use_ner=config.use_ner,
            use_state=config.use_state,
            drop_out=config.drop_out,
            distance_thresh=config.distance_thresh,
            device=device
        )
        self.device = device
        self.weight_label = torch.Tensor([1, pos_weight])
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.lr, weight_decay=0.001)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.5)
        if device == "cuda":
            self.model = self.model.cuda()
            self.weight_label = self.weight_label.cuda()
        self.re_criterion = nn.CrossEntropyLoss(weight=self.weight_label)
        self.ner_criterion = nn.CrossEntropyLoss()
        self.model_name = f"model_modeltype_{str(config.model_type)}_char_{str(config.use_char)}_pos_{str(config.use_pos)}_attn_{str(config.use_attn)}_ner_{str(config.use_ner)}_state_{str(config.use_state)}_distance_{str(config.distance_thresh)}.pth"
        self.log_interval = 100

    def evaluate(self, dataloader: DataLoader):
        self.model.eval()
        re_losses = []
        ner_losses = []
        predict_list = []
        target_list = []
        ner_target_list = []
        ner_pred_list = []
        with torch.no_grad():
            print("Start evaluate")
            for batch in tqdm(dataloader):
                if self.device == "cuda":
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
        ner_f1 = compute_NER_f1_macro(ner_pred_list, ner_target_list)
        re_precision, re_recall, re_f1, _ = compute_results(predict_list, target_list)
        return np.mean(re_losses), np.mean(ner_losses), ner_f1, re_precision, re_recall, re_f1

    def train(self, train_loader: DataLoader, dev_loader: Optional[DataLoader] = None):
        train_step = 0
        for i in range(self.config.epochs):
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
                assert self.config.use_ner
                ner_logits, re_logits = self.model(inputs)
                re_loss = self.re_criterion(re_logits, labels)
                ner_loss = self.ner_criterion(torch.permute(ner_logits, (0, 2, 1)), ner_labels)

                total_loss = re_loss + ner_loss
                total_loss.backward()
                if train_step % self.config.gradient_accumalation == 0:
                    nn.utils.clip_grad_norm(self.model.parameters(), self.config.gradient_clipping)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    train_interval_re_loss.append(re_loss.item())
                    train_interval_ner_loss.append(ner_loss.item())
                    train_re_loss.append(re_loss.item())
                    train_ner_loss.append(ner_loss.item())

                train_step += 1
                if train_step % self.log_interval == 0:
                    print(f"Epoch {i} step {train_step}")
                    print(f"Re loss: {np.mean(train_interval_re_loss)}")
                    print(f"Ner loss: {np.mean(train_interval_ner_loss)}")
                    train_interval_re_loss = []
                    train_interval_ner_loss = []
            print(f"Finish epoch {i}")
            print(f"Re loss: {np.mean(train_re_loss)}")
            print(f"Ner loss: {np.mean(train_ner_loss)}")
            train_re_loss = []
            train_ner_loss = []
            self.scheduler.step()
            if dev_loader is not None:
                # TODO eval
                pass

    def save_model(self):
        torch.save(self.model.state_dict(), os.path.join(self.config.checkpoint_path,
                                                         f"model_{str(datetime.datetime.now())}"))
