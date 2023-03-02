import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import os
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader

from config.cdr_config import CDRConfig
from tqdm import tqdm
from dataset.cdr_dataset import CDRDataset
from corpus.cdr_corpus import CDRCorpus
from model.cdr_model import GraphEncoder, GraphStateLSTM
from utils.metrics import compute_rel_f1, compute_NER_f1_macro, decode_ner, compute_results
from dataset.utils import get_cdr_dataset
from dataset.collator import Collator
from utils.utils import get_mean, seed_all

from sklearn.model_selection import train_test_split


def seed_all(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)


def get_mean(lis):
    return sum(lis) / len(lis)


if __name__ == "__main__":

    seed = random.randint(0, 100)
    seed_all(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./data/config.json", help="path to the config.json file", type=str)

    args = parser.parse_args()
    config_file_path = "data/config.json"
    config = CDRConfig.from_json_file(config_file_path)
    corpus = CDRCorpus(config)

    print("Loading vocabs .....")
    corpus.load_all_vocabs(config.saved_folder_path)

    train_dataset = get_cdr_dataset(config.saved_folder_path, "train")
    dev_dataset = get_cdr_dataset(config.saved_folder_path, "dev")
    test_dataset = get_cdr_dataset(config.saved_folder_path, "test")
    collator = Collator(corpus.word_vocab, corpus.pos_vocab, corpus.char_vocab, corpus.rel_vocab)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collator.collate
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collator.collate
    )
    if dev_dataset is not None:
        dev_loader = DataLoader(
            dev_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collator.collate
        )
    weighted = torch.Tensor([1, 3.65]).cuda()
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.001)

    re_criterion = nn.CrossEntropyLoss(weight=weighted)
    if config.use_ner:
        ner_criterion = nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.5)

    train_global_step = 0
    val_global_step = 0

    writer = SummaryWriter()

    best_f1 = -1

    for i in range(config.epochs):

        loss_epoch = []
        val_loss_epoch = []

        train_rel_loss = []
        train_ner_loss = []

        model.train()

        for train_batch in tqdm(train_loader):

            train_global_step += 1
            model.zero_grad()
            batch = [t.cuda() for t in train_batch]

            inputs = batch[:-2]
            ner_label_ids = batch[-2]
            label_ids = batch[-1]

            if config.use_ner:

                ner_logits, re_logits = model(inputs)

                re_loss = re_criterion(re_logits, label_ids)
                ner_loss = ner_criterion(ner_logits.permute(0, 2, 1), ner_label_ids)
                total_loss = re_loss + ner_loss

                total_loss.backward()

                if train_global_step % config.gradient_accumalation == 0:
                    nn.utils.clip_grad_norm(model.parameters(), config.gradient_clipping)
                    optimizer.step()
                    optimizer.zero_grad()

                writer.add_scalar("Loss/train_rel_loss", re_loss.item(), val_global_step)
                writer.add_scalar("Loss/train_ner_loss", ner_loss.item(), val_global_step)

                train_rel_loss.append(re_loss.item())
                train_ner_loss.append(ner_loss.item())

            else:
                re_logits = model(inputs)
                re_loss = re_criterion(re_logits, label_ids)
                re_loss.backward()

                if train_global_step % config.gradient_accumalation == 0:
                    nn.utils.clip_grad_norm(model.parameters(), config.gradient_clipping)
                    optimizer.step()
                    optimizer.zero_grad()

                writer.add_scalar("Loss/train_rel_loss", re_loss.item(), train_global_step)
                train_rel_loss.append(re_loss.item())

        scheduler.step()
        avg_train_rel_loss = get_mean(train_rel_loss)

        if len(train_ner_loss) > 0:
            avg_train_ner_loss = get_mean(train_ner_loss)
            print(f"epoch:{i+1}, train_rel_loss:{avg_train_rel_loss}, train_ner_loss:{avg_train_ner_loss}")

        else:
            print(f"epoch:{i+1}, train_rel_loss: {avg_train_rel_loss}")

        if dev_dataset is not None:

            print("Evaluate on dev set .......")
            model.eval()
            dev_rel_loss = []
            dev_ner_loss = []
            pred_list = []
            target_list = []
            ner_target_list = []
            ner_pred_list = []

            with torch.no_grad():
                for val_batch in tqdm(dev_loader):

                    val_global_step += 1

                    batch = [t.cuda() for t in val_batch]

                    inputs = batch[:-2]
                    ner_label_ids = batch[-2]
                    label_ids = batch[-1]

                    if config.use_ner:

                        ner_logits, re_logits = model(inputs)

                        re_loss = re_criterion(re_logits, label_ids)
                        ner_loss = ner_criterion(ner_logits.permute(0, 2, 1), ner_label_ids)

                        total_loss = re_loss + ner_loss
                        # for rel
                        pred_classes = torch.argmax(re_logits, dim=-1).cpu().data.numpy().tolist()
                        target_classes = label_ids.cpu().data.numpy().tolist()
                        pred_list.extend(pred_classes)
                        target_list.extend(target_classes)

                        # for ner
                        ner_pred_classes = torch.argmax(ner_logits, dim=-1).cpu().data.numpy().tolist()
                        ner_target_classes = ner_label_ids.cpu().data.numpy().tolist()

                        ner_pred_classes = decode_ner(ner_pred_classes)
                        ner_target_classes = decode_ner(ner_target_classes)

                        ner_target_list.extend(ner_target_classes)
                        ner_pred_list.extend(ner_pred_classes)

                        val_loss_epoch.append(total_loss.item())

                        writer.add_scalar("Loss/dev_rel_loss", re_loss.item(), val_global_step)
                        writer.add_scalar("Loss/dev_ner_loss", ner_loss.item(), val_global_step)

                        dev_rel_loss.append(re_loss.item())
                        dev_ner_loss.append(ner_loss.item())
                    else:
                        re_logits = model(inputs)

                        re_loss = re_criterion(re_logits, label_ids)
                        # for rel
                        pred_classes = torch.argmax(re_logits, dim=-1).cpu().data.numpy().tolist()
                        target_classes = label_ids.cpu().data.numpy().tolist()
                        pred_list.extend(pred_classes)
                        target_list.extend(target_classes)

                        val_loss_epoch.append(re_loss.item())
                        writer.add_scalar("Loss/dev_rel_loss", re_loss.item(), val_global_step)
                        dev_rel_loss.append(re_loss.item())

            # avg_train_rel_loss = get_mean(train_rel_loss)
            avg_dev_rel_loss = get_mean(dev_rel_loss)
            if len(dev_ner_loss) > 0:
                avg_dev_ner_loss = get_mean(dev_ner_loss)
                print(f"epoch:{i+1}, dev_rel_loss:{avg_dev_rel_loss}, dev_ner_loss:{avg_dev_ner_loss}")
                ner_f1 = compute_NER_f1_macro(ner_pred_list, ner_target_list)
                print(f"ner f1 score:{ner_f1}")
            else:
                print(f"epoch:{i+1}, dev_rel_loss: {avg_dev_rel_loss}")

            f1 = compute_rel_f1(target_list, pred_list)
            print(f"relation f1 score: {f1}")
            if f1 > best_f1:
                best_f1 = f1
                print("performance improved .... Save best model ...")
                torch.save(model.state_dict(), os.path.join(config.checkpoint_path, model_name))

if config.use_full:
    print("Save model ....")
    torch.save(model.state_dict(), os.path.join(config.checkpoint_path, model_name))

print("Evaluate on test set .......")
# print("Load best checkpoint .....")
# model.load_state_dict(torch.load(f"best_model_{best_f1}.pth"))
model.cuda()

model.eval()
test_rel_loss = []
test_ner_loss = []
pred_list = []
target_list = []
ner_target_list = []
ner_pred_list = []

with torch.no_grad():

    for val_batch in tqdm(test_loader):

        batch = [t.cuda() for t in val_batch]
        inputs = batch[:-2]
        ner_label_ids = batch[-2]
        label_ids = batch[-1]

        if config.use_ner:
            ner_logits, re_logits = model(inputs)
            re_loss = re_criterion(re_logits, label_ids)
            ner_loss = ner_criterion(ner_logits.permute(0, 2, 1), ner_label_ids)
            total_loss = re_loss + ner_loss
            # for rel
            pred_classes = torch.argmax(re_logits, dim=-1).cpu().data.numpy().tolist()
            target_classes = label_ids.cpu().data.numpy().tolist()

            pred_list.extend(pred_classes)
            target_list.extend(target_classes)

            # for ner
            ner_pred_classes = torch.argmax(ner_logits, dim=-1).cpu().data.numpy().tolist()
            ner_target_classes = ner_label_ids.cpu().data.numpy().tolist()

            ner_pred_classes = decode_ner(ner_pred_classes)
            ner_target_classes = decode_ner(ner_target_classes)

            ner_target_list.extend(ner_target_classes)
            ner_pred_list.extend(ner_pred_classes)

            test_rel_loss.append(re_loss.item())
            test_ner_loss.append(ner_loss.item())
        else:
            re_logits = model(inputs)

            re_loss = re_criterion(re_logits, label_ids)
            # for rel
            pred_classes = torch.argmax(re_logits, dim=-1).cpu().data.numpy().tolist()
            target_classes = label_ids.cpu().data.numpy().tolist()
            pred_list.extend(pred_classes)
            target_list.extend(target_classes)

            test_rel_loss.append(re_loss.item())

# avg_train_rel_loss = get_mean(train_rel_loss)
avg_test_rel_loss = get_mean(test_rel_loss)

if len(test_ner_loss) > 0:
    avg_test_ner_loss = get_mean(test_ner_loss)
    print(f"test_rel_loss:{avg_test_rel_loss}, test_ner_loss:{avg_test_ner_loss}")

    ner_f1 = compute_NER_f1_macro(ner_pred_list, ner_target_list)
    print(f"test ner f1 score:{ner_f1}")

else:
    print(f"test_rel_loss: {avg_test_rel_loss}")

p, r, f1, _ = compute_results(pred_list, target_list)
print("Results on test set")
print(f"precision: {p}, recall: {r}, f1: {f1} ")