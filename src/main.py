import argparse

from corpus.cdr_corpus import CDRCorpus
from dataset.cdr_dataset import CDRDataset
from config.cdr_config import CDRConfig
from dataset.utils import get_cdr_dataset, concat_dataset
from dataset.collator import Collator
from torch.utils.data import DataLoader
from model.trainer import Trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./data/config.json")
    args = parser.parse_args()
    config = CDRConfig.from_json_file(args.config)
    corpus = CDRCorpus(config)
    corpus.load_all_vocabs(config.saved_folder_path)
    train_dataset = get_cdr_dataset(corpus, config.saved_folder_path, "train")
    dev_dataset = get_cdr_dataset(corpus, config.saved_folder_path, "dev")
    test_dataset = get_cdr_dataset(corpus, config.saved_folder_path, "test")
    collator = Collator(corpus.word_vocab, corpus.pos_vocab, corpus.char_vocab, corpus.rel_vocab)
    train_dataset = concat_dataset([train_dataset, dev_dataset])
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collator.collate
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collator.collate
    )
    # device = "cuda"
    device = "cpu"
    trainer = Trainer(corpus, config, device)
    trainer.train(train_loader)
