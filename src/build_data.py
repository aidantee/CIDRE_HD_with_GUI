from config.cdr_config import CDRConfig
from tqdm import tqdm
from dataset.cdr_dataset import CDRDataset
from corpus.cdr_corpus import CDRCorpus
import argparse
import pickle
import os


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="data/config.json", help="path to the config.json file", type=str)
    args = parser.parse_args()
    config = CDRConfig.from_json_file(args.config)
    corpus = CDRCorpus(config)

    # if you still dont have the vocabs for the dataset. You need to call this method firstly.
    print("Preparing all vocabs .....")
    corpus.prepare_all_vocabs(config.saved_folder_path)
    print("Preparing all data ....")
    corpus.prepare_features_for_one_dataset(
        config.train_file_path, config.saved_folder_path, "train"
    )
    corpus.prepare_features_for_one_dataset(
        config.test_file_path, config.saved_folder_path, "test"
    )
    corpus.prepare_features_for_one_dataset(
        config.dev_file_path, config.saved_folder_path, "dev"
    )
