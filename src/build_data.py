from config.cdr_config import CDRConfig
from tqdm import tqdm
from dataset.cdr_dataset import CDRDataset
from corpus.cdr_corpus import CDRCorpus
import argparse
import pickle
import os


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument("--config", default=str, help="path to the config.json file", type=str)
    parser.add_argument("--type", default=str, help="type of dataset", type=str)

    args = parser.parse_args()
    config_file_path = "config.json"
    config = CDRConfig.from_json(config_file_path)
    corpus = CDRCorpus(config)

    # if you still dont have the vocabs for the dataset. You need to call this method firstly.
    print("Preparing all vocabs .....")
    corpus.prepare_all_vocabs(config.data.saved_data_path)

    print("Preparing all data ....")
    corpus.prepare_features_for_one_dataset(
        config.data.train_file_path, config.data.saved_data_path, "train"
    )
    corpus.prepare_features_for_one_dataset(
        config.data.test_file_path, config.data.saved_data_path, "test"
    )
    corpus.prepare_features_for_one_dataset(
        config.data.dev_file_path, config.data.saved_data_path, "dev"
    )
