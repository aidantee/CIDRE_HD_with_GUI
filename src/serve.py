import argparse
import os

import gradio as gr
import torch

from config.cdr_config import CDRConfig
from corpus.cdr_corpus import CDRCorpus
from dataset.collator import Collator
from model.cdr_model import GraphStateLSTM

parser = argparse.ArgumentParser()
parser.add_argument("--config", default='./config.json')
parser.add_argument("--predict_threshold", default=0.7, type=float)
parser.add_argument('--model_ckpt_path', type=str, default='checkpoints/cdr_2023_07_28_23_20_14/model.pth')
args = parser.parse_args()

# config_path = "./config.json"
# predict_threshold = 0.7
# model_ckpt_path = 'checkpoints/cdr_2023_07_28_23_20_14/model.pth'

config_path = os.getenv('CONFIG_PATH', './config.json')
predict_threshold = float(os.getenv('PREDICT_THRESHOLD', 0.7))
model_ckpt_path = os.getenv('MODEL_CKPT_PATH', None)

assert model_ckpt_path is not None, "environment variable MODEL_CKPT_PATH must be exported"

config = CDRConfig.from_json(config_path)
corpus = CDRCorpus(config)
corpus.load_all_vocabs(config.data.saved_data_path)
device = 'cpu'
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
collator = Collator(corpus.word_vocab, corpus.pos_vocab,
                    corpus.char_vocab, corpus.rel_vocab)


def convert_features_to_model_inputs(features):
    batch_inputs = [
        (
            features[0],
            features[1],
            features[2],
            features[3],
            features[4],
            features[5],
            features[6],
            features[7][pair[0]],
            features[7][pair[1]]
        )
        for pair in features[-1]
    ]
    batch_inputs = collator.test_collate(batch_inputs)
    return batch_inputs


def relation_extraction(input_docs: str):
    docs = input_docs.split('\n\n')
    res = ''
    for doc in docs:
        lines = doc.split('\n')
        features, pub_id = corpus.convert_one_doc_to_features(lines)
        batch_inputs = convert_features_to_model_inputs(features)
        _, outputs = model(batch_inputs)
        outputs = torch.softmax(outputs, dim=-1)
        predictions = [0 if float(logit[0]) > predict_threshold else 1 for logit in outputs]
        chem_dis_pair_ids = features[-1]
        res += f'{pub_id}\n'
        res += f'chem_id\tdis_id\tcid_relation\n'
        for idx, (chem_id, dis_id) in enumerate(chem_dis_pair_ids):
            res += f'{chem_id}\t{dis_id}\t{predictions[idx]}\n'
        res += '\n'
    return res


demo = gr.Interface(
    fn=relation_extraction,
    inputs=gr.Textbox(lines=2, placeholder="Document Here..."),
    outputs="text",
)
demo.launch()
