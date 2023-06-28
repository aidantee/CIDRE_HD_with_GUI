from corpus.cdr_corpus import CDRCorpus
from dataset.cdr_dataset import CDRDataset
import torch
from torch import Tensor
from typing import List, Tuple, Dict


def pad_sequences(list_token_ids, max_length, pad_idx) -> Tuple[Tensor, Tensor]:
    padded_list_token_ids = []
    padded_seq_mask = []
    for token_ids in list_token_ids:
        pad_length = max_length - len(token_ids)
        token_masked = [1] * len(token_ids) + [0] * pad_length
        padded_token_ids = token_ids + [pad_idx] * pad_length
        padded_list_token_ids.append(padded_token_ids)
        padded_seq_mask.append(token_masked)
    padded_list_token_ids = torch.tensor(padded_list_token_ids)
    padded_seq_mask = torch.tensor(padded_seq_mask)
    return padded_list_token_ids, padded_seq_mask


def pad_nodes(batch_list_nodes_idx, max_length, max_node, pad_idx=15):

    padded_list_nodes = []
    padded_nodes_masked = []
    # loop over batch of sentences
    for list_nodes_ids in batch_list_nodes_idx:

        sent_node_ids = []
        sent_nodes_masked = []

        for node_ids in list_nodes_ids:
            pad_lenght = max_node - len(node_ids)
            node_masked = [1] * len(node_ids) + [0] * pad_lenght
            padded_node_ids = node_ids + [pad_idx] * pad_lenght
            sent_node_ids.append(padded_node_ids)
            sent_nodes_masked.append(node_masked)

        pad_sent_length = max_length - len(list_nodes_ids)

        padded_node_ids = [[pad_idx] * max_node] * pad_sent_length
        padded_node_masked = [[0] * max_node] * pad_sent_length

        padded_sent_node_ids = sent_node_ids + padded_node_ids
        padded_sent_nodes_masked = sent_nodes_masked + padded_node_masked

        padded_list_nodes.append(padded_sent_node_ids)
        padded_nodes_masked.append(padded_sent_nodes_masked)
    padded_list_nodes = torch.tensor(padded_list_nodes)
    padded_nodes_masked = torch.tensor(padded_nodes_masked)
    return padded_list_nodes, padded_nodes_masked


def pad_entity(batch_list_entity_map, max_mentions, max_entity_span, pad_idx=15):

    padded_entities = []
    padded_entities_span_masked = []

    for sent_entity_map in batch_list_entity_map:

        padded_sent_entity_map = []
        padded_sent_entity_span_masked = []

        for mention in sent_entity_map:
            pad_length = max_entity_span - len(mention)

            if pad_length >= 0:
                padded_mention = mention + [pad_idx] * pad_length
                padded_masked = [1] * len(mention) + [0] * pad_length
            else:
                padded_masked = [1] * len(mention[:max_entity_span])
                padded_mention = mention[:max_entity_span]

            padded_sent_entity_map.append(padded_mention)
            padded_sent_entity_span_masked.append(padded_masked)

        pad_entity_mention = max_mentions - len(sent_entity_map)
        padded_sent_entity_span_masked = padded_sent_entity_span_masked + [[0] * max_entity_span] * pad_entity_mention
        padded_sent_entity_map = padded_sent_entity_map + [[pad_idx] * max_entity_span] * pad_entity_mention

        padded_entities.append(padded_sent_entity_map)
        padded_entities_span_masked.append(padded_sent_entity_span_masked)
    padded_entities = torch.tensor(padded_entities)
    padded_entities_span_masked = torch.tensor(padded_entities_span_masked)
    return padded_entities, padded_entities_span_masked


def pad_characters(list_char_ids, batch_max_length, max_char_length, char_pad_idx):
    padded_char_ids = []
    for doc_char_ids in list_char_ids:
        pad_length = batch_max_length - len(doc_char_ids)
        doc_padded_char_ids = []
        for token_chars in doc_char_ids:
            char_pad_length = max_char_length - len(token_chars)
            pad = [char_pad_idx] * char_pad_length
            padded_token_chars = token_chars + pad

            assert len(padded_token_chars) == max_char_length
            doc_padded_char_ids.append(padded_token_chars)
        padded_batch_chars = [[char_pad_idx] * max_char_length] * pad_length
        doc_padded_char_ids = doc_padded_char_ids + padded_batch_chars
        padded_char_ids.append(doc_padded_char_ids)
    padded_char_ids = torch.tensor(padded_char_ids)
    return padded_char_ids


def pad_tensor(list_tensor, batch_length):
    list_padded_tensor = []
    for tensor in list_tensor:
        pad_length = batch_length - tensor.shape[1]
        pad_tensor = torch.zeros(size=(pad_length, tensor.shape[-1]))
        padded_tensor = torch.cat([tensor.squeeze(0), pad_tensor], dim=0)

        assert padded_tensor.shape[0] == batch_length

        list_padded_tensor.append(padded_tensor)
    return torch.stack(list_padded_tensor, dim=0).detach()


def get_cdr_dataset(corpus: CDRCorpus, saved_folder_path: str, data_type: str, train_inter) -> CDRDataset:
    (
        all_doc_token_ids,
        all_in_nodes_idx,
        all_out_nodes_idx,
        all_in_edge_label_ids,
        all_out_edge_label_ids,
        all_doc_pos_ids,
        all_doc_char_ids,
        all_entity_mapping,
        all_ner_label_ids,
        all_labels,
    ) = corpus.load_all_features_for_one_dataset(saved_folder_path, data_type)
    dataset = CDRDataset(all_doc_token_ids,
                         all_in_nodes_idx,
                         all_out_nodes_idx,
                         all_in_edge_label_ids,
                         all_out_edge_label_ids,
                         all_doc_pos_ids,
                         all_doc_char_ids,
                         all_entity_mapping,
                         all_ner_label_ids,
                         all_labels,
                         train_inter)
    return dataset


def concat_dataset(datasets: List[CDRDataset]) -> CDRDataset:
    train_inter = datasets[0].train_inter
    res = CDRDataset(
        {k: v for dataset in datasets for k, v in dataset.all_doc_token_ids.items()},
        {k: v for dataset in datasets for k, v in dataset.all_in_nodes_idx.items()},
        {k: v for dataset in datasets for k, v in dataset.all_out_nodes_idx.items()},
        {k: v for dataset in datasets for k, v in dataset.all_in_edge_label_ids.items()},
        {k: v for dataset in datasets for k, v in dataset.all_out_edge_label_ids.items()},
        {k: v for dataset in datasets for k, v in dataset.all_pos_ids.items()},
        {k: v for dataset in datasets for k, v in dataset.all_char_ids.items()},
        {k: v for dataset in datasets for k, v in dataset.all_entity_mapping.items()},
        {k: v for dataset in datasets for k, v in dataset.all_ner_label_ids.items()},
        [label for dataset in datasets for label in dataset.labels],
        train_inter
    )
    return res


def split_train_test(dataset: CDRDataset, train_pud_ids: List[str]):
    train_dataset = CDRDataset(
        {k: v for k, v in dataset.all_doc_token_ids.items() if k in train_pud_ids},
        {k: v for k, v in dataset.all_in_nodes_idx.items() if k in train_pud_ids},
        {k: v for k, v in dataset.all_out_nodes_idx.items() if k in train_pud_ids},
        {k: v for k, v in dataset.all_in_edge_label_ids.items() if k in train_pud_ids},
        {k: v for k, v in dataset.all_out_edge_label_ids.items() if k in train_pud_ids},
        {k: v for k, v in dataset.all_pos_ids.items() if k in train_pud_ids},
        {k: v for k, v in dataset.all_char_ids.items() if k in train_pud_ids},
        {k: v for k, v in dataset.all_entity_mapping.items() if k in train_pud_ids},
        {k: v for k, v in dataset.all_ner_label_ids.items() if k in train_pud_ids},
        [label for label in dataset.labels if label[0] in train_pud_ids]
    )
    test_data = CDRDataset(
        {k: v for k, v in dataset.all_doc_token_ids.items() if k not in train_pud_ids},
        {k: v for k, v in dataset.all_in_nodes_idx.items() if k not in train_pud_ids},
        {k: v for k, v in dataset.all_out_nodes_idx.items() if k not in train_pud_ids},
        {k: v for k, v in dataset.all_in_edge_label_ids.items() if k not in train_pud_ids},
        {k: v for k, v in dataset.all_out_edge_label_ids.items() if k not in train_pud_ids},
        {k: v for k, v in dataset.all_pos_ids.items() if k not in train_pud_ids},
        {k: v for k, v in dataset.all_char_ids.items() if k not in train_pud_ids},
        {k: v for k, v in dataset.all_entity_mapping.items() if k not in train_pud_ids},
        {k: v for k, v in dataset.all_ner_label_ids.items() if k not in train_pud_ids},
        [label for label in dataset.labels if label[0] not in train_pud_ids]
    )
    return train_dataset, test_data
