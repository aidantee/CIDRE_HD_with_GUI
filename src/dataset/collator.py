import torch
from dataset.utils import pad_sequences, pad_characters, pad_entity, pad_nodes
from dataset.constants import NER_VOCAB
from allennlp.modules.elmo import batch_to_ids


class Collator:
    def __init__(self, word_vocab, pos_vocab, char_vocab, rel_vocab,
                 max_node_in=5,
                 max_node_out=30,
                 max_mentions=30,
                 max_entity_span=20,
                 max_char_length=96,
                 node_pad_idx=4,
                 entity_pad_idx=5,
                 edge_pad_idx=10, max_distance=100):
        self.word_vocab = word_vocab
        self.idx2word = {k: v for v, k in word_vocab.items()}
        self.pos_vocab = pos_vocab
        self.char_vocab = char_vocab
        self.rel_vocab = rel_vocab
        self.ner_vocab = NER_VOCAB
        self.max_node_in = max_node_in
        self.max_node_out = max_node_out
        self.max_mentions = max_mentions
        self.max_entity_span = max_entity_span
        self.max_char_length = max_char_length
        self.node_pad_idx = node_pad_idx
        self.entity_pad_idx = entity_pad_idx
        self.edge_pad_idx = edge_pad_idx
        self.max_distance = max_distance

    def collate(self, batch):
        (
            list_token_ids,
            list_in_nodes_idx,
            list_out_nodes_idx,
            list_in_edge_label_ids,
            list_out_edge_label_ids,
            list_pos_ids,
            list_char_ids,
            list_chem_en_map,
            list_dis_en_map,
            list_ner_label_ids,
            list_label_ids,
        ) = list(zip(*batch))
        #         print(list_chem_en_map)
        #         print(list_dis_en_map)
        batch_size = len(list_token_ids)
        batch_max_length = -1
        for token_ids in list_token_ids:
            batch_max_length = max(len(token_ids), batch_max_length)

        distance_chem = [[self.max_distance - 1 for _ in range(batch_max_length)] for _ in range(batch_size)]
        distance_dis = [[self.max_distance - 1 for _ in range(batch_max_length)] for _ in range(batch_size)]

        for batch_id, chem_en_map in enumerate(list_chem_en_map):
            for mention in chem_en_map:
                for token_id in mention:
                    for i in range(batch_max_length):
                        distance_chem[batch_id][i] = min(distance_chem[batch_id][i], abs(i - token_id))

        for batch_id, dis_en_map in enumerate(list_dis_en_map):
            for mention in dis_en_map:
                for token_id in mention:
                    for i in range(batch_max_length):
                        distance_dis[batch_id][i] = min(distance_dis[batch_id][i], abs(i - token_id))

        distance_chem_tensor = torch.tensor(distance_chem)
        distance_dis_tensor = torch.tensor(distance_dis)

        # ok
        token_ids_tensor, token_ids_mask_tensor = pad_sequences(
            list_token_ids, batch_max_length, self.word_vocab["<PAD>"]
        )
        pos_ids_tensor, _ = pad_sequences(list_pos_ids, batch_max_length, self.pos_vocab["<PAD>"])

        ner_label_ids_tensor, _ = pad_sequences(list_ner_label_ids, batch_max_length, self.ner_vocab["O"])

        # ok
        char_ids_tensor = pad_characters(
            list_char_ids, batch_max_length, self.max_char_length, self.char_vocab["<PAD>"]
        )

        list_texts = []
        for token_ids in list_token_ids:
            text = []
            for token_id in token_ids:
                text.append(self.idx2word[token_id])
            list_texts.append(text)
        elmo_input_tensor = batch_to_ids(list_texts)
        # ok
        in_nodes_idx_tensor, in_nodes_mask_tensor = pad_nodes(
            list_in_nodes_idx, batch_max_length, self.max_node_in, self.node_pad_idx
        )
        out_nodes_idx_tensor, out_nodes_mask_tensor = pad_nodes(
            list_out_nodes_idx, batch_max_length, self.max_node_out, self.node_pad_idx
        )

        # ok
        chem_entity_map_tensor, chem_entity_map_mask_tensor = pad_entity(
            list_chem_en_map, self.max_mentions, self.max_entity_span, self.entity_pad_idx
        )

        # ok
        dis_entity_map_tensor, dis_entity_map_mask_tensor = pad_entity(
            list_dis_en_map, self.max_mentions, self.max_entity_span, self.entity_pad_idx
        )

        in_edge_idx_tensor, in_edge_idx_mask = pad_nodes(
            list_in_edge_label_ids, batch_max_length, self.max_node_in, self.rel_vocab["<PAD>"]
        )
        out_edge_idx_tensor, out_edge_idx_mask = pad_nodes(
            list_out_edge_label_ids, batch_max_length, self.max_node_out, self.rel_vocab["<PAD>"]
        )

        chem_start_idx = chem_entity_map_tensor[..., 0]
        dis_start_idx = dis_entity_map_tensor[..., 0]

        start_distant = torch.abs(
            chem_start_idx.unsqueeze(2).repeat(1, 1, self.max_mentions)
            - dis_start_idx.unsqueeze(1).repeat(1, self.max_mentions, 1)
        )

        label_ids_tensor = torch.tensor(list_label_ids)
        return (
            token_ids_tensor,
            token_ids_mask_tensor,
            pos_ids_tensor,
            char_ids_tensor,
            in_nodes_idx_tensor,
            in_nodes_mask_tensor,
            out_nodes_idx_tensor,
            out_nodes_mask_tensor,
            in_edge_idx_tensor,
            in_edge_idx_mask,
            out_edge_idx_tensor,
            out_edge_idx_mask,
            chem_entity_map_tensor,
            chem_entity_map_mask_tensor,
            dis_entity_map_tensor,
            dis_entity_map_mask_tensor,
            start_distant,
            elmo_input_tensor,
            distance_chem_tensor,
            distance_dis_tensor,
            ner_label_ids_tensor,
            label_ids_tensor,
        )

    def test_collate(self, batch):
        (
            list_token_ids,
            list_in_nodes_idx,
            list_out_nodes_idx,
            list_in_edge_label_ids,
            list_out_edge_label_ids,
            list_pos_ids,
            list_char_ids,
            list_chem_en_map,
            list_dis_en_map,
        ) = list(zip(*batch))
        #         print(list_chem_en_map)
        #         print(list_dis_en_map)
        batch_size = len(list_token_ids)
        batch_max_length = -1
        for token_ids in list_token_ids:
            batch_max_length = max(len(token_ids), batch_max_length)

        distance_chem = [[self.max_distance - 1 for _ in range(batch_max_length)] for _ in range(batch_size)]
        distance_dis = [[self.max_distance - 1 for _ in range(batch_max_length)] for _ in range(batch_size)]

        for batch_id, chem_en_map in enumerate(list_chem_en_map):
            for mention in chem_en_map:
                for token_id in mention:
                    for i in range(batch_max_length):
                        distance_chem[batch_id][i] = min(distance_chem[batch_id][i], abs(i - token_id))

        for batch_id, dis_en_map in enumerate(list_dis_en_map):
            for mention in dis_en_map:
                for token_id in mention:
                    for i in range(batch_max_length):
                        distance_dis[batch_id][i] = min(distance_dis[batch_id][i], abs(i - token_id))

        distance_chem_tensor = torch.tensor(distance_chem)
        distance_dis_tensor = torch.tensor(distance_dis)

        # ok
        token_ids_tensor, token_ids_mask_tensor = pad_sequences(
            list_token_ids, batch_max_length, self.word_vocab["<PAD>"]
        )
        pos_ids_tensor, _ = pad_sequences(list_pos_ids, batch_max_length, self.pos_vocab["<PAD>"])

        # ok
        char_ids_tensor = pad_characters(
            list_char_ids, batch_max_length, self.max_char_length, self.char_vocab["<PAD>"]
        )

        list_texts = []
        for token_ids in list_token_ids:
            text = []
            for token_id in token_ids:
                text.append(self.idx2word[token_id])
            list_texts.append(text)
        elmo_input_tensor = batch_to_ids(list_texts)
        # ok
        in_nodes_idx_tensor, in_nodes_mask_tensor = pad_nodes(
            list_in_nodes_idx, batch_max_length, self.max_node_in, self.node_pad_idx
        )
        out_nodes_idx_tensor, out_nodes_mask_tensor = pad_nodes(
            list_out_nodes_idx, batch_max_length, self.max_node_out, self.node_pad_idx
        )

        # ok
        chem_entity_map_tensor, chem_entity_map_mask_tensor = pad_entity(
            list_chem_en_map, self.max_mentions, self.max_entity_span, self.entity_pad_idx
        )

        # ok
        dis_entity_map_tensor, dis_entity_map_mask_tensor = pad_entity(
            list_dis_en_map, self.max_mentions, self.max_entity_span, self.entity_pad_idx
        )

        in_edge_idx_tensor, in_edge_idx_mask = pad_nodes(
            list_in_edge_label_ids, batch_max_length, self.max_node_in, self.rel_vocab["<PAD>"]
        )
        out_edge_idx_tensor, out_edge_idx_mask = pad_nodes(
            list_out_edge_label_ids, batch_max_length, self.max_node_out, self.rel_vocab["<PAD>"]
        )

        chem_start_idx = chem_entity_map_tensor[..., 0]
        dis_start_idx = dis_entity_map_tensor[..., 0]

        start_distant = torch.abs(
            chem_start_idx.unsqueeze(2).repeat(1, 1, self.max_mentions)
            - dis_start_idx.unsqueeze(1).repeat(1, self.max_mentions, 1)
        )

        return (
            token_ids_tensor,
            token_ids_mask_tensor,
            pos_ids_tensor,
            char_ids_tensor,
            in_nodes_idx_tensor,
            in_nodes_mask_tensor,
            out_nodes_idx_tensor,
            out_nodes_mask_tensor,
            in_edge_idx_tensor,
            in_edge_idx_mask,
            out_edge_idx_tensor,
            out_edge_idx_mask,
            chem_entity_map_tensor,
            chem_entity_map_mask_tensor,
            dis_entity_map_tensor,
            dis_entity_map_mask_tensor,
            start_distant,
            elmo_input_tensor,
            distance_chem_tensor,
            distance_dis_tensor,
        )