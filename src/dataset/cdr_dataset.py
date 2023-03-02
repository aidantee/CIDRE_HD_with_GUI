from torch.utils.data import Dataset
from dataset.constansts import NER_VOCAB


class CDRDataset(Dataset):
    def __init__(
            self,
            all_doc_token_ids,
            all_in_nodes_idx,
            all_out_nodes_idx,
            all_in_edge_label_ids,
            all_out_edge_label_ids,
            all_pos_ids,
            all_char_ids,
            all_entity_mapping,
            all_ner_label_ids,
            labels,
            word_vocab,
            char_vocab,
            rel_vocab,
            pos_vocab
    ):

        super(CDRDataset, self).__init__()

        self.pos_vocab = pos_vocab
        self.rel_vocab = rel_vocab
        self.char_vocab = char_vocab
        self.word_vocab = word_vocab
        self.idx2word = {k: v for v, k in word_vocab.items()}
        self.all_doc_token_ids = all_doc_token_ids

        self.all_in_nodes_idx = all_in_nodes_idx
        self.all_in_edge_label_ids = all_in_edge_label_ids

        self.all_out_nodes_idx = all_out_nodes_idx
        self.all_out_edge_label_ids = all_out_edge_label_ids
        self.all_pos_ids = all_pos_ids
        self.all_char_ids = all_char_ids

        self.all_ner_label_ids = all_ner_label_ids
        self.labels = labels
        self.all_entity_mapping = all_entity_mapping
        # self.all_flair = all_flair
        self.ner_vocab = NER_VOCAB

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        label = self.labels[idx]
        pud_id, c_id, d_id, rel = label

        if rel == "CID":
            label_ids = 1
        else:
            label_ids = 0

        token_ids = self.all_doc_token_ids[pud_id]
        in_nodes_idx = self.all_in_nodes_idx[pud_id]
        in_edge_label_ids = self.all_in_edge_label_ids[pud_id]
        out_nodes_idx = self.all_out_nodes_idx[pud_id]
        out_edge_label_ids = self.all_out_edge_label_ids[pud_id]
        pos_ids = self.all_pos_ids[pud_id]
        char_ids = self.all_char_ids[pud_id]
        assert len(char_ids) == len(pos_ids) == len(token_ids)
        chem_entity_map = self.all_entity_mapping[pud_id][c_id]
        dis_entity_map = self.all_entity_mapping[pud_id][d_id]
        ner_label_ids = self.all_ner_label_ids[pud_id]
        return (
            token_ids,
            in_nodes_idx,
            out_nodes_idx,
            in_edge_label_ids,
            out_edge_label_ids,
            pos_ids,
            char_ids,
            chem_entity_map,
            dis_entity_map,
            ner_label_ids,
            label_ids,
        )
