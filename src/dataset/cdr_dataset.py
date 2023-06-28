from torch.utils.data import Dataset


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
            train_inter
    ):

        super(CDRDataset, self).__init__()

        self.train_inter = train_inter
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

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        label = self.labels[idx]
        pud_id, c_id, d_id, rel = label

        if rel == "CID":
            label_ids = 1
        else:
            label_ids = 0

        if self.train_inter:
            key = label
        else:
            key = pud_id

        token_ids = self.all_doc_token_ids[key]
        in_nodes_idx = self.all_in_nodes_idx[key]
        in_edge_label_ids = self.all_in_edge_label_ids[key]
        out_nodes_idx = self.all_out_nodes_idx[key]
        out_edge_label_ids = self.all_out_edge_label_ids[key]
        pos_ids = self.all_pos_ids[key]
        char_ids = self.all_char_ids[key]
        assert len(char_ids) == len(pos_ids) == len(token_ids)
        chem_entity_map = self.all_entity_mapping[key][c_id]
        dis_entity_map = self.all_entity_mapping[key][d_id]
        ner_label_ids = self.all_ner_label_ids[key]
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
