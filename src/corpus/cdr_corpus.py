"""This module describe how we prepare the training, development and testing dataset from Biocreative CDR5 corpus."""
import codecs
import itertools
import json
import os
import pickle
import random
from collections import defaultdict
from itertools import groupby
from typing import Any, Dict, List, Tuple, Union
import numpy as np
import copy
from config.cdr_config import CDRConfig
from utils.constants import CHEMICAL_STRING, DISEASE_STRING
from utils.spacy_utils import nlp

ner_vocab = {"O": 0, "B_Chemical": 1, "I_Chemical": 2, "B_Disease": 3, "I_Disease": 4}
ner_idx2label = {0: "O", 1: "B_Chemical", 2: "I_Chemical", 3: "B_Disease", 4: "I_Disease"}
# idx2word = {k: v for v, k in word_vocab.items()}
ADJACENCY_REL = "node"
ROOT_REL = "root"
SELF_REL = "self"
SOME_SPECIFIC_MENTIONS = [
    "tyrosine-methionine-aspartate-aspartate",
    "anemia/thrombocytopenia/emesis/rash",
    "metoprolol/alpha-hydroxymetoprolol",
    "glutamate/N-methyl-D-aspartate",
    "cyclosporine-and-prednisone-treated",
    "platinum/paclitaxel-refractory",
]
LabelAnnotationType = Dict[Tuple[Any, Any, Any], Any]
DocAnnotationType = Dict[Any, List[Union[List[List[Any]], Any, Any]]]


class CDRCorpus:
    def __init__(self, config: CDRConfig):
        """[summary]

        Args:
            config ([type]): [description]
        """
        self.config = config
        self.list_feature_names = [
            "all_doc_token_ids.pkl",
            "all_in_nodes_idx.pkl",
            "all_out_nodes_idx.pkl",
            "all_in_edge_label_ids.pkl",
            "all_out_edge_label_ids.pkl",
            "all_doc_pos_ids.pkl",
            "all_doc_char_ids.pkl",
            "all_entity_mapping.pkl",
            "all_ner_labels.pkl",
            "labels.pkl",
        ]
        self.word_vocab = dict()
        self.char_vocab = dict()
        self.rel_vocab = dict()
        self.pos_vocab = dict()
        self.list_vocab_names = ["word_vocab.json", "rel_vocab.json", "pos_vocab.json", "char_vocab.json"]

    def load_vocab(self, file_path):
        with open(file_path) as f:
            vocab = json.load(f)
            return vocab

    def save_vocab(self, vocab, file_path):
        with open(file_path, "w") as f:
            json.dump(vocab, f)

    def load_tensor(self, tensor_file_path):
        with open(tensor_file_path, "rb") as f:
            tensor = pickle.load(f)
            return tensor

    def load_numpy(self, numpy_file_path):
        with open(numpy_file_path, "rb") as f:
            matrix = np.load(f)
            return matrix

    def save_feature(self, feature, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(feature, f)

    def load_feature(self, file_path):
        with open(file_path, "rb") as f:
            feature = pickle.load(f)
        return feature

    def load_all_vocabs(self, saved_folder_path):
        if os.path.exists(os.path.join(saved_folder_path, self.list_vocab_names[0])):
            self.word_vocab = self.load_vocab(os.path.join(saved_folder_path, self.list_vocab_names[0]))
            self.rel_vocab = self.load_vocab(os.path.join(saved_folder_path, self.list_vocab_names[1]))
            self.pos_vocab = self.load_vocab(os.path.join(saved_folder_path, self.list_vocab_names[2]))
            self.char_vocab = self.load_vocab(os.path.join(saved_folder_path, self.list_vocab_names[3]))
        else:
            raise Exception(
                "You have not prepared the vocabs. Please prepare them and the features by running the build_data scipt"
            )

    def load_all_features_for_one_dataset(self, saved_folder_path, data_type):
        list_features = []
        for feature_name in self.list_feature_names:
            feature = self.load_feature(os.path.join(saved_folder_path, data_type, feature_name))
            list_features.append(feature)
        return list_features

    def prepare_features_for_one_dataset(self, data_file_path, saved_folder_path, data_type, train_inter):
        if os.path.exists(os.path.join(saved_folder_path, self.list_vocab_names[0])):
            self.word_vocab = self.load_vocab(os.path.join(saved_folder_path, self.list_vocab_names[0]))
            self.rel_vocab = self.load_vocab(os.path.join(saved_folder_path, self.list_vocab_names[1]))
            self.pos_vocab = self.load_vocab(os.path.join(saved_folder_path, self.list_vocab_names[2]))
            self.char_vocab = self.load_vocab(os.path.join(saved_folder_path, self.list_vocab_names[3]))
        else:
            raise Exception("Feature vocabs not found. Please call prepare_all_vocabs firstly  .......")

        if train_inter:
            in_adjacency_dict, out_adjacency_dict, entity_mapping_dict, labels = self.process_inter_dataset(
                data_file_path
            )
        else:
            in_adjacency_dict, out_adjacency_dict, entity_mapping_dict, labels = self.process_dataset(data_file_path)
        features = self.convert_examples_to_features(
            in_adjacency_dict,
            out_adjacency_dict,
            entity_mapping_dict,
            self.word_vocab,
            self.rel_vocab,
            self.pos_vocab,
            self.char_vocab,
        )
        features = list(features)
        features.append(labels)

        print("Saving generated features .......")

        for feature_name, feature in list(zip(self.list_feature_names, features)):
            self.save_feature(feature, os.path.join(saved_folder_path, data_type, feature_name))

    def prepare_all_vocabs(self, saved_folder_path, train_inter) -> None:
        """[summary]

        Returns:
            [type]: [description]
        """
        if not train_inter:
            (
                train_in_adjacency_dict,
                train_out_adjacency_dict,
                train_entity_mapping_dict,
                train_all_labels,
            ) = self.process_dataset(self.config.data.train_file_path)
            (
                dev_in_adjacency_dict,
                dev_out_adjacency_dict,
                dev_entity_mapping_dict,
                dev_all_labels,
            ) = self.process_dataset(self.config.data.dev_file_path)
            (
                test_in_adjacency_dict,
                test_out_adjacency_dict,
                test_entity_mapping_dict,
                test_all_labels,
            ) = self.process_dataset(self.config.data.test_file_path)
        else:
            (
                train_in_adjacency_dict,
                train_out_adjacency_dict,
                train_entity_mapping_dict,
                train_all_labels,
            ) = self.process_inter_dataset(self.config.data.train_file_path)
            (
                dev_in_adjacency_dict,
                dev_out_adjacency_dict,
                dev_entity_mapping_dict,
                dev_all_labels,
            ) = self.process_inter_dataset(self.config.data.dev_file_path)
            (
                test_in_adjacency_dict,
                test_out_adjacency_dict,
                test_entity_mapping_dict,
                test_all_labels,
            ) = self.process_inter_dataset(self.config.data.test_file_path)
        print("Saving vocabs .......")
        vocabs = self.create_vocabs([train_in_adjacency_dict, dev_in_adjacency_dict, test_in_adjacency_dict])
        for vocab_name, vocab in list(zip(self.list_vocab_names, vocabs)):
            self.save_vocab(vocab, os.path.join(saved_folder_path, vocab_name))

    def make_pairs(self, entity_annotations: List[Tuple[Any, Any, Any, Any, Any]]) -> List[Tuple[Any, Any]]:
        """[summary]

        Args:
            entity_annotations (List[Tuple[Any, Any, Any, Any, Any]]): [description]

        Returns:
            List[Tuple[Any, Any]]: [description]
        """
        chem_entity_ids = [anno[-1] for anno in entity_annotations if anno[-2] == CHEMICAL_STRING]
        dis_entity_ids = [anno[-1] for anno in entity_annotations if anno[-2] == DISEASE_STRING]

        chem_entity_ids = list(set(chem_entity_ids))
        dis_entity_ids = list(set(dis_entity_ids))

        chem_dis_pair_ids = list(itertools.product(chem_entity_ids, dis_entity_ids))

        return chem_dis_pair_ids

    def get_valid_entity_mentions(
        self, entity_mentions_annotations: List[Tuple[Any, Any, Any, Any, Any]], invalid_id: str = "-1"
    ) -> List[Tuple[Any, Any, Any, Any, Any]]:
        """Remove all entity which has unknown id.

        Args:
            entity_mentions_annotations (List[Tuple[Any, Any, Any, Any, Any]]): list of entity mention annotations,
            whose each element is a tuple of (start_offset, end_offset, text, entity type, mesh_id).
            invalid_id (int, optional): The unknown entity id from CDR5. Defaults to '-1'.

        Returns:
            [type]: [description]
        """

        # remove entity anno in document's title and entity with id = -1
        return [mention_anno for mention_anno in entity_mentions_annotations if mention_anno[-1] != invalid_id]

    def remove_entity_mention_in_title(
        self, entity_mentions_annotations: List[Tuple[Any, Any, Any, Any, Any]], title
    ) -> List[Tuple[Any, Any, Any, Any, Any]]:
        """[summary]

        Args:
            entity_mentions_annotations (List[Tuple[Any, Any, Any, Any, Any]]): [description]
            title ([type]): [description]

        Returns:
            List[Tuple[Any, Any, Any, Any, Any]]: [description]
        """
        return [mention_anno for mention_anno in entity_mentions_annotations if int(mention_anno[1]) >= len(title)]

    def read_raw_dataset(self, file_path: str) -> Tuple[LabelAnnotationType, DocAnnotationType]:
        """Read the raw biocreative CDR5 dataset

        Args:
            file_path (str): path to the dataset

        Returns:
            Tuple[Label_annotation_type, Doc_annotation_type]: A tuple of two dictionary, the label
            annotation whose each key is a tuple of (chemical_mesh_id, disease_mesh_id, document_id)
            and its value is the relation (eg: CID or None)
            the document annotation contains key, values pairs with key is the document id
            and value is a list whose elements are the document title, abstract and
            list of entity mention annotations respectively.
        """

        with open(file_path) as f_raw:
            lines = f_raw.read().split("\n")

            raw_doc_annotations = [list(group) for k, group in groupby(lines, lambda x: x == "") if not k]
            label_annotations = {}
            doc_annotations = {}

            for doc_annos in raw_doc_annotations:

                title = None
                abstract = None
                current_annotations = []

                for anno in doc_annos:

                    if "|t|" in anno:
                        pud_id, title = anno.strip().split("|t|")
                    elif "|a|" in anno:
                        pub_id, abstract = anno.strip().split("|a|")
                    else:
                        splits = anno.strip().split("\t")
                        if len(splits) == 4:
                            _, rel, e1_id, e2_id = splits
                            label_annotations[(e1_id, e2_id, pud_id)] = rel
                        elif len(splits) == 6:
                            _, start, end, mention, label, kg_ids = splits
                            for kg_id in kg_ids.split("|"):
                                current_annotations.append([int(start), int(end), mention, label, kg_id])
                        elif len(splits) == 7:
                            _, start, end, mention, label, kg_ids, split_mentions = splits
                            for kg_id in kg_ids.split("|"):
                                current_annotations.append([int(start), int(end), mention, label, kg_id])

                assert title is not None and abstract is not None
                doc_annotations[pud_id] = [title, abstract, current_annotations]

            return label_annotations, doc_annotations

    def create_sentence_adjacency_dict(self, sent, debug=True):
        """[summary]

        Args:
            sent ([type]): [description]
            debug (bool, optional): [description]. Defaults to True.

        Returns:
            [type]: [description]
        """
        in_sent_adjacency_dict = {}
        out_sent_adjacency_dict = {}

        root = sent.root

        assert root is not None

        # if debug == True:
        #     svg = displacy.render(sent, style="dep", jupyter=True, options={"collapse_punct": False})

        # dependency rel
        for token in sent:
            out_sent_adjacency_dict[token] = [(token, SELF_REL)]
            # out adjacency dict of this token appends all the token's childrens and their dependency relation
            for child in token.children:
                out_sent_adjacency_dict[token].append((child, child.dep_))
            # in adjacency dict of this tokens append itself, its head and its dependency relation

            if token != root:
                in_sent_adjacency_dict[token] = [(token, SELF_REL), (token.head, token.dep_)]
                # in_sent_adjacency_dict[token] = [(token.head, token.dep_)]
            else:
                in_sent_adjacency_dict[token] = [(token, SELF_REL)]
                # in_sent_adjacency_dict[token] = []

        return in_sent_adjacency_dict, out_sent_adjacency_dict, root

    def create_document_adjacency_dict(
        self, doc, list_in_sent_adjacency_dict, list_out_sent_adjacency_dict, list_root
    ):

        in_doc_adjacency_dict = {}
        out_doc_adjacency_dict = {}
        # create in and out adjacency dict for document

        for token in doc:
            for in_sent_adjacency_dict in list_in_sent_adjacency_dict:
                if token in in_sent_adjacency_dict:
                    # the token in current sentence
                    in_doc_adjacency_dict[token] = in_sent_adjacency_dict[token]

            for out_sent_adjacency_dict in list_out_sent_adjacency_dict:
                if token in out_sent_adjacency_dict:
                    # the token in current sentence
                    out_doc_adjacency_dict[token] = out_sent_adjacency_dict[token]

        list_tokens = list(doc)
        # extra rel between adjacency tokens
        for i in range(1, len(list_tokens) - 1):
            in_doc_adjacency_dict[list_tokens[i]].append((list_tokens[i - 1], ADJACENCY_REL))
            out_doc_adjacency_dict[list_tokens[i]].append((list_tokens[i + 1], ADJACENCY_REL))

        in_doc_adjacency_dict[list_tokens[-1]].append((list_tokens[-2], ADJACENCY_REL))
        out_doc_adjacency_dict[list_tokens[0]].append((list_tokens[1], ADJACENCY_REL))

        # connect root of adjacency sentences, i.e document with more than 2 sentences
        if len(list_root) >= 2:
            for i in range(1, len(list_root) - 1):
                in_doc_adjacency_dict[list_root[i]].append((list_root[i - 1], ROOT_REL))
                out_doc_adjacency_dict[list_root[i]].append((list_root[i + 1], ROOT_REL))

            in_doc_adjacency_dict[list_root[-1]].append((list_root[-2], ROOT_REL))
            out_doc_adjacency_dict[list_root[0]].append((list_root[1], ROOT_REL))

        # check
        for token, nodes in out_doc_adjacency_dict.items():
            assert len(nodes) != 31

        # visualize_dependency_graph(in_doc_adjacency_dict)
        return in_doc_adjacency_dict, out_doc_adjacency_dict

    def create_features_one_doc(self, pud_id, abstract, entity_annotations, offset_span=20, debug=False):
        """[summary]

        Args:
            pud_id ([type]): [description]
            abstract ([type]): [description]
            entity_annotations ([type]): [description]
            offset_span (int, optional): [description]. Defaults to 20.
            debug (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: [description]
        """
        # sentence tokenize
        doc = nlp(abstract)

        list_in_sentence_adjacency_dicts = []
        list_out_sentence_adjacency_dicts = []
        list_roots = []
        for sent in doc.sents:

            # create adjacency list for sentence and find the root of the sentence.
            in_sent_adjacency_dict, out_sent_adjacency_dict, sent_root = self.create_sentence_adjacency_dict(
                sent, debug=debug
            )
            list_in_sentence_adjacency_dicts.append(in_sent_adjacency_dict)
            list_out_sentence_adjacency_dicts.append(out_sent_adjacency_dict)
            list_roots.append(sent_root)

        # merge all sentence denpendency to create document dependency tree.
        in_doc_adjacency_dict, out_doc_adjacency_dict = self.create_document_adjacency_dict(
            doc, list_in_sentence_adjacency_dicts, list_out_sentence_adjacency_dicts, list_roots
        )

        # mapping entity spans to document ids
        entity_mapping = {}
        n_sents = len(list(doc.sents))
        has_dis_mask = [False for i in range(n_sents)]
        has_chem_mask = [False for i in range(n_sents)]
        for en_anno in entity_annotations:
            start, end, mention, label, kg_id = en_anno
            key = (start, end, mention, label, kg_id)
            entity_mapping[key] = []
            current_sent = 0
            for token in doc:
                token_start = token.idx
                token_end = token_start + len(token)
                if token_start >= start and token_end <= end:
                    entity_mapping[key].append(token)
                    if label == "Chemical":
                        has_chem_mask[current_sent] = True
                    else:
                        has_dis_mask[current_sent] = True
                if token.is_sent_end:
                    current_sent += 1
            if len(entity_mapping[key]) == 0:
                current_sent = 0
                for token in doc:
                    token_start = token.idx
                    token_end = token_start + len(token)
                    if token_start <= start and token_end >= end and mention in token.text:
                        entity_mapping[key].append(token)
                        if label == "Chemical":
                            has_chem_mask[current_sent] = True
                        else:
                            has_dis_mask[current_sent] = True
                        break
                    if token.is_sent_end:
                        current_sent += 1

        for i in range(n_sents):
            if has_chem_mask[i] and has_dis_mask[i]:
                with open("invalid_pud_id.txt", "a") as outfile:
                    outfile.write(pud_id + "\n")
                break

            try:
                assert entity_mapping[key] != []
            except:
                print(en_anno)
                print(pud_id)
                print(abstract[start - 50 : end + 50])

        return in_doc_adjacency_dict, out_doc_adjacency_dict, entity_mapping, doc

    def preprocess_one_doc(self, pud_id, title, abstract, entity_annotations, debug=False):
        """[summary]

        Args:
            pud_id ([type]): [description]
            title ([type]): [description]
            abstract ([type]): [description]
            entity_annotations ([type]): [description]
            debug (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: [description]
        """
        # remove all annotations of invalid enity (i.e entity id equals -1)
        entity_annotations = self.get_valid_entity_mentions(entity_annotations)
        if not self.config.data.use_title:
            raise Exception("Warning is not using title")
            # subtract title offset plus one space
            for en_anno in entity_annotations:
                en_anno[0] -= len(title) + 1
                en_anno[1] -= len(title) + 1
            # remove entity mention in the title
            entity_annotations = self.remove_entity_mention_in_title(entity_annotations, title)

        # make all pairs chemical disease entities
        chem_dis_pair_ids = self.make_pairs(entity_annotations)
        # create doc_adjacency_dict and entity_to_tokens_mapping

        in_doc_adjacency_dict, out_doc_adjacency_dict, entity_mapping, doc = self.create_features_one_doc(
            pud_id, title + " " + abstract if self.config.data.use_title else abstract, entity_annotations
        )

        # print(chem_dis_pair_ids)
        return chem_dis_pair_ids, in_doc_adjacency_dict, out_doc_adjacency_dict, entity_mapping, doc

    def process_dataset(self, file_path, mesh_filtering=True):
        """[summary]

        Args:
            file_path ([type]): [description]

        Returns:
            [type]: [description]
        """
        label_annotations, doc_annotations = self.read_raw_dataset(file_path)

        label_docs = defaultdict(list)
        in_adjacency_dict = {}
        entity_mapping_dict = {}
        out_adjacency_dict = {}

        max_doc_length = -1

        # process all document
        for pud_id, doc_anno in doc_annotations.items():
            title, abstract, entity_annotations = doc_anno
            (
                chem_dis_pair_ids,
                in_doc_adjacency_dict,
                out_doc_adjacency_dict,
                entity_mapping,
                _,
            ) = self.preprocess_one_doc(pud_id, title, abstract, entity_annotations)
            label_docs[pud_id] = chem_dis_pair_ids
            in_adjacency_dict[pud_id] = in_doc_adjacency_dict
            out_adjacency_dict[pud_id] = out_doc_adjacency_dict
            entity_mapping_dict[pud_id] = entity_mapping

            max_doc_length = max(len(in_doc_adjacency_dict), max_doc_length)

        # gather positive examples and negative examples
        pos_doc_examples = defaultdict(list)
        neg_doc_examples = defaultdict(list)

        unfilterd_positive_count = 0
        unfilterd_negative_count = 0

        for pud_id in doc_annotations.keys():
            for c_e, d_e in label_docs[pud_id]:
                if (c_e, d_e, pud_id) in label_annotations:
                    pos_doc_examples[pud_id].append((c_e, d_e))
                    unfilterd_positive_count += 1
                else:
                    neg_doc_examples[pud_id].append((c_e, d_e))
                    unfilterd_negative_count += 1

        print("original number of positive samples: ", unfilterd_positive_count)
        print("original number of negative samples: ", unfilterd_negative_count)
        print("max document length: ", max_doc_length)

        if self.config.data.mesh_filtering:
            ent_tree_map = defaultdict(list)
            with codecs.open(self.config.data.mesh_path, "r", encoding="utf-16-le") as f:
                lines = [l.rstrip().split("\t") for i, l in enumerate(f) if i > 0]
                [ent_tree_map[l[1]].append(l[0]) for l in lines]
                neg_doc_examples, n_filterd_samples = self.filter_with_mesh_vocab(
                    ent_tree_map, pos_doc_examples, neg_doc_examples
                )

            print("number of negative examples are filterd:", n_filterd_samples)

        all_labels = []
        for pud_id, value in pos_doc_examples.items():
            for c_id, d_id in value:
                key = (pud_id, c_id, d_id, "CID")
                all_labels.append(key)

        for pud_id, value in neg_doc_examples.items():
            for c_id, d_id in value:
                key = (pud_id, c_id, d_id, "NULL")
                all_labels.append(key)

        random.shuffle(all_labels)
        print("total samples: ", len(all_labels))
        return in_adjacency_dict, out_adjacency_dict, entity_mapping_dict, all_labels

    def filter_with_mesh_vocab(self, mesh_tree, pos_doc_examples, neg_doc_examples):
        """[summary]

        Args:
            mesh_tree ([type]): [description]
            pos_doc_examples ([type]): [description]
            neg_doc_examples ([type]): [description]

        Returns:
            [type]: [description]
        """
        neg_filterd_exampled = defaultdict(list)
        n_filterd_samples = 0
        negative_count = 0
        hypo_count = 0
        # i borrowed this code from https://github.com/patverga/bran/blob/master/src/processing/utils/filter_hypernyms.py
        for doc_id in neg_doc_examples.keys():
            # get nodes for all the positive diseases
            pos_e2_examples = [(pos_node, pe) for pe in pos_doc_examples[doc_id] for pos_node in mesh_tree[pe[1]]]
            # chemical
            pos_e1_examples = [(pos_node, pe) for pe in pos_doc_examples[doc_id] for pos_node in mesh_tree[pe[0]]]

            for ne in neg_doc_examples[doc_id]:
                neg_e1 = ne[0]
                neg_e2 = ne[1]
                example_hyponyms = 0
                for neg_node in mesh_tree[ne[1]]:
                    hyponyms = [
                        pos_node for pos_node, pe in pos_e2_examples if neg_node in pos_node and neg_e1 == pe[0]
                    ]
                    example_hyponyms += len(hyponyms)
                if example_hyponyms == 0:
                    negative_count += 1
                    neg_filterd_exampled[doc_id].append((neg_e1, neg_e2))
                else:
                    hypo_count += example_hyponyms
                    n_filterd_samples += 1

        return neg_filterd_exampled, n_filterd_samples


    def get_spacy_pos_tag_from_wordnet(self, treebank_tag):
        if treebank_tag.startswith("j") or treebank_tag.startswith("s"):
            return "ADJ"
        elif treebank_tag.startswith("v"):
            return "VEB"
        elif treebank_tag.startswith("n"):
            return "NOUN"
        elif treebank_tag.startswith("r"):
            return "ADV"
        else:
            return ""

    def convert_tokens_to_ids(self, list_tokens, vocab):
        token_ids = []
        for token in list_tokens:
            if token not in vocab:
                token_ids.append(vocab["<UNK>"])
            else:
                token_ids.append(vocab[token])
        return token_ids

    def create_vocabs(self, list_adjacency_dict, min_freq=1):

        list_words = []
        list_rels = []
        list_poses = []
        list_chars = []

        for adjacency_dict in list_adjacency_dict:
            for pud_id in adjacency_dict.keys():
                for token, value in adjacency_dict[pud_id].items():
                    list_words.append(token.text)
                    list_poses.append(token.tag_)
                    list_chars.extend(list(token.text))
                    for t, rel in value:
                        list_rels.append(rel)

        word_vocab = list(set(list_words))
        word_vocab.append("<UNK>")
        word_vocab.append("<PAD>")

        word_vocab = {value: key for key, value in enumerate(word_vocab)}

        rel_vocab = list(set(list_rels))
        rel_vocab.append("<PAD>")
        rel_vocab = {value: key for key, value in enumerate(rel_vocab)}

        pos_vocab = list(set(list_poses))
        pos_vocab.append("<PAD>")
        pos_vocab = {value: key for key, value in enumerate(pos_vocab)}

        char_vocab = list(set(list_chars))
        char_vocab.append("<PAD>")
        char_vocab = {value: key for key, value in enumerate(char_vocab)}

        print(f"word vocab: {len(word_vocab)} unique words")
        print(f"dependency rel vocab: {len(rel_vocab)} unique relations")
        print(f"char vocab: {len(char_vocab)} unique characters")

        return word_vocab, rel_vocab, pos_vocab, char_vocab

    def create_in_out_features_for_doc(
        self, in_adjacency_dict, out_adjacency_dict, doc_tokens, rel_vocab, pos_vocab, char_vocab
    ):

        all_in_nodes_idx = []
        all_out_nodes_idx = []
        all_in_edge_label_ids = []
        all_out_edge_label_ids = []
        all_poses_ids = []
        all_char_ids = []

        max_node_in = -1
        max_node_out = -1
        max_char_length = -1

        for token in doc_tokens:
            all_poses_ids.append(pos_vocab[token.tag_])
            char_ids = [char_vocab[c] for c in token.text]
            max_char_length = max(max_char_length, len(char_ids))
            all_char_ids.append(char_ids)
        # print(out_adjacency_dict)

        # create features for incoming nodes
        for key, in_adjacent in in_adjacency_dict.items():
            key_in_ids = []
            in_edge_label_ids = []

            for token, rel in in_adjacent:
                key_in_ids.append(doc_tokens.index(token))
                in_edge_label_ids.append(rel_vocab[rel])

            max_node_in = max(max_node_in, len(key_in_ids))

            # print(key_in_ids)
            # assert 1== 0

            assert len(key_in_ids) == len(in_edge_label_ids)

            all_in_nodes_idx.append(key_in_ids)
            all_in_edge_label_ids.append(in_edge_label_ids)

        # create features for outgoing nodes
        for key, out_adjacent in out_adjacency_dict.items():

            key_out_ids = []
            out_edge_label_ids = []

            for token, rel in out_adjacent:
                key_out_ids.append(doc_tokens.index(token))
                out_edge_label_ids.append(rel_vocab[rel])

            # print(key_out_ids)

            assert len(key_out_ids) == len(out_edge_label_ids)

            max_node_out = max(max_node_out, len(key_out_ids))
            all_out_nodes_idx.append(key_out_ids)
            all_out_edge_label_ids.append(out_edge_label_ids)

        return (
            all_in_nodes_idx,
            all_out_nodes_idx,
            all_in_edge_label_ids,
            all_out_edge_label_ids,
            all_poses_ids,
            all_char_ids,
            max_node_in,
            max_node_out,
            max_char_length,
        )

    def convert_examples_to_features(
        self,
        in_adjacency_dicts,
        out_adjacency_dicts,
        entity_mapping_dicts,
        word_vocab,
        rel_vocab,
        pos_vocab,
        char_vocab,
    ):

        all_in_nodes_idx = {}
        all_out_nodes_idx = {}
        all_in_edge_label_ids = {}
        all_out_edge_label_ids = {}
        all_doc_token_ids = {}
        all_doc_pos_ids = {}
        all_doc_char_ids = {}
        all_doc_hypernym_ids = {}
        all_doc_synonym_ids = {}

        all_enitty_mapping = {}

        max_node_in = -1
        max_node_out = -1
        max_char_length = -1

        for pud_id, in_doc_adjacency_dict in in_adjacency_dicts.items():

            doc_tokens = list(in_doc_adjacency_dict.keys())

            doc_token_texts = [tok.text for tok in doc_tokens]
            out_doc_adjacency_dict = out_adjacency_dicts[pud_id]

            (
                doc_in_nodes_idx,
                doc_out_nodes_idx,
                doc_in_edge_label_ids,
                doc_out_edge_label_ids,
                doc_poses_ids,
                doc_char_ids,
                max_doc_node_in,
                max_doc_node_out,
                max_doc_char_length,
            ) = self.create_in_out_features_for_doc(
                in_doc_adjacency_dict, out_doc_adjacency_dict, doc_tokens, rel_vocab, pos_vocab, char_vocab
            )

            doc_token_ids = self.convert_tokens_to_ids(doc_token_texts, word_vocab)

            all_doc_token_ids[pud_id] = doc_token_ids
            all_in_nodes_idx[pud_id] = doc_in_nodes_idx
            all_out_nodes_idx[pud_id] = doc_out_nodes_idx
            all_in_edge_label_ids[pud_id] = doc_in_edge_label_ids
            all_out_edge_label_ids[pud_id] = doc_out_edge_label_ids
            all_doc_pos_ids[pud_id] = doc_poses_ids
            all_doc_char_ids[pud_id] = doc_char_ids

            max_node_in = max(max_node_in, max_doc_node_in)
            max_node_out = max(max_node_out, max_doc_node_out)
            max_char_length = max(max_doc_char_length, max_char_length)

        max_entity_span = -1
        max_mentions = -1
        all_ner_labels = {}

        for pud_id, in_doc_adjacency_dict in in_adjacency_dicts.items():

            doc_tokens = list(in_doc_adjacency_dict.keys())
            entitty_to_tokens = defaultdict(list)

            ner_label = []

            for token in doc_tokens:
                ner_label.append(ner_vocab["O"])

            for key, mapping_tokens in entity_mapping_dicts[pud_id].items():

                _, _, _, en_type, en_id = key
                list_idx_mention = []

                count = 0
                for token in mapping_tokens:
                    list_idx_mention.append(doc_tokens.index(token))
                    max_entity_span = max(max_entity_span, len(list_idx_mention))
                    if count == 0:
                        tag = "B_" + en_type
                        ner_label[doc_tokens.index(token)] = ner_vocab[tag]
                        count += 1
                    else:
                        tag = "I_" + en_type
                        ner_label[doc_tokens.index(token)] = ner_vocab[tag]

                assert len(list_idx_mention) != 0
                entitty_to_tokens[en_id].append(list_idx_mention)

                assert len(entitty_to_tokens[en_id]) != 0
                max_mentions = max(max_mentions, len(entitty_to_tokens[en_id]))

            all_enitty_mapping[pud_id] = entitty_to_tokens
            all_ner_labels[pud_id] = ner_label
            # assert 1==0

        print("max node in: ", max_node_in)
        print("max node out: ", max_node_out)
        print("max entity spans: ", max_entity_span)
        print("max entity mentions: ", max_mentions)
        print("max characters length: ", max_char_length)

        return (
            all_doc_token_ids,
            all_in_nodes_idx,
            all_out_nodes_idx,
            all_in_edge_label_ids,
            all_out_edge_label_ids,
            all_doc_pos_ids,
            all_doc_char_ids,
            all_enitty_mapping,
            all_ner_labels,
        )

    def process_inter_dataset(self, file_path, mesh_path=None, use_log=True):

        label_annotations, doc_annotations = self.read_raw_dataset(file_path)
        temp = copy.deepcopy(doc_annotations)
        label_docs = defaultdict(list)
        in_adjacency_dict = {}
        entity_mapping_dict = {}
        out_adjacency_dict = {}
        docs_dict = {}
        max_doc_length = -1

        # process all document
        for pud_id, doc_anno in doc_annotations.items():
            title, abstract, entity_annotations = doc_anno
            (
                chem_dis_pair_ids,
                in_doc_adjacency_dict,
                out_doc_adjacency_dict,
                entity_mapping,
                doc,
            ) = self.preprocess_one_doc(pud_id, title, abstract, entity_annotations)
            label_docs[pud_id] = chem_dis_pair_ids
            in_adjacency_dict[pud_id] = in_doc_adjacency_dict
            out_adjacency_dict[pud_id] = out_doc_adjacency_dict
            entity_mapping_dict[pud_id] = entity_mapping

            # use this to create intra sentence relation
            docs_dict[pud_id] = doc
            max_doc_length = max(len(in_doc_adjacency_dict), max_doc_length)

        # gather positive examples and negative examples
        pos_doc_examples = defaultdict(list)
        neg_doc_examples = defaultdict(list)

        unfilterd_positive_count = 0
        unfilterd_negative_count = 0

        for pud_id in doc_annotations.keys():
            for c_e, d_e in label_docs[pud_id]:
                if (c_e, d_e, pud_id) in label_annotations:
                    pos_doc_examples[pud_id].append((c_e, d_e))
                    unfilterd_positive_count += 1
                else:
                    neg_doc_examples[pud_id].append((c_e, d_e))
                    unfilterd_negative_count += 1

        print("original number positive samples: ", unfilterd_positive_count)
        print("original number negative samples: ", unfilterd_negative_count)
        print("max document length: ", max_doc_length)

        if self.config.data.mesh_filtering:
            ent_tree_map = defaultdict(list)
            with codecs.open(self.config.data.mesh_path, "r", encoding="utf-16-le") as f:
                lines = [l.rstrip().split("\t") for i, l in enumerate(f) if i > 0]
                [ent_tree_map[l[1]].append(l[0]) for l in lines]
                neg_doc_examples, n_filterd_samples = self.filter_with_mesh_vocab(
                    ent_tree_map, pos_doc_examples, neg_doc_examples
                )
            print("number negative examples is filterd:", n_filterd_samples)

        all_labels = []

        for pud_id, value in pos_doc_examples.items():
            for c_id, d_id in value:
                key = (pud_id, c_id, d_id, "CID")
                all_labels.append(key)

        for pud_id, value in neg_doc_examples.items():
            for c_id, d_id in value:
                key = (pud_id, c_id, d_id, "NULL")
                all_labels.append(key)

        pos_count = 0
        neg_count = 0

        inter_labels = []
        inter_abstract_labels = []

        inter_in_adjacency_dict = {}
        inter_out_adjacency_dict = {}
        inter_entity_mapping_dict = {}

        max_sub_abstract = -1

        neg_abstract_label = 0
        pos_abstract_label = 0

        t1 = 0
        t2 = 0

        with open("log.txt", "w") as f:
            for label in all_labels:
                # print(label)
                # label = ('17111419', 'D002125', 'D006996', 'NULL')
                # en_annos = [[225, 236, 'Apomorphine', 'Chemical', 'D001058'], [253, 269, 'dopamine agonist', 'Chemical', 'D018491'],
                doc_id, chem_id, dis_id, l = label
                doc_anno = copy.deepcopy(temp[doc_id])
                title, abstract, entity_annotations = doc_anno
                # in_adjacency = in_adjacency_dict[doc_id]
                # out_adjacency = out_adjacency_dict[doc_id]

                entity_annotations = self.get_valid_entity_mentions(entity_annotations)
                # print(len(entity_annotations))
                if not self.config.data.use_title:
                    print("in here")
                    # subtract title offset plus one space
                    for en_anno in entity_annotations:
                        en_anno[0] -= len(title) + 1
                        en_anno[1] -= len(title) + 1
                    # remove entity mention in the title
                    entity_annotations = self.remove_entity_mention_in_title(entity_annotations)

                doc_t = docs_dict[doc_id]
                # get all mentions of chemical and disease
                chem_annos = []
                dis_annos = []

                for en_anno in entity_annotations:
                    if chem_id == en_anno[-1]:
                        chem_annos.append(en_anno)
                    elif dis_id == en_anno[-1]:
                        dis_annos.append(en_anno)

                combine_mentions = list(itertools.product(chem_annos, dis_annos))
                check = False
                check_sents = {}
                chem_count = 0
                dis_count = 0

                for sent_idx, sent in enumerate(doc_t.sents):
                    check_sents[sent_idx] = {}
                    check_sents[sent_idx]["en_annos"] = []
                    check_sents[sent_idx]["check"] = False

                    is_sent_contains_another_mention_pair = False
                    for pair_mentions in combine_mentions:
                        chem_mention, dis_mention = pair_mentions

                        if (
                                not self.is_sent_contains_more_than_2_mentions_of_different_entity_type(
                                    sent, chem_mention, dis_mention
                                )
                                and not is_sent_contains_another_mention_pair
                        ):
                            if self.is_in_sent(sent, chem_mention) or self.is_in_sent(sent, dis_mention):
                                if self.is_in_sent(sent, chem_mention):
                                    if self.is_sent_contains_another_mention(sent, dis_mention, dis_annos):
                                        continue
                                if self.is_in_sent(sent, dis_mention):
                                    if self.is_sent_contains_another_mention(sent, chem_mention, chem_annos):
                                        continue
                                    # print(sent.text, chem_mention, dis_mention)
                                    # do some thing
                                if (
                                        self.is_in_sent(sent, chem_mention)
                                        and chem_mention not in check_sents[sent_idx]["en_annos"]
                                ):
                                    check_sents[sent_idx]["en_annos"].append(chem_mention)
                                    chem_count += 1

                                if (
                                        self.is_in_sent(sent, dis_mention)
                                        and dis_mention not in check_sents[sent_idx]["en_annos"]
                                ):
                                    check_sents[sent_idx]["en_annos"].append(dis_mention)
                                    dis_count += 1
                                try:
                                    # assert dis_mention[2] in sent.text
                                    assert 1 == 1
                                except:
                                    print(sent)
                                    print(chem_mention)
                                    print(dis_mention)
                                    assert 1 == 0
                                check_sents[sent_idx]["check"] = True
                        else:
                            is_sent_contains_another_mention_pair = True

                    if check_sents[sent_idx]["check"] == True:
                        check_sents[sent_idx]["sent"] = sent

                if chem_count > 0 and dis_count > 0:

                    list_in_sentence_adjacency_dicts = []
                    list_out_sentence_adjacency_dicts = []

                    list_roots = []
                    sub_abstract = []
                    # intra_labels.append(label)

                    for sent_idx in check_sents.keys():
                        if check_sents[sent_idx]["check"] == True:
                            sent = check_sents[sent_idx]["sent"]
                            (
                                in_sent_adjacency_dict,
                                out_sent_adjacency_dict,
                                sent_root,
                            ) = self.create_sentence_adjacency_dict(sent, debug=False)
                            list_in_sentence_adjacency_dicts.append(in_sent_adjacency_dict)
                            list_out_sentence_adjacency_dicts.append(out_sent_adjacency_dict)
                            list_roots.append(sent_root)
                            for token in sent:
                                sub_abstract.append(token)

                    in_doc_adjacency_dict, out_doc_adjacency_dict = self.create_document_adjacency_dict(
                        sub_abstract, list_in_sentence_adjacency_dicts, list_out_sentence_adjacency_dicts, list_roots
                    )

                    if use_log:
                        doc_abstracts = ""
                        for sent_idx in check_sents.keys():
                            if check_sents[sent_idx]["check"]:
                                sent_text = check_sents[sent_idx]["sent"].text
                                for anno in check_sents[sent_idx]["en_annos"]:
                                    if anno[2] in sent_text:
                                        sent_text = sent_text.replace(anno[2], "[" + anno[2] + "]")
                                        doc_abstracts = doc_abstracts + " " + sent_text
                                    # else:
                                    #     print(anno[2])
                                    #     print(sent_text)
                                    #     print(label)
                                    #     assert 1==0

                        # print(doc_abstracts)

                        f.write(doc_abstracts + "\n")
                        f.write(str(label) + "\n")
                        f.write(f"[{chem_annos[0][2]}-{dis_annos[0][2]}]" + "\n")

                    if label[-1] == "CID":
                        t1 += 1
                    elif label[-1] == "NULL":
                        t2 += 1

                    inter_labels.append(label)

                    assert len(list_in_sentence_adjacency_dicts) == len(list_out_sentence_adjacency_dicts)
                    max_sub_abstract = max(max_sub_abstract, len(sub_abstract))
                    # print(len(list_in_sentence_adjacency_dicts))
                    # assert 1==0

                    entity_mapping = {}
                    offset_span = 20
                    # count = -1

                    for sent_idx in check_sents.keys():
                        if check_sents[sent_idx]["check"] == True:
                            sent_label = (label[0], label[1], label[2], label[3], sent_idx)

                            # print(list(in_sent_adjacency_dict.keys()))
                            for en_anno in check_sents[sent_idx]["en_annos"]:

                                start, end, mention, t, kg_id = en_anno
                                key = (start, end, mention, t, kg_id)
                                entity_mapping[key] = []

                                for token, adjacen in in_doc_adjacency_dict.items():
                                    token_start = token.idx
                                    token_end = token_start + len(token)

                                    # print(f'mention start: {start}, mention end: {end}, token:{token.text},  token_start: {token_start}, token_end: {token_end}')

                                    if token_start >= start and token_end <= end:
                                        entity_mapping[key].append(token)
                                    # some annotations which form abc-#mention-abcxyz, so we extra token spans to a predefined threshhold.
                                    elif (
                                            token_start >= start - offset_span and token_end <= end + offset_span
                                    ) and mention in token.text:
                                        entity_mapping[key].append(token)
                                    # hard code for some specific mention
                                    elif token.text in SOME_SPECIFIC_MENTIONS:
                                        entity_mapping[key].append(token)
                                try:
                                    assert entity_mapping[key] != []
                                except:
                                    print(check_sents[sent_idx]["en_annos"])
                                    print(check_sents[sent_idx]["sent"].start_char)
                                    print(check_sents[sent_idx]["sent"].end_char)
                                    print(check_sents[sent_idx]["sent"])
                                    for token in check_sents[sent_idx]["sent"]:
                                        print("token: ", token)
                                        print("start idx: ", token.idx)
                                        print("end idx: ", token.idx + len(token))
                                    print(in_doc_adjacency_dict)
                                    print(en_anno)
                                    print(pud_id)
                                    print(sub_abstract)
                                    print("*****************")
                                    assert 1 == 0

                                # assert 1==0

                        # print(entity_mapping)
                        # assert 1==0

                    # label = ('17111419', 'D002125', 'D006996', 'NULL')
                    # sent_label = (label[0], label[1], label[2], label[3], sent_idx)

                    inter_in_adjacency_dict[label[0]] = in_doc_adjacency_dict
                    inter_out_adjacency_dict[label[0]] = out_doc_adjacency_dict
                    inter_entity_mapping_dict[label[0]] = entity_mapping

                    # abstract_label = sent_label[:-1]
                    # intra_abstract_labels.append(abstract_label)

                    # intra_dep_graph[sent_label] = list_dep_graph
                    # count +=1
                    if label[-1] == "CID":
                        pos_count += 1
                    elif label[-1] == "NULL":
                        neg_count += 1

        print("positive count: ", pos_count)
        print("negative count: ", neg_count)

        print("t1: ", t1)
        print("t2: ", t2)

        print("max sub abstract length: ", max_sub_abstract)

        # print(all_labels[:10])

        # assert 1==0
        # print('total samples: ',len(all_labels))
        print("inter samples: ", len(inter_labels))

        inter_abstract_labels = set(inter_abstract_labels)
        # print('max sent length: ',max_sent_length)
        # print('number of abstract label: ',len(intra_abstract_labels))

        # print(intra_in_adjacency_dict)

        return inter_in_adjacency_dict, inter_out_adjacency_dict, inter_entity_mapping_dict, inter_labels

    def is_sent_contains_more_than_2_mentions_of_different_entity_type(self, sent, chem_mention, dis_mention):

        chem_entity_in_sent = chem_mention[0] >= sent.start_char and chem_mention[1] <= sent.end_char
        dis_entity_in_sent = dis_mention[0] >= sent.start_char and dis_mention[1] <= sent.end_char
        return chem_entity_in_sent and dis_entity_in_sent

    def is_in_sent(self, sent, mention):
        in_sent = mention[0] >= sent.start_char and mention[1] <= sent.end_char
        return in_sent

    def is_sent_contains_another_mention(self, sent, mention, annos):
        check = False
        for m in annos:
            if m == mention:
                continue
            check = check or self.is_in_sent(sent, m)
        return check