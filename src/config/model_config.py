from typing import Dict, Any
from config.base_config import BaseConfig


class GraphEncoderConfig(BaseConfig):
    required_arguments = {
        "contextual_word_embedding_dim",
        "word_embedding_dim",
        "char_embedding_dim",
        "edge_embedding_dim",
        "pos_embedding_dim",
        "max_distance_mention_token",
        "distance_embedding_dim",
        "combined_hidden_dim",
        "lstm_hidden_dim",
        "hidden_dim",
        "in_attn_heads",
        "out_attn_heads",
        "use_char", "use_pos", "use_word", "use_distance", "use_state", "use_attn",
        "kernel_size", "n_filters",
        "time_step",
        "drop_out"
    }

    def __init__(self,
                 contextual_word_embedding_dim, word_embedding_dim,
                 char_embedding_dim, edge_embedding_dim, pos_embedding_dim,
                 max_distance_mention_token, distance_embedding_dim,
                 combined_hidden_dim, lstm_hidden_dim, hidden_dim,
                 in_attn_heads, out_attn_heads, use_elmo,
                 use_char, use_pos, use_word, use_distance, use_state, use_attn,
                 kernel_size, n_filters, time_step, drop_out):
        self.use_elmo = use_elmo
        self.contextual_word_embedding_dim = contextual_word_embedding_dim
        self.word_embedding_dim = word_embedding_dim
        self.char_embedding_dim = char_embedding_dim
        self.edge_embedding_dim = edge_embedding_dim
        self.pos_embedding_dim = pos_embedding_dim
        self.max_distance_mention_token = max_distance_mention_token
        self.distance_embedding_dim = distance_embedding_dim
        self.combined_hidden_dim = combined_hidden_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.hidden_dim = hidden_dim
        self.in_attn_heads = in_attn_heads
        self.out_attn_heads = out_attn_heads
        self.use_char = use_char
        self.use_pos = use_pos
        self.use_word = use_word
        self.use_distance = use_distance
        self.use_state = use_state
        self.use_attn = use_attn
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.time_step = time_step
        self.drop_out = drop_out

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GraphEncoderConfig":
        cls.check_required(d)
        config = GraphEncoderConfig(d['contextual_word_embedding_dim'],
                                    d['word_embedding_dim'],
                                    d['char_embedding_dim'],
                                    d['edge_embedding_dim'],
                                    d['pos_embedding_dim'],
                                    d['max_distance_mention_token'],
                                    d['distance_embedding_dim'],
                                    d['combined_hidden_dim'],
                                    d['lstm_hidden_dim'],
                                    d['hidden_dim'],
                                    d['in_attn_heads'],
                                    d['out_attn_heads'],
                                    d.get('use_elmo', True),
                                    d['use_char'], d['use_pos'], d['use_word'],
                                    d['use_distance'], d['use_state'], d['use_attn'],
                                    d['kernel_size'], d['n_filters'],
                                    d['time_step'], d['drop_out'])
        return config


class GraphLSTMConfig(BaseConfig):
    required_arguments = {"encoder",
                          "relation_classes",
                          "ner_classes",
                          "entity_hidden_dim",
                          "max_distance_mention_mention",
                          "distance_embedding_dim",
                          "use_ner",
                          "drop_out",
                          "distance_thresh",
                          "use_distance"}

    def __init__(self, encoder: GraphEncoderConfig,
                 relation_classes,
                 ner_classes,
                 entity_hidden_dim,
                 max_distance_mention_mention,
                 distance_embedding_dim,
                 use_ner,
                 drop_out,
                 distance_thresh,
                 use_distance):
        self.encoder = encoder
        self.relation_classes = relation_classes
        self.ner_classes = ner_classes
        self.entity_hidden_dim = entity_hidden_dim
        self.max_distance_mention_mention = max_distance_mention_mention
        self.distance_embedding_dim = distance_embedding_dim
        self.use_ner = use_ner
        self.drop_out = drop_out
        self.distance_thresh = distance_thresh
        self.use_distance = use_distance

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GraphLSTMConfig":
        cls.check_required(d)
        config = GraphLSTMConfig(GraphEncoderConfig.from_dict(d['encoder']),
                                 d['relation_classes'],
                                 d['ner_classes'],
                                 d['entity_hidden_dim'],
                                 d['max_distance_mention_mention'],
                                 d['distance_embedding_dim'],
                                 d['use_ner'],
                                 d['drop_out'],
                                 d['distance_thresh'],
                                 d['use_distance'])
        return config
