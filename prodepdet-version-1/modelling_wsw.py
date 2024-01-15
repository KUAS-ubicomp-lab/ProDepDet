# coding=utf-8
import transformers
import torch
import torch.nn as nn


class WSWEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.segment_embeddings = nn.Embedding(config.max_segment_embeddings, config.hidden_size)
        self.speaker_embeddings = nn.Embedding(config.max_speaker_embeddings, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids, segment_ids, and speaker_ids are contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.register_buffer("segment_ids", torch.arange(config.max_segment_embeddings).expand((1, -1)))
        self.register_buffer("speaker_ids", torch.arange(config.max_speaker_embeddings).expand((1, -1)))

        # End copy
        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )
        self.segment_embeddings = nn.Embedding(
            config.max_segment_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )
        self.speaker_embeddings = nn.Embedding(
            config.max_speaker_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

        # Soft prompt embeddings
        self.prompt_embeddings = nn.Embedding(config.prompt_num, config.hidden_size)
        tokenizer = transformers.BertTokenizer.from_pretrained('bert-base')
        print(tokenizer.encode(["<mask>"], add_special_tokens=False))
        print(type(tokenizer.encode(["<mask>"], add_special_tokens=False)))
        self.prompt_embeddings = nn.Embedding(int(config.prompt_num), int(config.hidden_size))
        self._init_weights(self.prompt_embeddings)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def init_prompt_emb(self, init_ids):
        prompt_weights = self.word_embeddings(init_ids).detach()
        self.prompt_embeddings.weight.data = prompt_weights
        print("init_prompt_done, check_if_requires_grad")
