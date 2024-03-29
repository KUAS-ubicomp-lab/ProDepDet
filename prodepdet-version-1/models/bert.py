from typing import Optional

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import AutoModelForMaskedLM
from transformers.models.bert.modeling_bert import (
    BertEmbeddings
)
from transformers.utils import logging

from .base import PromptEmbeddings, PromptModel

logger = logging.get_logger(__name__)


class PromptBertEmbeddings(PromptEmbeddings, BertEmbeddings):
    def __init__(self, config):
        super().__init__(config)

    def forward(self,
                input_ids: Optional[torch.LongTensor] = None,
                token_type_ids: Optional[torch.LongTensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                past_key_values_length: int = 0,
                ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        batch_size, seq_length = input_shape

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length: seq_length + past_key_values_length]

        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
            inputs_embeds = self.prepare_prompt_embeddings(inputs_embeds, self.word_embeddings)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            position_embeddings = self.prepare_position_embeddings(position_embeddings)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class PromptBert(PromptModel):
    def __init__(self, args, num_labels):
        super().__init__()
        self.backbone = AutoModelForMaskedLM.from_pretrained(args.backbone)
        self.config = self.backbone.config
        self.backbone.bert.embeddings = nn.Embedding(self.config.prompt_len, self.config.hidden_size)
        self.num_labels = num_labels

        for k, v in vars(args).items():
            setattr(self.config, k, v)

        self.config.mask_id = 103
        self.init_prompt_emb([self.config.mask_id] * self.config.prompt_len)

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            verbalizer: dict = {0: 4997, 1: 3893}
    ):

        # Extend attention_mask and token_type_ids
        attention_mask, token_type_ids = self.prepare_attention_mask(attention_mask, token_type_ids)

        lm_output = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )['logits']  # [batch_size, seq_length, vocab_size]
        mask_logits = lm_output[:, 0]  # [batch_size, vocab_size], select the logits of <mask>, which is the 1st token
        logits = torch.cat([mask_logits[:, 4997].unsqueeze(1), mask_logits[:, 3893].unsqueeze(1)], dim=1)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        return loss, logits
