# coding=utf-8
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.models.bert.modeling_bert import (
    BertEmbeddings
)
from transformers import AutoModelForMaskedLM
from typing import Optional

from .base import PromptEmbeddings


class PromptWSWEmbeddings(PromptEmbeddings, BertEmbeddings):
    def __init__(self, config):
        super().__init__(config)

    def forward(
            self, input_ids=None, token_type_ids=None, position_ids=None, segment_ids=None, speaker_ids=None,
            inputs_embeds=None, past_key_values_length=0
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        batch_size, seq_length = input_shape

        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length: seq_length + past_key_values_length]

        if segment_ids is None:
            segment_ids = self.segment_ids[:, past_key_values_length: seq_length + past_key_values_length]

        if speaker_ids is None:
            speaker_ids = self.speaker_ids[:, past_key_values_length: seq_length + past_key_values_length]

        if inputs_embeds is None:
            # Sentence embeddings
            select_word = (input_ids >= 0).int()
            word_embeds = self.word_embeddings(input_ids * select_word) * select_word.unsqueeze(2)

            # Prompt embeddings
            select_prompt = (input_ids < 0).int()
            prompt_ids = - (input_ids + 1)
            prompt_embs = self.prompt_embeddings(prompt_ids * select_prompt) * select_prompt.unsqueeze(2)

            # Sum to get input_embed
            inputs_embeds = word_embeds + prompt_embs
            inputs_embeds = self.prepend_prompt_embeddings(inputs_embeds, self.word_embeddings)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            position_embeddings = self.prepend_position_embeddings(position_embeddings)
            segment_embeddings = self.segment_embeddings(segment_ids)
            segment_embeddings = self.prepend_segment_embeddings(segment_embeddings)
            speaker_embeddings = self.speaker_embeddings(speaker_ids)
            speaker_embeddings = self.prepend_speaker_embeddings(speaker_embeddings)
            embeddings += position_embeddings, segment_embeddings, speaker_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class PromptWSW(nn.Module):
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
            segment_ids: Optional[torch.Tensor] = None,
            speaker_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            verbalizer=None
    ):

        # Extend attention_mask and token_type_ids
        if verbalizer is None:
            verbalizer = {0: 4997, 1: 3893}
        attention_mask, token_type_ids = self.prepare_attention_mask(attention_mask, token_type_ids)

        lm_output = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            segment_ids=segment_ids,
            speaker_ids=speaker_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )['logits']
        mask_logits = lm_output[:, 0]

        # Select scores based on verbalizer
        logits = []
        for label_mapped_token_id in verbalizer.values():
            logits.append(mask_logits[:, label_mapped_token_id].unsqueeze(1))
        logits = torch.cat([mask_logits[:, 4997].unsqueeze(1), mask_logits[:, 3893].unsqueeze(1)], dim=1)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "binary_classification"
                else:
                    self.config.problem_type = "multi_class_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "binary_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_class_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        return loss, logits
