from typing import *

import torch
from openprompt.data_utils import InputFeatures
from openprompt.prompts import MixedTemplate
from torch import nn
from transformers import PreTrainedModel, PreTrainedTokenizer


class GenerateTemplate(MixedTemplate):
    registered_inputflag_names = ["soft_token_ids", "loss_ids", 'shortenable_ids']

    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 text: Optional[List[str]] = None,
                 prompt_encoder_type: str = "transformer",
                 placeholder_mapping: dict = {'"mask"} {"placeholder": "text_a"} {"placeholder": "text_b"'},
                 ):
        super().__init__(model=model,
                         tokenizer=tokenizer,
                         placeholder_mapping=placeholder_mapping,
                         )
        self.new_transformer_head = None
        self.new_mlp_head = None
        self.new_lstm_head = None
        self.new_ids = None
        self.new_embedding = None
        self.num_soft_token = None
        self.prompt_encoder_type = prompt_encoder_type
        self.text = text

    def on_text_set(self):
        super().on_text_set()
        self.num_soft_token = sum([soft_id != 0 for soft_id in self.soft_token_ids])
        self.generate_parameters()

    def generate_parameters(self) -> None:
        if self.num_soft_token == 0: return
        self.new_embedding = nn.Embedding(self.num_soft_token, self.embedding_size)
        self.new_ids = nn.Parameter(torch.LongTensor(list(range(self.num_soft_token))), requires_grad=False)
        if self.prompt_encoder_type == "lstm":
            self.new_lstm_head = nn.LSTM(
                input_size=self.embedding_size,
                hidden_size=self.embedding_size,
                num_layers=2,
                bidirectional=True,
                batch_first=True
            )
            self.new_mlp_head = nn.Sequential(
                nn.Linear(2 * self.embedding_size, self.embedding_size),
                nn.ReLU(),
                nn.Linear(self.embedding_size, self.embedding_size)
            )
        elif self.prompt_encoder_type == "mlp":
            self.new_mlp_head = nn.Sequential(
                nn.Linear(self.embedding_size, self.embedding_size),
                nn.ReLU(),
                nn.Linear(self.embedding_size, self.embedding_size)
            )
        elif self.prompt_encoder_type == "transformer":
            self.new_transformer_head = nn.Transformer(
                d_model=768,
                nhead=12,
                num_encoder_layers=12,
                activation='gelu',
                custom_encoder='ctx_model'
            )
        else:
            raise ValueError("unknown prompt encoder type")

    def process_batch(self, batch: Union[Dict, InputFeatures]) -> Union[Dict, InputFeatures]:
        r"""
        Convert input_ids to inputs_embeds
        for normal tokens, use the embedding layer of PLM
        for new tokens, use a brand-new embedding layer, with MLP or LSTM head
        """
        inputs_embeds = self.raw_embedding(batch['input_ids'])

        if self.num_soft_token != 0:
            new_embeds = self.new_embedding(self.new_ids).unsqueeze(0)
            if self.prompt_encoder_type == "lstm":
                new_embeds = self.new_lstm_head(new_embeds)[0]
            elif self.prompt_encoder_type == "transformer":
                new_embeds = self.new_transformer_head(new_embeds)[0]
            new_embeds = self.new_mlp_head(new_embeds)

            replace_idxs = torch.nonzero(batch['soft_token_ids'] > 0).view(-1, self.num_soft_token, 2)
            for b in range(replace_idxs.shape[0]):
                for i in range(self.num_soft_token):
                    inputs_embeds[b][replace_idxs[b][i][1]] = new_embeds[0][i]

        batch['input_ids'] = None
        batch['inputs_embeds'] = inputs_embeds
        return batch


class EvalTemplate:
    def __init__(self, template):
        self.template = template

    def fill(self, prompt='', full_demo='', input='', output=''):
        return self.template.replace('[PROMPT]', prompt).replace(
            '[full_DEMO]', full_demo).replace('[INPUT]', input).replace('[OUTPUT]', output)
