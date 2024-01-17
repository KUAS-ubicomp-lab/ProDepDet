import torch
import torch.nn as nn
from openprompt import PromptForClassification
from openprompt.plms import load_plm
from openprompt.prompts import SoftTemplate, SoftVerbalizer
from transformers import Trainer

from .data_processor import data_processor_list
from .utils import decorate
from .modelling_roberta import RobertaEmbeddings
from .modelling_wsw import WSWEmbeddings
from .training_args import TrainingArguments


class PromptKernel(Trainer):

    def __init__(self, prompt_emb=None, **kwargs):
        args = kwargs['args']
        args.demonstration_type = "prompt_tuning"
        self.prompt_emb = prompt_emb
        self.demonstration_n = args.demonstration_sample
        self.latent_dropout = args.latent_dropout
        self.out_dir_root = args.output_dir
        self.pt_demonstration_input_layer = args.pt_demonstration_input_layer
        self.pt_demonstration_output_layer = args.pt_demonstration_output_layer
        self.pt_activation = args.backbone.activation

        processor = data_processor_list[args.dataset]()

        # Model
        template_text = '{"soft": None, "duplicate": ' + str(args.prompt_len) + ', "same": True} {"mask"} {' \
                                                                                '"placeholder": "text_a"} {' \
                                                                                '"placeholder": "text_b"}'

        model, template, verbalizer, plm, tokenizer, model_config, tokenizer_wrapper_class, model_type = self.get_model(
            args.backbone, processor, args)

        self.set_active_state_dict(model)  # Only save soft prompts

        # Initialize transformers.trainer
        kwargs['model'] = model
        kwargs['tokenizer'] = tokenizer
        kwargs['train_dataset'] = processor.train_dataset
        kwargs['eval_dataset'] = processor.eval_dataset
        super().__init__(**kwargs)

        self.config = model_config
        self.plm = plm
        self.template_text = template_text
        self.template = template
        self.verbalizer = verbalizer
        self.tokenizer_wrapper_class = tokenizer_wrapper_class
        self.prompt_emb = template.soft_embeds

        # Soft prompt transfer
        self.source_model_type = model_type
        self.target_model_type = None

        print('Trainable parameters:')
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                print(n, p.shape)

        print(f'Template: {self.template}')
        print(f'Verbalizer: {self.verbalizer}')
        print(f'Raw input example: {self.train_dataset[0]}')
        print(f'Wrapped input example: {self.template.wrap_one_example(self.train_dataset[0])}')

    def get_model(self, model_name, processor, args):
        model_type = []

        if 'bert-' in model_name:
            model_type = 'bert'

        elif 'roberta-' in model_name:
            model_type = 'roberta'

        elif 'wsw-' in model_name:
            model_type = 'wsw'

        # Load openprompt models
        if (not hasattr(self, 'args')) or args.backbone != model_name:
            plm, tokenizer, model_config, tokenizer_wrapper_class = load_plm(model_type, model_name)
        else:
            plm = self.plm
            tokenizer = self.tokenizer
            model_config = self.config
            tokenizer_wrapper_class = self.tokenizer_wrapper_class
            model_type = self.source_model_type

        # Load soft template
        if hasattr(self, 'template') and self.template is not None:
            template = self.template
        else:
            template_text = '{"soft": None, "duplicate": ' + str(
                args.prompt_len) + ', "same": True} {"mask"} {"placeholder": "text_a"} {"placeholder": "text_b"}'
            template = SoftTemplate(model=plm, tokenizer=tokenizer, text=template_text,
                                    soft_embeds=self.get_prompt_emb(args, model_config), num_tokens=args.prompt_len)

        # Load soft verbalizer. This is the first time of using a soft verbalizer in prompt task transferring to our
        # knowledge.
        if hasattr(self, 'verbalizer') and self.verbalizer is not None:
            verbalizer = self.verbalizer
        else:
            verbalizer = SoftVerbalizer(tokenizer, model=plm, classes=processor.labels,
                                        label_words=processor.label_words)

        # Set In-Context Demonstrations (ICD). This is the first time of using ICD in out-of-domain task transfer to
        # our knowledge.
        if hasattr(self, 'demonstration_type') and args.demonstration_type is not None:
            self.set_in_context_demonstrations(demonstration_type=self.demonstration_type,
                                               demonstration_sample=self.get_prompt_emb(args, model_config))

        if hasattr(self, 'model') and self.model is not None:
            model = self.model
        else:
            model = PromptForClassification(plm=plm, template=template, verbalizer=verbalizer, freeze_plm=True)

            if hasattr(args, "model_parallel") and args.model_parallel:
                print('parallelize model!')
                model.parallelize()

        _keys_to_ignore_on_save = []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                _keys_to_ignore_on_save.append(n)

        model._keys_to_ignore_on_save = _keys_to_ignore_on_save

        return model, template, verbalizer, plm, tokenizer, model_config, tokenizer_wrapper_class, model_type

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(inputs)
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        loss = nn.CrossEntropyLoss()(outputs, inputs['label'])

        return (loss, outputs) if return_outputs else loss

    @staticmethod
    def set_active_state_dict(module: nn.Module, includes=['prompt_model.template.soft_embeds']):
        def _caller(_org_func, includes, *args, **kwargs):
            state_dict = _org_func(*args, **kwargs)
            keys = list(state_dict.keys())
            for n in keys:
                if n not in includes:
                    state_dict.pop(n)
            return state_dict

        includes = includes
        if hasattr(module.state_dict, "__wrapped__"):
            raise RuntimeWarning(
                "The forward function might have been wrapped by a decorator, is it intended? Do you freeze the "
                "parameters twice?")
        module.state_dict = decorate(module.state_dict, _caller, extras=(includes,), kwsyntax=True)

    def set_in_context_demonstrations(self, demonstration_type, demonstration_sample):
        self.demonstration_n = demonstration_sample
        self.latent_dropout = nn.Dropout(0.1)
        self.config.demonstration_type = demonstration_type

        if TrainingArguments.demonstration_type == "prompt_tuning":
            self.demonstration_n = demonstration_sample
            self.prompt_emb = nn.Embedding(demonstration_sample, nn.Embedding.embedding_dim)
            nn.LayerNorm(nn.Embedding.embedding_dim, eps=self.config.layer_norm_eps)
            self.prompt_emb.weight = self.prompt_emb
            self.pt_demonstration_input_layer = nn.Linear(nn.Embedding.embedding_dim, nn.Embedding.embedding_dim * 4)
            self.pt_activation = nn.GELU
            self.pt_demonstration_output_layer = nn.Linear(nn.Embedding.embedding_dim * 4, nn.Embedding.embedding_dim)
            nn.LayerNorm(nn.Embedding.embedding_dim, eps=self.config.layer_norm_eps)
        else:
            raise NotImplementedError

    @torch.no_grad()
    def get_prompt_emb(self, args, config):
        prompt_emb = []

        if 'bert-' in args.backbone:
            prompt_emb = args.backbone.bert.embeddings.prompt_embeddings

        if 'roberta-' in args.backbone:
            prompt_emb = RobertaEmbeddings(config=config)

        if 'wsw-' in args.backbone:
            prompt_emb = WSWEmbeddings(config=config)

        return prompt_emb.prompt_embeddings.weight
