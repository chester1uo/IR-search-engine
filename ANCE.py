import torch
from transformers import PreTrainedModel, BertConfig, BertModel, AutoTokenizer, RobertaConfig, RobertaModel,RobertaTokenizer
import re
from typing import Optional
import os
import json
import numpy as np
from tqdm.notebook import tqdm_notebook
import time


class AnceModel(PreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = 'ance_encoder'
    load_tf_weights = None
    _keys_to_ignore_on_load_missing = [r'position_ids']
    _keys_to_ignore_on_load_unexpected = [r'pooler', r'classifier']

    def __init__(self, config: RobertaConfig):
        super().__init__(config)
        self.config = config
        self.roberta = RobertaModel(config)
        self.embeddingHead = torch.nn.Linear(config.hidden_size, 768)
        self.norm = torch.nn.LayerNorm(768)
        self.init_weights()

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def init_weights(self):
        self.roberta.init_weights()
        self.embeddingHead.apply(self._init_weights)
        self.norm.apply(self._init_weights)

    def forward(self, input_ids: torch.Tensor,attention_mask: Optional[torch.Tensor] = None, ):
        input_shape = input_ids.size()
        device = input_ids.device
        if attention_mask is None:
            attention_mask = (
                torch.ones(input_shape, device=device)
                if input_ids is None
                else (input_ids != self.roberta.config.pad_token_id)
            )
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        pooled_output = sequence_output[:, 0, :]
        pooled_output = self.norm(self.embeddingHead(pooled_output))
        return pooled_output

