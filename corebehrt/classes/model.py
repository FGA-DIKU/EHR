import torch
import torch.nn as nn
from transformers import BertModel
from transformers.models.roformer.modeling_roformer import RoFormerEncoder

from corebehrt.classes.embeddings import EhrEmbeddings
from corebehrt.classes.activations import SwiGLU
from corebehrt.classes.heads import FineTuneHead, MLMHead


class BertEHREncoder(BertModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = EhrEmbeddings(vocab_size=config.vocab_size,
                                        hidden_size=config.hidden_size,
                                        type_vocab_size=config.type_vocab_size,
                                        layer_norm_eps=config.layer_norm_eps,
                                        hidden_dropout_prob=config.hidden_dropout_prob)

        # Activate transformer++ recipe
        config.rotary_value = False
        self.encoder = RoFormerEncoder(config)
        for layer in self.encoder.layer:
            layer.intermediate.intermediate_act_fn = SwiGLU(config.intermediate_size)

    def forward(self, batch: dict):
        position_ids = {key: batch[key] for key in ["age", "abspos"]}
        outputs = super().forward(
            input_ids=batch["concept"],
            attention_mask=torch.ones_like(batch["concept"]),
            token_type_ids=batch["segment"],
            position_ids=position_ids,
        )
        return outputs


class BertEHRModel(BertEHREncoder):
    def __init__(self, config):
        super().__init__(config)
        self.loss_fct = nn.CrossEntropyLoss()
        self.cls = MLMHead(hidden_size=config.hidden_size)

    def forward(self, batch: dict):
        outputs = super().forward(batch)

        sequence_output = outputs[0]  # Last hidden state
        logits = self.cls(sequence_output)
        outputs.logits = logits

        if batch.get("target") is not None:
            outputs.loss = self.get_loss(logits, batch["target"])

        return outputs

    def get_loss(self, logits, labels):
        """Calculate loss for masked language model."""
        return self.loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))


class BertForFineTuning(BertEHREncoder):
    def __init__(self, config):
        super().__init__(config)

        self.loss_fct = nn.BCEWithLogitsLoss()
        self.cls = FineTuneHead(hidden_size=config.hidden_size)

    def forward(self, batch: dict):
        outputs = super().forward(batch)

        sequence_output = outputs[0]  # Last hidden state
        logits = self.cls(sequence_output, batch["attention_mask"])
        outputs.logits = logits

        if batch.get("target") is not None:
            outputs.loss = self.get_loss(logits, batch["target"])

        return outputs

    def get_loss(self, hidden_states, labels):
        return self.loss_fct(hidden_states.view(-1), labels.view(-1))
