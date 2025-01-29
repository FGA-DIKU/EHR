import torch
import torch.nn as nn
from transformers import ModernBertModel

from corebehrt.classes.embeddings import EhrEmbeddings
from corebehrt.classes.heads import FineTuneHead, MLMHead
from transformers.modeling_outputs import MaskedLMOutput, SequenceClassifierOutput


class BertEHREncoder(ModernBertModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = EhrEmbeddings(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            type_vocab_size=config.type_vocab_size,
            embedding_dropout=config.embedding_dropout,
        )

    def forward(self, batch: dict):
        input_ids = batch["concept"]
        token_type_ids = batch["segment"]
        attention_mask = torch.ones_like(input_ids)
        position_ids = {key: batch[key] for key in ["age", "abspos"]}

        inputs_embeds = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )

        return super().forward(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask
        )


class BertEHRModel(BertEHREncoder):
    def __init__(self, config):
        super().__init__(config)
        self.loss_fct = nn.CrossEntropyLoss()
        self.cls = MLMHead(
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
        )

    def forward(self, batch: dict):
        outputs = super().forward(batch)

        sequence_output = outputs[0]  # Last hidden state
        logits = self.cls(sequence_output)

        loss = None
        if batch.get("target") is not None:
            loss = self.get_loss(logits, batch["target"])

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.get("hidden_states", None),
            attentions=outputs.get("attentions", None),
        )

    def get_loss(self, logits, labels):
        """Calculate loss for masked language model."""
        return self.loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))


class BertForFineTuning(BertEHREncoder):
    def __init__(self, config):
        super().__init__(config)

        self.loss_fct = nn.BCEWithLogitsLoss()
        self.cls = FineTuneHead(hidden_size=config.hidden_size)

    @torch._dynamo.disable
    def forward(self, batch: dict):
        outputs = super().forward(batch)

        sequence_output = outputs[0]  # Last hidden state
        logits = self.cls(sequence_output, batch["attention_mask"])

        loss = None
        if batch.get("target") is not None:
            loss = self.get_loss(logits, batch["target"])

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.get("hidden_states", None),
            attentions=outputs.get("attentions", None),
        )

    def get_loss(self, hidden_states, labels):
        return self.loss_fct(hidden_states.view(-1), labels.view(-1))
