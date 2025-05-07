import torch


class FineTuneHead(torch.nn.Module):
    def __init__(self, hidden_size: int, pool_type: str = "bigru"):
        super().__init__()
        self.classifier = torch.nn.Linear(hidden_size, 1)
        
        if pool_type == "bigru":
            self.pool = BiGRU(hidden_size)
        elif pool_type == "cls":
            self.pool = self.pool_cls
        else:
            raise ValueError(f"pool_type must be either 'bigru' or 'cls', got {pool_type}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        return_embedding: bool = False,
    ) -> torch.Tensor:
        if isinstance(self.pool, BiGRU):
            return self.pool(
                hidden_states,
                attention_mask=attention_mask,
                return_embedding=return_embedding,
            )
        else:
            x = self.pool(hidden_states)
            if return_embedding:
                return x
            return self.classifier(x)

    def pool_cls(self, x: torch.Tensor) -> torch.Tensor:
        """Use the CLS token embedding as the pooled representation."""
        return x[:, 0]


class BiGRU(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn_hidden_size = hidden_size // 2
        self.rnn = torch.nn.GRU(
            hidden_size, self.rnn_hidden_size, batch_first=True, bidirectional=True
        )
        # Adjust the input size of the classifier based on the bidirectionality
        classifier_input_size = hidden_size
        self.classifier = torch.nn.Linear(classifier_input_size, 1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        return_embedding: bool = False,
    ) -> torch.Tensor:
        lengths = attention_mask.sum(dim=1).cpu()
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            hidden_states, lengths, batch_first=True, enforce_sorted=False
        )
        # Pass the hidden states through the RNN
        output, _ = self.rnn(packed)
        # Unpack it back to a padded sequence
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        last_sequence_idx = lengths - 1

        # Use the last output of the RNN as input to the classifier
        # When bidirectional, we need to concatenate the last output from the forward
        # pass and the first output from the backward pass
        forward_output = output[
            torch.arange(output.shape[0]), last_sequence_idx, : self.rnn_hidden_size
        ]  # Last non-padded output from the forward pass
        backward_output = output[
            :, 0, self.rnn_hidden_size :
        ]  # First output from the backward pass
        x = torch.cat((forward_output, backward_output), dim=-1)
        if return_embedding:
            return x
        x = self.classifier(x)
        return x
