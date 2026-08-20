"""
Vanilla RNN encoder and decoder.
"""

import torch
import torch.nn as nn


class VanillaRNNEncoder(nn.Module):

    def __init__(
        self,
        input_dim,
        embedding_dim,
        hidden_dim,
        padding_idx
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            input_dim,
            embedding_dim,
            padding_idx=padding_idx
        )

        self.rnn = nn.RNN(
            embedding_dim,
            hidden_dim,
            batch_first=True
        )


    def forward(self, src):
        embedded = self.embedding(src)

        outputs, hidden = self.rnn(
            embedded
        )

        return outputs, hidden


class VanillaRNNDecoder(nn.Module):

    def __init__(
        self,
        output_dim,
        embedding_dim,
        hidden_dim,
        padding_idx
    ):
        super().__init__()

        self.output_dim = output_dim

        self.embedding = nn.Embedding(
            output_dim,
            embedding_dim,
            padding_idx=padding_idx
        )

        self.rnn = nn.RNN(
            embedding_dim,
            hidden_dim,
            batch_first=True
        )

        self.fc_out = nn.Linear(
            hidden_dim,
            output_dim
        )


    def forward(
        self,
        input_token,
        hidden
    ):
        input_token = (
            input_token.unsqueeze(1)
        )

        embedded = self.embedding(
            input_token
        )

        output, hidden = self.rnn(
            embedded,
            hidden
        )

        prediction = self.fc_out(
            output.squeeze(1)
        )

        return prediction, hidden
