"""
Bahdanau additive attention.
"""

import torch
import torch.nn as nn


class BahdanauAttention(nn.Module):

    def __init__(
        self,
        hidden_dim
    ):
        super().__init__()

        self.encoder_projection = nn.Linear(
            hidden_dim,
            hidden_dim,
            bias=False
        )

        self.decoder_projection = nn.Linear(
            hidden_dim,
            hidden_dim,
            bias=False
        )

        self.energy_projection = nn.Linear(
            hidden_dim,
            1,
            bias=False
        )


    def forward(
        self,
        decoder_hidden,
        encoder_outputs,
        mask
    ):
        decoder_features = (
            self.decoder_projection(
                decoder_hidden
            ).unsqueeze(1)
        )

        encoder_features = (
            self.encoder_projection(
                encoder_outputs
            )
        )

        energy = torch.tanh(
            encoder_features
            +
            decoder_features
        )

        energy = self.energy_projection(
            energy
        ).squeeze(-1)

        energy = energy.masked_fill(
            ~mask,
            -1e9
        )

        attention = torch.softmax(
            energy,
            dim=1
        )

        context = torch.bmm(
            attention.unsqueeze(1),
            encoder_outputs
        ).squeeze(1)

        return context, attention


class BahdanauDecoder(nn.Module):

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

        self.attention = (
            BahdanauAttention(
                hidden_dim
            )
        )

        self.gru = nn.GRU(
            embedding_dim + hidden_dim,
            hidden_dim,
            batch_first=True
        )

        self.fc_out = nn.Linear(
            embedding_dim
            +
            hidden_dim
            +
            hidden_dim,
            output_dim
        )


    def forward(
        self,
        input_token,
        hidden,
        encoder_outputs,
        src_mask
    ):
        input_token = (
            input_token.unsqueeze(1)
        )

        embedded = self.embedding(
            input_token
        )

        context, attention = (
            self.attention(
                hidden[-1],
                encoder_outputs,
                src_mask
            )
        )

        gru_input = torch.cat(
            [
                embedded,
                context.unsqueeze(1)
            ],
            dim=2
        )

        output, hidden = self.gru(
            gru_input,
            hidden
        )

        output = output.squeeze(1)

        prediction = self.fc_out(
            torch.cat(
                [
                    output,
                    context,
                    embedded.squeeze(1)
                ],
                dim=1
            )
        )

        return (
            prediction,
            hidden,
            attention
        )
