//
//  TransformerRecommender.swift
//  
//
//  Created by VENKATA KOUSHIK NAGASARAPU on 5/11/24.
//


# -*- coding: utf-8 -*-

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('..')

from biga.models import PerturbedModel

class TransformerRecommender(PerturbedModel):
    def __init__(self, config, dataset, **kwargs):
        super(TransformerRecommender, self).__init__(config, **kwargs)

        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.embed_dim = config['embed_dim']
        self.num_heads = config['num_heads']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        self.max_seq_length = config['max_seq_length']

        # Embedding layers for users and items
        self.user_embedding = nn.Embedding(self.num_users, self.embed_dim)
        self.item_embedding = nn.Embedding(self.num_items, self.embed_dim)

        # Positional encoding
        self.position_embedding = nn.Embedding(self.max_seq_length, self.embed_dim)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=4 * self.embed_dim,
            dropout=self.dropout,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=self.num_layers
        )

        # Prediction layer
        self.fc_out = nn.Linear(self.embed_dim, 1)

        # Dropout for regularization
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, user_seq, item_seq, pred=False):
        """
        user_seq: Tensor of shape (batch_size, seq_length)
        item_seq: Tensor of shape (batch_size, seq_length)
        """
        batch_size, seq_length = user_seq.size()

        # Embedding lookup
        user_embed = self.user_embedding(user_seq)  # Shape: (batch_size, seq_length, embed_dim)
        item_embed = self.item_embedding(item_seq)  # Shape: (batch_size, seq_length, embed_dim)

        # Sum user and item embeddings
        seq_embedding = user_embed + item_embed

        # Add positional encoding
        positions = torch.arange(seq_length, device=seq_embedding.device).expand(batch_size, seq_length)
        seq_embedding += self.position_embedding(positions)

        # Apply transformer encoder
        seq_embedding = seq_embedding.permute(1, 0, 2)  # Shape: (seq_length, batch_size, embed_dim)
        transformer_output = self.transformer(seq_embedding)  # Shape: (seq_length, batch_size, embed_dim)
        transformer_output = transformer_output.permute(1, 0, 2)  # Shape: (batch_size, seq_length, embed_dim)

        # Extract final token's embedding for prediction
        final_output = transformer_output[:, -1, :]  # Shape: (batch_size, embed_dim)

        # Apply dropout and feed through prediction layer
        final_output = self.dropout_layer(final_output)
        scores = self.fc_out(final_output).squeeze(-1)  # Shape: (batch_size,)

        return scores

    def predict(self, interaction, pred=False):
        user_seq = interaction['user_seq']
        item_seq = interaction['item_seq']

        scores = self.forward(user_seq, item_seq, pred=pred)
        return scores

    def full_sort_predict(self, interaction, pred=False):
        user_seq = interaction['user_seq']
        item_seq = interaction['item_seq']

        scores = self.forward(user_seq, item_seq, pred=pred)
        return scores
