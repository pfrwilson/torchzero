"""
Implementation of transformers from the original paper:

@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}

Author: Paul Wilson
"""

from torch import nn
from torchzero.nn import MLP
import torch
import einops

INF = torch.tensor(1e10)


class MultiHeadAttention(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, d_k: int | None = None, d_v: int | None = None
    ):
        """
        Implements multi-head scaled dot product self attention.
        Args:
            d_model (int): the number of tokens for the input features.
            n_heads (int): the number of attention heads.
            d_k (optional, int): the dimension of key vectors for each head. Defaults
                to d_model divided by the number of heads.
            d_v (optional, int): the dimension of the value vectors for each head. Defaults
                to d_model divided by the number of heads.
        """
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads

        self.d_k = d_k = d_k or (d_model // n_heads)
        self.d_v = d_v = d_v or (d_model // n_heads)

        self.W_q = nn.Linear(d_model, d_k * n_heads)
        self.W_k = nn.Linear(d_model, d_k * n_heads)
        self.W_v = nn.Linear(d_model, d_v * n_heads)

        self.W_o = nn.Linear(d_v * n_heads, d_model)

        self.scale = 1 / torch.sqrt(torch.tensor(d_k).float())

    def forward(self, X, mask=None):
        """
        X - tensor of shape batch_size, sequence_length, input_dim
        mask - tensor of shape batch_size, n_heads, sequence_length, sequence_length
        """

        B, n_tokens, d = X.shape

        Q = self.W_q(X)
        K = self.W_k(X)
        V = self.W_v(X)

        Q = einops.rearrange(Q, "b n (h d_k) -> b h n d_k", d_k=self.d_k)
        K_t = einops.rearrange(K, "b n (h d_k) -> b h d_k n", d_k=self.d_k)
        V = einops.rearrange(V, "b n (h d_v) -> b h n d_v", d_v=self.d_v)

        attn_scores = Q @ K_t * self.scale
        if mask is not None:
            attn_scores[mask == 0] = -INF

        attn = attn_scores.softmax(-1)
        attn_weighted_values = attn @ V

        # concatenate across head dim
        attn_weighted_values = einops.rearrange(
            attn_weighted_values, "b h n d_v -> b n (h d_v)"
        )

        layer_output = self.W_o(attn_weighted_values)
        return layer_output, attn


class MultiHeadAttentionWithRelativePositionalEmbeddings(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_k: int | None = None,
        d_v: int | None = None,
        max_distance=512,
        key_only_for_relative_pos_emb=False,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        self.d_k = d_k = d_k or (d_model // n_heads)
        self.d_v = d_v = d_v or (d_model // n_heads)

        self.W_q = nn.Linear(d_model, d_k * n_heads)
        self.W_k = nn.Linear(d_model, d_k * n_heads)
        self.W_v = nn.Linear(d_model, d_v * n_heads)

        self.W_o = nn.Linear(d_v * n_heads, d_model)

        self.scale = 1 / torch.sqrt(torch.tensor(d_k).float())

        self.max_distance = max_distance
        self.num_rel_embeddings = max_distance + (max_distance - 1)
        self.P_k = nn.Parameter(
            data=torch.randn(self.num_rel_embeddings, self.n_heads)
        )
        self.P_v = nn.Parameter(
            data=torch.randn(self.num_rel_embeddings, n_heads, self.d_v)
        ) 
        self.key_only_for_relative_pos_emb = key_only_for_relative_pos_emb

    def generate_rel_position_embedding_matrices(self, sequence_length):
        if sequence_length > self.max_distance:
            raise ValueError(
                f"Received a sequence length of {sequence_length},"
                f"but we are only equipped for tokens up to {self.max_distance} tokens apart."
            )
        zero_offset_idx = self.num_rel_embeddings // 2
        A_k = []
        A_v = []
        for i in range(sequence_length):
            A_k.append(
                self.P_k[zero_offset_idx - i : zero_offset_idx + sequence_length - i]
            )
            A_v.append(
                self.P_v[zero_offset_idx - i : zero_offset_idx + sequence_length - i]
            )
        A_k = torch.stack(A_k)  # n_out, n_in, h
        A_v = torch.stack(A_v)  # n_out, n_in, h, d_v

        return A_k, A_v

    def forward(self, X, mask=None):
        """
        X - tensor of shape batch_size, sequence_length, input_dim
        mask - tensor of shape batch_size, n_heads, sequence_length, sequence_length
        """
        B, n_tokens, d = X.shape

        Q = self.W_q(X)
        K = self.W_k(X)
        V = self.W_v(X)

        Q = einops.rearrange(Q, "b n (h d_k) -> b h n d_k", d_k=self.d_k)
        K_t = einops.rearrange(K, "b n (h d_k) -> b h d_k n", d_k=self.d_k)
        V = einops.rearrange(V, "b n (h d_v) -> b h n d_v", d_v=self.d_v)

        A_k, A_v = self.generate_rel_position_embedding_matrices(n_tokens)
        A_k = einops.repeat(A_k, "n_out n_in h -> b n_out n_in h", b=B)
        A_k = einops.rearrange(A_k, "b n_out n_in h -> b h n_out n_in")

        attn_scores = (Q @ K_t + A_k) * self.scale
        if mask is not None:
            attn_scores[mask == 0] = -INF

        attn = attn_scores.softmax(-1)
        attn_weighted_values = attn @ V  # b h n d_v

        if not self.key_only_for_relative_pos_emb:
            # compute value components from relative position embeddings
            A_v = einops.repeat(A_v, "n_out n_in h d_k -> b n_out n_in h d_k", b=B)
            A_v = einops.rearrange(A_v, "b n_out n_in h d_k -> b h n_out n_in d_k")
            attn_broadcast = einops.repeat(attn, "b h n_out n_in -> b h n_out n_in 1")
            attn_weighted_abs_pos_value = (attn_broadcast * A_v).sum(-2)
            attn_weighted_values = attn_weighted_values + attn_weighted_abs_pos_value

        # concatenate across head dim
        attn_weighted_values = einops.rearrange(
            attn_weighted_values, "b h n d_v -> b n (h d_v)"
        )

        layer_output = self.W_o(attn_weighted_values)
        return layer_output, attn


class TransformerEncoder(nn.Module):
    """
    Main body of the transformer encoder, which processes a sequence of tokens
    to a sequence of contextualized tokens of the same shape.
    """
    def __init__(
        self, n_layers=6, n_heads=8, d_model=512, d_feed_forward=768, dropout=0.1
    ):
        super().__init__()

        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_feed_forward = d_feed_forward
        self.dropout = dropout

        self.attn_blocks = nn.ModuleList(
            [self.build_attn_layer() for _ in range(self.n_layers)]
        )
        self.attn_layernorms = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(self.n_layers)]
        )
        self.feed_forward_blocks = nn.ModuleList(
            [MLP(d_model, d_feed_forward, d_model) for _ in range(self.n_layers)]
        )
        self.ff_layernorms = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(self.n_layers)]
        )
        self.dropout = nn.Dropout(p=self.dropout)

    def build_attn_layer(self):
        return MultiHeadAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
        )

    def forward(self, X, mask=None):
        layer_outputs = []
        attentions = []

        for i in range(self.n_layers):
            attn_layer_out, attn = self.attn_blocks[i](X, mask=mask)
            X = self.attn_layernorms[i](X + attn_layer_out)

            ff_out = self.feed_forward_blocks[i](X)
            X = self.ff_layernorms[i](X + ff_out)
            X = self.dropout(X)

            layer_outputs.append(X)
            attentions.append(attn)

        return X, {"attentions": attentions, "layer_outputs": layer_outputs}


class TransformerEncoderWithRelativePosEmbeddings(TransformerEncoder):
    def __init__(
        self,
        n_layers=6,
        n_heads=8,
        d_model=512,
        d_feed_forward=768,
        dropout=0.1,
        key_only_for_pos_emb=False,
        max_distance=512, 
    ):
        self.key_only_for_pos_emb = key_only_for_pos_emb
        self.max_distance = max_distance
        super().__init__(
            n_layers, n_heads, d_model, d_feed_forward, dropout
        )

    def build_attn_layer(self):
        return MultiHeadAttentionWithRelativePositionalEmbeddings(
            self.d_model, self.n_heads, max_distance=self.max_distance, 
            key_only_for_relative_pos_emb=self.key_only_for_pos_emb
        )


class TransformerForSequenceGeneration(nn.Module):
    def __init__(
        self,
        vocab_size,
        pad_token,
        max_sequence_length=512,
        n_layers=6,
        n_heads=8,
        d_model=512,
        d_feed_forward=768,
        dropout=0.1,
    ):
        self.token_embeddings = nn.Embedding(vocab_size, d_model, pad_token)
        self.postition_embeddings = nn.Embedding(max_sequence_length, d_model)
        self.transformer_encoder = TransformerEncoder(
            n_layers=n_layers,
            n_heads=n_heads,
            d_model=d_model,
            d_feed_forward=d_feed_forward,
            dropout=dropout,
        )
        self.classifier = nn.Linear(d_model, vocab_size)

        self._checked_padding = False
        self.pad_token = pad_token
        self.d_model = d_model

    def forward(self, X):
        ...

    def _create_mask(self, X):
        B, N = X.shape
        padded_locations = X == self.pad_token


if __name__ == '__main__': 
    model = TransformerEncoderWithRelativePosEmbeddings()
    input = torch.randn(4, 64, 512)
    import torchinfo
    torchinfo.summary(
        model, input_data=input
    )