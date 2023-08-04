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
from .mlp import MLP
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
        assert (
            d == self.d_model
        ), f"Expected {self.d_model=} features as input but got {d} features, input shape {X.shape}"

        Q = self.W_q(X)
        K = self.W_k(X)
        V = self.W_v(X)

        Q = einops.rearrange(Q, "b n (h d_k) -> b h n d_k", d_k=self.d_k)
        K_t = einops.rearrange(K, "b n (h d_k) -> b h d_k n", d_k=self.d_k)
        V = einops.rearrange(V, "b n (h d_v) -> b h n d_v", d_v=self.d_v)

        attn_scores = (Q @ K_t * self.scale)
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
        

class Transformer(nn.Module): 
    def __init__(self, n_layers=6, n_heads=8, d_model=512, d_feed_forward=768, dropout=0.1):
        super().__init__()

        self.n_layers = n_layers 
        self.n_heads = n_heads 
        self.d_model = d_model
        self.d_feed_forward = d_feed_forward
        self.dropout = dropout 

        self.attn_blocks = nn.ModuleList(
            [
                MultiHeadAttention(
                    d_model=self.d_model, 
                    n_heads=self.n_heads,
                ) for _ in range(self.n_layers)
            ]
        )
        self.attn_layernorms = nn.ModuleList(
            [
                nn.LayerNorm(d_model) 
                for _ in range(self.n_layers)
            ]
        )
        self.feed_forward_blocks = nn.ModuleList(
            [
                MLP(d_model, d_feed_forward, d_model)
                for _ in range(self.n_layers)
            ]
        )
        self.ff_layernorms = nn.ModuleList(
            [
                nn.LayerNorm(d_model) 
                for _ in range(self.n_layers)
            ]
        )
        self.dropout = nn.Dropout(p=self.dropout)

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

        return X, {
            'attentions': attentions, 
            'layer_outputs': layer_outputs
        }
            

# TODO
class TransformerForSequencePrediction(nn.Module):
    def __init__(self, vocab_size, pad_token, n_layers=6, n_heads=8, d_model=512, d_feed_forward=768, dropout=0.1): 
        self.embeddings = nn.Embedding(
            vocab_size, d_model, pad_token
        )