import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias = True, out_proj_bias = True):
        super().__init__()

        self.in_proj = nn.Linear(d_embed, 3*d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, causal_mask= False) -> torch.Tensor:
        # x: B, Seq_len, Dim

        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        interm_shape = (batch_size, sequence_length, self.n_heads, self.d_head)
        
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # B, seq_len, dim -> B, seg_len, H, dim/h -> B, H, seg_len, dim/h
        q = q.view(interm_shape).transpose(1, 2)
        k = k.view(interm_shape).transpose(1, 2)
        v = v.view(interm_shape).transpose(1, 2)

        # B, H, seq_len, dim/h -> B, H, dim/h, seq_len 
        weight = q@k.transpose(-1, -2)

        if causal_mask:
            # Mask where the upper triangle (above the principal diagonal) is made off of 1
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)

        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim= -1)

        # B, H, Seq_len, Seg_len @ B, H, Seg_len, dim/h -> B, H, Seq_len, dim/h
        output = weight @ v

        # B, H, Seq_len, dim/h -> B, seq_len, h, dim/h
        output = output.transpose(1,2)

        output = output.reshape(input_shape)

        output = self.out_proj(output)

        # B, Seg_len, dim
        return output
    
class CrossAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, d_cross: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, y):
        # x: (latent): B, seq_len_q, dim_q
        # y: (context): B, seq_len_kv, dim_kv = B, 77, 168

        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape

        interm_shape = (batch_size, -1, self.n_heads, self.d_head)

        # Multiply query by wq
        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        q = q.view(interm_shape).transpose(1, 2)
        k = k.view(interm_shape).transpose(1, 2)
        v = v.view(interm_shape).transpose(1, 2)

        weight = q @ k.transpose(-1, -2)

        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim=-1)

        output = weight @ v

        output = output.transpose(1, 2).contiguous()

        output = output.view(input_shape)

        output = self.out_proj(output)

        return output