import math
import torch
import numpy as np
from torch import Tensor, nn
from typing import Tuple, Type, Optional, Any
from einops import rearrange

class RAttention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert (self.internal_dim % num_heads == 0), "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1

        if coords.dtype != self.positional_encoding_gaussian_matrix.dtype:
            coords = coords.to(self.positional_encoding_gaussian_matrix.dtype)

        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=self.positional_encoding_gaussian_matrix.dtype)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W


class RmodwihTranslate(nn.Module):
    def __init__(self, embedding_dim, image_self_attn=False):
        super().__init__()
        self.head = 4
        self.self_attn = RAttention(embedding_dim, self.head)
        self.image_self_attn = image_self_attn
        if self.image_self_attn:
            self.img_self_attn = RAttention(embedding_dim, self.head)
            self.image_norm = nn.LayerNorm(embedding_dim)
        self.cross_attn = RAttention(embedding_dim, self.head)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.pe_layer = PositionEmbeddingRandom(embedding_dim // 2)
        self.mlp = nn.Linear(embedding_dim, embedding_dim)
    
    def forward(self, image_feature, text_feature):
        
        text_len = text_feature.shape[1]
        text_pe = self.pe_layer((1, text_len)).squeeze().unsqueeze(0).transpose(-1, -2)
        w = 60
        h = image_feature.size(1) // w
        image_pe = rearrange(self.pe_layer((h, w)).unsqueeze(0), 'b d h w -> b (h w) d')
        
        text_q = text_feature + text_pe
        text_feature_out = self.self_attn(q=text_q, k=text_q, v=text_feature)
        text_feature = text_feature + text_feature_out
        text_feature = self.norm1(text_feature)
        
        if self.image_self_attn:
            image_q = image_feature + image_pe
            image_feature_out = self.self_attn(q=image_q, k=image_q, v=image_feature)
            image_feature = image_feature + image_feature_out
            image_feature = self.image_norm(image_feature)
        
        image_q = image_feature + image_pe
        text_k = text_feature + text_pe
        image_feature_out = self.cross_attn(q=image_q, k=text_k, v=text_feature)
        image_feature = image_feature + image_feature_out
        image_feature = self.norm2(image_feature)
        
        image_feature_out = self.mlp(image_feature)
        image_feature = image_feature + image_feature_out
        image_feature = self.norm3(image_feature)

        return image_feature, text_feature


class RInterMod(nn.Module):
    def __init__(self, dim_in, embedding_dim, depth):
        super().__init__()
        self.dim_in = dim_in
        self.embedding_dim = embedding_dim
        self.depth = depth
        self.hidden_dim = 2048
        # self.lora_hidden_dim = 8
        
        self.proj_in = nn.Linear(dim_in, self.hidden_dim)
        self.proj_out = nn.Linear(self.hidden_dim, dim_in)
        
        self.text_proj_in = nn.Linear(embedding_dim, self.hidden_dim)
        # self.text_proj_in_2 = nn.Linear(self.lora_hidden_dim, self.hidden_dim)
        
        self.layers = nn.ModuleList()
        for i in range(self.depth):
            self.layers.append(
                RmodwihTranslate(self.hidden_dim, True if (i == 0) else False)
            )
            # self.layers.append(
            #     RmodwihTranslate(self.embedding_dim, False)
            # )
    
    def forward(self, image_feature, text_feature):
        image_feature = self.proj_in(image_feature)
        text_feature = self.text_proj_in(text_feature)
        for layer in self.layers:
            image_feature, text_feature = layer(image_feature, text_feature)
        image_feature = self.proj_out(image_feature)
        return image_feature
