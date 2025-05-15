import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.optim as optim


class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim):
      # embed_dim = pacth_size * patch_size * in_channels
      # TODO
      super().__init__()
      self.num_patches = (image_size // patch_size) ** 2
      
      self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
      # TODO
      ## x: (B, C, H, W)
      x = self.proj(x).flatten(2).transpose(1, 2)   # -> (B, num_patches, embed_dim)
      return x
    

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
      # TODO
      super().__init__()
      self.num_heads = num_heads
      self.head_dim = embed_dim // num_heads
      self.scale = self.head_dim ** -0.5  # the scale factor for the dot product

      self.q = nn.Linear(embed_dim, embed_dim)
      self.k = nn.Linear(embed_dim, embed_dim)
      self.v = nn.Linear(embed_dim, embed_dim)
      self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
      # TODO
      ## x: (B, num_patches + class_token, embed_dim)
      B, num_patches, embed_dim = x.shape
      q = self.q(x).reshape(B, num_patches, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, num_heads, num_patches, head_dim)
      k = self.k(x).reshape(B, num_patches, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # or (B, h_i, x_i, d_k)
      v = self.v(x).reshape(B, num_patches, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
      attention = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, num_patches, num_patches)
      attention = attention.softmax(dim=-1)
      attention = attention @ v   # (B, num_heads, num_patches, head_dim)
      x = attention.transpose(1, 2).reshape(B, num_patches, embed_dim)  # (B, num_patches, embed_dim)

      x = self.proj(x)
      return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout):
        # TODO
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )


    def forward(self, x):
        # TODO
        x = x + self.dropout1(self.attention(self.norm1(x)))
        x = x + self.mlp(self.norm2(x))
        return x
    

class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim, num_heads, mlp_dim, num_layers, num_classes, dropout=0.1):
        # TODO
        super().__init__()
        self.pacth_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        self.num_patches = self.pacth_embed.num_patches

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        self.transformer = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout)
            for i in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.mlp_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # TODO
        x = self.pacth_embed(x) # (B, num_patches, embed_dim)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1) # (1, 1, embed_dim) -> (B, 1, embed_dim)
        x = torch.cat([cls_token, x], dim=1)    # (B, num_patches + 1, embed_dim)

        x = x + self.pos_embed
        x = self.dropout(x)

        x = self.transformer(x)
        x = self.norm(x)
        logits = self.mlp_head(x[:, 0])
        return logits
