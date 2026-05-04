"""
Model architectures. Three CNN encoders + one Transformer decoder.

All three encoders produce output of shape [batch, num_patches, embed_dim]
ready to be fed as `memory` to the TransformerDecoder.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------

class ShallowCNN(nn.Module):
    """2 conv layers + projection. Minimal baseline."""

    def __init__(self, embed_dim: int = 256, in_channels: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 256, kernel_size=3, padding=1)
        self.proj = nn.Conv2d(256, embed_dim, kernel_size=1)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = self.proj(x)
        x = x.flatten(2).permute(0, 2, 1)   # [B, N, D]
        return x


class DeepCNN(nn.Module):
    """5 conv layers with more aggressive depth growth."""

    def __init__(self, embed_dim: int = 256, in_channels: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, 3, padding=1)
        self.proj = nn.Conv2d(256, embed_dim, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv5(x))
        x = self.proj(x)
        x = x.flatten(2).permute(0, 2, 1)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        res = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + res)


class ResNetCNN(nn.Module):
    """Small ResNet-style encoder with skip connections."""

    def __init__(self, embed_dim: int = 256, in_channels: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.res1 = ResidualBlock(64)
        self.conv2 = nn.Conv2d(64, 256, 3, padding=1)
        self.res2 = ResidualBlock(256)
        self.proj = nn.Conv2d(256, embed_dim, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.res1(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = self.res2(x)
        x = F.max_pool2d(x, 2)
        x = self.proj(x)
        x = x.flatten(2).permute(0, 2, 1)
        return x


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class TransformerDecoder(nn.Module):
    """
    Standard Transformer decoder:
      - learned token + positional embeddings
      - causal self-attention over previously generated tokens
      - cross-attention over CNN feature-map positions
    """

    def __init__(self, vocab_size: int, embed_dim: int = 256, max_length: int = 150,
                 num_heads: int = 8, num_layers: int = 3, ff_dim: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_length, embed_dim)
        layer = nn.TransformerDecoderLayer(embed_dim, num_heads, ff_dim, dropout,
                                            batch_first=True)
        self.transformer = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, tgt, memory):
        seq_len = tgt.shape[1]
        positions = torch.arange(seq_len, device=tgt.device)
        x = self.embedding(tgt) + self.pos_embedding(positions)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=tgt.device)
        x = self.transformer(x, memory, tgt_mask=mask)
        return self.fc_out(x)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

ENCODERS = {
    "shallow": ShallowCNN,
    "deep": DeepCNN,
    "resnet": ResNetCNN,
}


def build_model(encoder_name: str, vocab_size: int, embed_dim: int = 256,
                max_length: int = 150, num_heads: int = 8,
                num_decoder_layers: int = 3, ff_dim: int = 512,
                dropout: float = 0.1):
    encoder_cls = ENCODERS[encoder_name]
    cnn = encoder_cls(embed_dim=embed_dim)
    decoder = TransformerDecoder(
        vocab_size=vocab_size, embed_dim=embed_dim, max_length=max_length,
        num_heads=num_heads, num_layers=num_decoder_layers, ff_dim=ff_dim,
        dropout=dropout,
    )
    return cnn, decoder


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)