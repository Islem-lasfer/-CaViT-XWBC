"""HybridNet : branche CNN (Inception + convs dilatées + CBAM + ResBlocks SE)
+ branche ViT, fusionnées par cross-attention."""

import torch
import torch.nn as nn


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0,
                 dilation=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class InceptionMultiScale(nn.Module):
    def __init__(self, in_ch, branch_ch=64):
        super().__init__()
        self.branch1 = ConvBNReLU(in_ch, branch_ch, kernel_size=1)
        self.branch2 = nn.Sequential(
            ConvBNReLU(in_ch, branch_ch, kernel_size=1),
            ConvBNReLU(branch_ch, branch_ch, kernel_size=3, padding=1),
        )
        self.branch3 = nn.Sequential(
            ConvBNReLU(in_ch, branch_ch, kernel_size=1),
            ConvBNReLU(branch_ch, branch_ch, kernel_size=5, padding=2),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBNReLU(in_ch, branch_ch, kernel_size=1),
        )
        self.out_channels = branch_ch * 4

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x),
                          self.branch3(x), self.branch4(x)], dim=1)


class DilatedConvFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.r1 = ConvBNReLU(channels, channels, 3, padding=1, dilation=1)
        self.r2 = ConvBNReLU(channels, channels, 3, padding=2, dilation=2)
        self.r4 = ConvBNReLU(channels, channels, 3, padding=4, dilation=4)
        self.fuse = ConvBNReLU(channels * 3, channels, kernel_size=1)

    def forward(self, x):
        return self.fuse(torch.cat([self.r1(x), self.r2(x), self.r4(x)], dim=1))


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = x * self.channel_att(x)
        return x * self.spatial_att(x)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, channels), nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        s = self.fc(self.pool(x).view(b, c)).view(b, c, 1, 1)
        return x * s


class ResBlockSE(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SEBlock(out_ch, reduction)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = None
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = x if self.shortcut is None else self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.se(self.bn2(self.conv2(out)))
        return self.relu(out + identity)


class CNNBranch(nn.Module):
    def __init__(self, in_ch=3, stem_ch=64, inception_branch_ch=64, deep_ch=512):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, stem_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_ch), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.inception = InceptionMultiScale(stem_ch, branch_ch=inception_branch_ch)
        inception_out = self.inception.out_channels
        self.dilated = DilatedConvFusion(inception_out)
        self.cbam = CBAM(inception_out)
        self.res_blocks = nn.Sequential(
            ResBlockSE(inception_out, deep_ch, stride=2),
            ResBlockSE(deep_ch, deep_ch, stride=1),
            ResBlockSE(deep_ch, deep_ch, stride=1),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.out_dim = deep_ch

    def forward(self, x):
        x = self.stem(x)
        x = self.inception(x)
        x = self.dilated(x)
        x = self.cbam(x)
        x = self.res_blocks(x)
        return self.gap(x).flatten(1)  # (B, deep_ch)


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_ch=3, embed_dim=384, dropout=0.1):
        super().__init__()
        assert img_size % patch_size == 0, "img_size doit être divisible par patch_size"
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        b = x.shape[0]
        x = self.proj(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        return self.dropout(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim=384, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop1(attn_out)
        return x + self.drop2(self.ffn(self.norm2(x)))


class ViTBranch(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_ch=3, embed_dim=384,
                 depth=6, num_heads=8, mlp_ratio=4.0, dropout=0.1, out_dim=512):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_ch, embed_dim, dropout)
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, out_dim)

    def forward(self, x):
        x = self.patch_embed(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.proj(x[:, 0]), x[:, 1:]  # (cls_out, patch_tokens)


class CrossAttentionFusion(nn.Module):
    def __init__(self, cnn_dim=512, vit_token_dim=384, num_heads=8,
                 dropout=0.1, fusion_dim=1024):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=cnn_dim, kdim=vit_token_dim, vdim=vit_token_dim,
            num_heads=num_heads, dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(cnn_dim)
        self.fuse_fc = nn.Linear(cnn_dim * 2, fusion_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, cnn_feat, vit_patch_tokens):
        q = cnn_feat.unsqueeze(1)
        attn_out, attn_weights = self.cross_attn(
            q, vit_patch_tokens, vit_patch_tokens, need_weights=True
        )
        attn_out = self.norm(attn_out.squeeze(1))
        fused = self.act(self.fuse_fc(torch.cat([cnn_feat, attn_out], dim=-1)))
        return self.dropout(fused), attn_weights


class ClassificationHead(nn.Module):
    def __init__(self, in_dim=1024, hidden1=1024, hidden2=512,
                 num_classes=23, drop1=0.4, drop2=0.3):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden1)
        self.drop1 = nn.Dropout(drop1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.drop2 = nn.Dropout(drop2)
        self.fc3 = nn.Linear(hidden2, num_classes)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.drop1(self.act(self.fc1(x)))
        x = self.drop2(self.act(self.fc2(x)))
        return self.fc3(x)


class HybridNet(nn.Module):
    def __init__(self, num_classes=23, img_size=224, cnn_deep_ch=512,
                 vit_embed_dim=384, vit_depth=6, vit_heads=8, vit_dropout=0.1,
                 fusion_dropout=0.1, head_drop1=0.4, head_drop2=0.3):
        super().__init__()
        self.cnn_branch = CNNBranch(in_ch=3, deep_ch=cnn_deep_ch)
        self.vit_branch = ViTBranch(img_size=img_size, patch_size=16, in_ch=3,
                                    embed_dim=vit_embed_dim, depth=vit_depth,
                                    num_heads=vit_heads, dropout=vit_dropout,
                                    out_dim=cnn_deep_ch)
        self.fusion = CrossAttentionFusion(
            cnn_dim=cnn_deep_ch, vit_token_dim=vit_embed_dim,
            num_heads=vit_heads, dropout=fusion_dropout, fusion_dim=1024,
        )
        self.head = ClassificationHead(
            in_dim=1024, hidden1=1024, hidden2=512,
            num_classes=num_classes, drop1=head_drop1, drop2=head_drop2,
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, return_attention=False):
        cnn_feat = self.cnn_branch(x)
        _, patches = self.vit_branch(x)
        fused, attn_w = self.fusion(cnn_feat, patches)
        logits = self.head(fused)
        return (logits, attn_w) if return_attention else logits


def build_model(num_classes=23, **kwargs):
    return HybridNet(num_classes=num_classes, **kwargs)
