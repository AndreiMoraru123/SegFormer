""" Efficient Vision Transformer (ViT) SegFormer model """

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# Transformers & friends
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_


class PatchEmbedding(nn.Module):
    """ Image to Patch Embedding with overlap"""

    def __init__(self, patch_size, stride, in_channels, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(self.in_channels, self.embed_dim, kernel_size=self.patch_size,
                              stride=self.stride, padding=self.patch_size // 2)
        self.norm = nn.LayerNorm(self.embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
        return x, H, W


class EfficientSelfAttention(nn.Module):
    """ Efficient Self Attention """

    def __init__(self, attn_dim, num_heads, dropout, spatial_reduction_ratio):
        super().__init__()
        assert attn_dim % num_heads == 0, "attn_dim must be divisible by num_heads"
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.sr_ratio = spatial_reduction_ratio

        if self.sr_ratio > 1:
            self.sr = nn.Conv2d(in_channels=self.attn_dim, out_channels=self.attn_dim,
                                kernel_size=self.sr_ratio, stride=self.sr_ratio)
            self.norm = nn.LayerNorm(self.attn_dim)

        # Query, Key, Value projections for all the heads
        self.q = nn.Linear(self.attn_dim, self.attn_dim, bias=True)
        self.kv = nn.Linear(self.attn_dim, self.attn_dim * 2, bias=True)

        # Scale factor
        self.scale = (self.attn_dim // self.num_heads) ** -0.5

        # Linear projection
        self.proj = nn.Linear(self.attn_dim, self.attn_dim)

        # Dropout
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x, H, W):
        Q = self.q(x)
        Q = rearrange(Q, 'b hw (n d) -> b n hw d', n=self.num_heads)

        if self.sr_ratio > 1:
            x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
            x = self.sr(x)
            x = rearrange(x, 'b c h w -> b (h w) c')
            x = self.norm(x)

        K, V = self.kv(x).chunk(2, dim=-1)
        K = rearrange(K, 'b hw (n d) -> b n hw d', n=self.num_heads)
        V = rearrange(V, 'b hw (n d) -> b n hw d', n=self.num_heads)

        # Dot product
        attn = torch.einsum('b n i d, b n j d -> b n i j', Q, K) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        x = torch.einsum('b n i j, b n j d -> b n i d', attn, V)
        x = rearrange(x, 'b n hw d -> b hw (n d)')
        x = self.proj(x)
        x = self.dropout(x)
        attn_output = {'attn': attn, 'query': Q, 'key': K, 'value': V}
        return x, attn_output


class MixFeedForward(nn.Module):
    """ Mix Feed Forward Network """

    def __init__(self, in_features, out_features, hidden_features, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = dropout

        # Depth-wise convolution
        self.conv = nn.Conv2d(in_channels=hidden_features, out_channels=hidden_features,
                              kernel_size=3, bias=True, padding=1, groups=hidden_features)

        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        x = self.conv(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """ Transformer Block: Norm->Self-Attention->Norm->FFN """

    def __init__(self, dim, num_heads, dropout, drop_path, spatial_reduction_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = EfficientSelfAttention(attn_dim=dim, num_heads=num_heads, dropout=dropout,
                                           spatial_reduction_ratio=spatial_reduction_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.ffn = MixFeedForward(in_features=dim, out_features=dim,
                                  hidden_features=dim * 4,
                                  dropout=dropout)

    def forward(self, x, H, W):
        skip = x
        x = self.norm1(x)
        x, attn_output = self.attn(x, H, W)
        x = self.drop_path(x)
        x = x + skip

        skip = x
        x = self.norm2(x)
        x = self.ffn(x, H, W)
        x = self.drop_path(x)
        x = x + skip
        return x, attn_output


class MixTransformerStage(nn.Module):
    """ Mix Transformer Stage: Transformer Block * num_blocks """

    def __init__(self, patch_embed, blocks, norm):
        super().__init__()
        self.patch_embed = patch_embed
        self.blocks = nn.ModuleList(blocks)
        self.norm = norm

    def forward(self, x):
        stage_output = {'patch_embed_input': x}
        x, H, W = self.patch_embed(x)
        stage_output['patch_embed_h'] = H
        stage_output['patch_embed_w'] = W
        stage_output['patch_embed_output'] = x

        for block in self.blocks:
            x, attn_output = block(x, H, W)

        x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

        for k, v in attn_output.items():
            stage_output[k] = v
        del attn_output
        return x, stage_output


class MixTransformer(nn.Module):
    """ Transformer Encoder: A set of 4 consecutive transformer stages """
    def __init__(self, in_channels, embed_dims, num_heads, depths,
                 spatial_reduction_ratios, dropout, drop_path):
        super().__init__()
        self.stages = nn.ModuleList()
        self.num_stages = len(depths)

        for stage in range(self.num_stages):
            blocks = []
            for _ in range(depths[stage]):
                blocks.append(TransformerBlock(spatial_reduction_ratio=spatial_reduction_ratios[stage],
                                               dim=embed_dims[stage], num_heads=num_heads[stage],
                                               dropout=dropout, drop_path=drop_path))
            if stage == 0:
                patch_size = 7
                stride = 4
                in_channels = in_channels
            else:
                patch_size = 3
                stride = 2
                in_channels = embed_dims[stage - 1]

            patch_embed = PatchEmbedding(patch_size=patch_size, stride=stride,
                                         in_channels=in_channels, embed_dim=embed_dims[stage])
            norm = nn.LayerNorm(embed_dims[stage], eps=1e-6)
            self.stages.append(MixTransformerStage(patch_embed=patch_embed, blocks=blocks, norm=norm))

    def forward(self, x):
        outputs = []
        for stage in self.stages:
            x, _ = stage(x)
            outputs.append(x)
        return outputs

    def get_attn_outputs(self, x):
        outputs = []
        for stage in self.stages:
            x, attn_output = stage(x)
            outputs.append(attn_output)
        return outputs


class DecoderHead(nn.Module):
    """ Decoder Head """
    def __init__(self, in_channels, num_classes, embed_dim, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.dropout_p = dropout

        # 1x1 conv to fuse multi-scale output from encoder
        self.layers = nn.ModuleList([nn.Conv2d(in_channels=ch,
                                               out_channels=self.embed_dim,
                                               kernel_size=(1, 1))
                                     for ch in reversed(self.in_channels)])
        self.linear_fuse = nn.Conv2d(in_channels=self.embed_dim * len(self.layers),
                                     out_channels=self.embed_dim,
                                     kernel_size=(1, 1), bias=False)
        self.bn = nn.BatchNorm2d(self.embed_dim, eps=1e-5)

        # 1x1 conv to get num_class channel predictions
        self.linear_pred = nn.Conv2d(self.embed_dim,
                                     self.num_classes,
                                     kernel_size=(1, 1))
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.linear_fuse.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        feature_size = x[0].shape[2:]

        x = [layer(xi) for layer, xi in zip(self.layers, reversed(x))]
        x = [F.interpolate(xi, size=feature_size, mode='bilinear', align_corners=False)
             for xi in x[:-1]] + [x[-1]]  # add last layer without up-sampling b/c it's already correct size

        # concat multiscale features & fuse with 1x1 conv
        x = torch.cat(x, dim=1)
        x = self.linear_fuse(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.linear_pred(x)

        return x


class SegFormer(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.embed_dims = [64, 128, 320, 512]
        self.num_heads = [1, 2, 5, 8]
        self.depths = [3, 4, 18, 3]
        self.spatial_reduction_ratios = [8, 4, 2, 1]
        self.dropout = 0.1
        self.drop_path = 0.1

        self.backbone = MixTransformer(in_channels=self.in_channels, embed_dims=self.embed_dims,
                                       num_heads=self.num_heads, depths=self.depths,
                                       spatial_reduction_ratios=self.spatial_reduction_ratios,
                                       dropout=self.dropout, drop_path=self.drop_path)

        self.decoder = DecoderHead(in_channels=self.embed_dims, num_classes=self.num_classes,
                                   embed_dim=256, dropout=self.dropout)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        image_hw = x.shape[2:]
        x = self.backbone(x)
        x = self.decoder(x)
        x = F.interpolate(x, size=image_hw, mode='bilinear', align_corners=False)
        return x

    def get_attn_outputs(self, x):
        return self.backbone.get_attn_outputs(x)

    def get_last_self_attn(self, x):
        attn_outputs = self.backbone.get_attn_outputs(x)
        return attn_outputs[-1].get('attn', None)