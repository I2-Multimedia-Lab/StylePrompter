import math
from torch.nn import Module
from models.encoders.swin_helpers import PatchEmbed, BasicLayer, PatchMerging
import torch
from torch import nn
from operator import mul
from functools import reduce
import torch.nn.functional as F


class MLP(Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, activation='tanh'):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.activation = activation

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = _get_activation_fn(self.activation)(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class SwinEncoder(Module):
    def __init__(self, tokens_num=18,
                 img_size=224, patch_size=4, in_chans=3,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 pretrained_window_sizes=[0, 0, 0, 0],
                 ffn_dim=1024,
                 style_dim=512):
        super(SwinEncoder, self).__init__()

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.style_dim = style_dim

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               pretrained_window_size=pretrained_window_sizes[i_layer])
            self.layers.append(layer)

        # [64, 32, 16, 8]
        self.layers_input_size = [img_size // 4, img_size // 4 // 2, img_size // 4 // 4, img_size // 4 // 8]
        self.layers_input_dim = [embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8]

        self.tokens_num = tokens_num
        self.latent_tok = nn.Parameter(torch.ones(self.tokens_num, self.embed_dim))

        self.dim_up_layers = nn.ModuleList()
        for i_layer in range(self.num_layers - 1):
            layer = nn.Sequential(
                nn.Linear(self.layers_input_dim[i_layer], self.layers_input_dim[i_layer] * 2),
                norm_layer(self.layers_input_dim[i_layer] * 2)
            )
            self.dim_up_layers.append(layer)

        self.prediction_head = MLP(3, self.layers_input_dim[-1], ffn_dim, self.style_dim)
        self._init_tokens(patch_size, self.embed_dim)

    def frozen(self):
        for p in self.parameters():
            p.requires_grad = False

    def _init_tokens(self, patch_size, dim):
        val = math.sqrt(6. / float(3 * reduce(mul, (patch_size, patch_size), 1) + dim))
        nn.init.uniform_(self.latent_tok.data, -val, val)
        # init_data = torch.randn(1, self.latent_tok.shape[-1])
        # self.latent_tok.data = init_data.repeat(self.tokens_num, 1)

    def forward(self, x):

        b, nc, h, w = x.shape
        # x.shape(B, 3, 256, 256)
        x = self.patch_embed(x)  # (B, 4096, 96)
        x = self.pos_drop(x)  # (B, 4096, 96)

        features = []
        latent_tok = self.latent_tok.unsqueeze(0).repeat(b, 1, 1)  # (B, 18, 96)

        # layer1: (B, 64x64, 96)  -> (B, 32x32, 192)
        # layer2: (B, 32x32, 192) -> (B, 16x16, 384)
        # layer3: (B, 16x16, 384) -> (B, 8x8, 768)
        # layer4: (B, 8x8, 768)   -> (B, 8x8, 768)
        for idx, layer in enumerate(self.layers):
            x, latent_tok = layer(x, tokens=latent_tok)
            features.append(x)
            # (B, 18, 96) -> (B, 18, 192) -> (B, 18, 384) -> (B, 18, 768)
            if idx < self.num_layers - 1:
                latent_tok = self.dim_up_layers[idx](latent_tok)

        latents = self.prediction_head(latent_tok)

        return latents, features[::-1]


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "tanh":
        return torch.tanh
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
