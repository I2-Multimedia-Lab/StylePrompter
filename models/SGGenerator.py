import math
import torch
from torch import nn
from torch.nn import functional as F
from models.stylegan2.model import Generator


class SGGenerator(nn.Module):

    def __init__(
            self,
            n_latent=18,
            refined_layer=4,
            fpn_shape=[8, 8, 16, 32],
            fpn_channels=[768, 768, 384, 192],
            ffn_dim=1024,
            head_num=4,
            dropout=0.1,
            channel_multiplier=2,
            output_size=1024
    ):
        super().__init__()

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.generator = Generator(output_size, 512, 8)
        self.frozen_generator()

        self.n_latent = n_latent

        self.refined_layer = refined_layer
        print(f'Refine for layer {self.refined_layer} at resolution {2 ** ((self.refined_layer + 4) // 2)}')

        self.smart_block = SMART(
            kv_channels=fpn_channels,
            kv_shape=fpn_shape,
            q_channel=self.channels[2 ** ((self.refined_layer + 4) // 2)],
            q_shape=2 ** ((self.refined_layer + 4) // 2),
            embed_dim=self.channels[2 ** ((self.refined_layer + 4) // 2)],
            ffn_dim=ffn_dim,
            head_num=head_num,
            dropout=dropout
        )

    def frozen_generator(self):
        for p in self.generator.parameters():
            p.requires_grad = False

    def get_trainable_params(self):
        return list(self.smart_block.parameters())

    def make_noise(self):
        return self.generator.make_noise()

    def forward(self,
                latent,
                features=None,
                noise=None,
                input_is_latent=True,
                randomize_noise=True,
                beta=1.0,
                flow=None
                ):
        # codes: latent codes (B, 18, 512)
        # features: list, 0: (B, 8x8, 768); 1:(B, 8x8, 768); 2:(B, 16x16, 384); 3:(B, 32x32, 192)

        if not input_is_latent:
            latent = self.generator.style(latent)
            if latent.ndim < 3:
                latent = latent.unsqueeze(1).repeat(1, self.n_latent, 1)

        B, n, _ = latent.shape

        if noise is None:
            if randomize_noise:
                noise = [None] * self.generator.num_layers
            else:
                noise = [
                    getattr(self.generator.noises, f'noise_{i}') for i in range(self.generator.num_layers)
                ]

        out = self.generator.input(latent)
        out = self.generator.conv1(out, latent[:, 0], noise=noise[0])

        skip = self.generator.to_rgb1(out, latent[:, 1])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
                self.generator.convs[::2], self.generator.convs[1::2], noise[1::2], noise[2::2], self.generator.to_rgbs
        ):

            if i == self.refined_layer:
                if features is not None:
                    B, C, H, W = out.shape
                    pre = out.view(B, C, H * W).transpose(1, 2)  # (B, L, C)
                    post = self.smart_block(pre, features, beta=beta, flow=flow)
                    out = post.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

            out = conv1(out, latent[:, i], noise=noise1)  # (B, 512, 8, 8)
            out = conv2(out, latent[:, i + 1], noise=noise2)

            # refined_layer = 4 (index start from 0) for official, equals to F_5 in the paper (index start from 1)
            if i + 1 == self.refined_layer:
                if features is not None:
                    B, C, H, W = out.shape
                    pre = out.view(B, C, H * W).transpose(1, 2)  # (B, L, C)
                    post = self.smart_block(pre, features, beta=beta, flow=flow)
                    out = post.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

            skip = to_rgb(out, latent[:, i + 2], skip)

            i += 2

        image = skip

        return image


class SMART(nn.Module):
    def __init__(
            self,
            kv_channels,
            kv_shape,
            q_channel,
            q_shape,
            embed_dim=512,
            ffn_dim=2048,
            head_num=4,
            dropout=0.1,
            activation='relu'
    ):
        super().__init__()

        self.head_num = head_num
        self.embed_dim = embed_dim
        self.fpn_num = len(kv_channels)
        self.fpn_shape = kv_shape
        self.q_channel = q_channel
        self.q_shape = q_shape
        self.attn_idx = self.get_attn_idx(self.q_shape, self.fpn_shape)

        # cross attention
        self.query_proj = nn.Linear(q_channel, embed_dim)
        self.input_proj = nn.ModuleList()
        for i in range(self.fpn_num):
            # feature maps encoder
            self.input_proj.append(
                nn.Sequential(
                    nn.Linear(kv_channels[i], embed_dim * 2),
                )
            )
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        # ffn
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ffn_dim, q_channel)
        self.dropout3 = nn.Dropout(dropout)

        # self.__init_parameters()

    def __init_parameters(self):
        for p in self.parameters():
            torch.nn.init.constant_(p, 1e-3)

    def get_attn_idx(self, q_shape, fpn_shape):
        table_list = []
        spatial_shapes = torch.as_tensor(fpn_shape) ** 2
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.cumsum(0)[:-1]))

        for i in range(self.fpn_num):
            s = fpn_shape[i]
            if s <= q_shape:
                table = torch.stack(torch.arange(s * s).chunk(s, dim=0)).unsqueeze(0).unsqueeze(0).float()
                table = F.interpolate(table, (q_shape, q_shape), mode='area')
                table_list.append((table.int().squeeze() + level_start_index[i]).unsqueeze(-1))
            else:
                radio = s // q_shape
                h_idx = torch.arange(s).repeat(s)
                w_idx = torch.arange(s).repeat_interleave(s) * s
                k_idx_1 = torch.arange(radio).repeat(radio)
                k_idx_2 = torch.arange(radio).repeat_interleave(radio) * s
                k_idx = k_idx_1 + k_idx_2
                hw_idx = h_idx + w_idx
                unfold_idx = hw_idx[:, None] + k_idx
                table = torch.cat((unfold_idx[::radio].chunk(s)[::radio]))
                table_list.append(table.view(q_shape, q_shape, -1) + level_start_index[i])

        return torch.cat(table_list, dim=-1).view(q_shape*q_shape, -1).long()

    def get_attn_idx_with_flow(self, q_shape, fpn_shape, flow):
        fpn_num = len(fpn_shape)
        table_list = []
        spatial_shapes = torch.as_tensor(fpn_shape) ** 2
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.cumsum(0)[:-1]))

        x_offset, y_offset = flow[0]
        q_h_idx = torch.round(torch.arange(q_shape, device="cuda").repeat(q_shape) + x_offset.flatten()).long()
        q_w_idx = torch.round(torch.arange(q_shape, device="cuda").repeat_interleave(q_shape) + y_offset.flatten()).long() * q_shape
        q_hw_shape = q_h_idx + q_w_idx
        q_hw_shape[q_hw_shape < 0] = 0
        q_hw_shape[q_hw_shape >= q_shape**2] = q_shape - 1

        # print(q_hw_shape)
        return q_hw_shape

    def forward(self, tgt, features, beta=None, flow=None, mask=None):
        # print('Attention Start')
        # tgt: (B, H*W, C), features: (B, h*w, c)

        if flow is not None:
            attn_idx = self.attn_idx[self.get_attn_idx_with_flow(self.q_shape, self.fpn_shape, flow)]
            # attn_idx = self.get_attn_idx_with_flow(self.q_shape, self.fpn_shape, flow)
        else:
            attn_idx = self.attn_idx

        if beta is None:
            _beta1 = 1.0
            _beta2 = 1.0
        else:
            if type(beta) in [tuple, list]:
                assert len(beta) == 2, "value number should no more than two"
                _beta1 = beta[0]
                _beta2 = beta[1]
            else:
                _beta1 = beta
                _beta2 = beta

        # generator feature map as query or backbone feature map as query
        B, L, C = tgt.shape
        shortcut = tgt
        query = self.query_proj(tgt).view(B, L, self.head_num, C // self.head_num).permute(0, 2, 1, 3)

        k = []
        v = []
        spatial_shapes = []
        radio = []
        for i in range(self.fpn_num):
            _, l, _ = features[i].shape
            spatial_shapes.append(l)
            radio.append(L/l)

            _k, _v = self.input_proj[i](features[i]).chunk(2, -1)
            k.append(_k.view(-1, l, self.head_num, C // self.head_num).transpose(1, 2))
            v.append(_v.view(-1, l, self.head_num, C // self.head_num).transpose(1, 2))

        k = torch.cat(k, dim=2)  # (B, hn, L, C)
        v = torch.cat(v, dim=2) * _beta1  # (B, hn, L, C)

        attn = query.unsqueeze(3) @ k[:, :, attn_idx, :].transpose(-1, -2)

        x = (attn @ v[:, :, attn_idx]).squeeze(3).transpose(1, 2).contiguous().view(B, -1, C)
        x = shortcut + self.dropout1(self.proj(x))

        x = x + _beta2 * self.dropout3(self.linear2(self.dropout2(self.activation(self.linear1(x)))))
        # print('Attention End')
        return x


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
