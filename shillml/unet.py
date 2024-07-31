import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels):
        super(GroupNorm, self).__init__()
        self.group_norm = nn.GroupNorm(num_groups, num_channels)

    def forward(self, x):
        return self.group_norm(x)


def nonlinearity(x):
    return F.silu(x)


def normalize(x, temb, name):
    return GroupNorm(32, x.size(1))(x)


def upsample(x, name, with_conv):
    x = F.interpolate(x, scale_factor=2, mode='nearest')
    if with_conv:
        x = nn.Conv2d(x.size(1), x.size(1), kernel_size=3, stride=1, padding=1)(x)
    return x


def downsample(x, name, with_conv):
    if with_conv:
        x = nn.Conv2d(x.size(1), x.size(1), kernel_size=3, stride=2, padding=1)(x)
    else:
        x = F.avg_pool2d(x, 2, 2)
    return x


class ResnetBlock(nn.Module):
    def __init__(self, in_ch, out_ch=None, conv_shortcut=False, dropout=0.0):
        super(ResnetBlock, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch if out_ch else in_ch
        self.conv_shortcut = conv_shortcut
        self.dropout = dropout
        self.conv1 = nn.Conv2d(in_ch, self.out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(self.out_ch, self.out_ch, kernel_size=3, padding=1)
        if conv_shortcut:
            self.shortcut = nn.Conv2d(in_ch, self.out_ch, kernel_size=1)
        elif in_ch != self.out_ch:
            self.shortcut = nn.Conv2d(in_ch, self.out_ch, kernel_size=1)

    def forward(self, x, temb):
        h = nonlinearity(normalize(x, temb, 'norm1'))
        h = self.conv1(h)
        h = h + nonlinearity(temb)[:, :, None, None]
        h = nonlinearity(normalize(h, temb, 'norm2'))
        h = F.dropout(h, self.dropout)
        h = self.conv2(h)
        if self.in_ch != self.out_ch:
            x = self.shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super(AttnBlock, self).__init__()
        self.q = nn.Conv2d(in_ch, in_ch, kernel_size=1)
        self.k = nn.Conv2d(in_ch, in_ch, kernel_size=1)
        self.v = nn.Conv2d(in_ch, in_ch, kernel_size=1)
        self.proj_out = nn.Conv2d(in_ch, in_ch, kernel_size=1)

    def forward(self, x, temb):
        h = normalize(x, temb, 'norm')
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)

        w = torch.einsum('bchw,bChw->bhwC', q, k) * (q.size(1) ** -0.5)
        w = torch.softmax(w.flatten(2), dim=-1).view_as(w)
        h = torch.einsum('bhwC,bChw->bchw', w, v)
        h = self.proj_out(h)

        return x + h


class Model(nn.Module):
    def __init__(self, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks=2, attn_resolutions=(16,), dropout=0.0, resamp_with_conv=True):
        super(Model, self).__init__()
        self.num_resolutions = len(ch_mult)
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.middle = nn.ModuleList()

        curr_res = ch
        for i_level, mult in enumerate(ch_mult):
            self.down.append(nn.Sequential(
                *[ResnetBlock(curr_res, curr_res * mult, dropout=dropout) for _ in range(num_res_blocks)],
                AttnBlock(curr_res * mult) if curr_res in attn_resolutions else nn.Identity(),
                downsample if i_level != self.num_resolutions - 1 else nn.Identity()
            ))
            curr_res *= mult

        self.middle.append(ResnetBlock(curr_res, curr_res, dropout=dropout))
        self.middle.append(AttnBlock(curr_res))
        self.middle.append(ResnetBlock(curr_res, curr_res, dropout=dropout))

        for i_level, mult in reversed(list(enumerate(ch_mult))):
            self.up.append(nn.Sequential(
                *[ResnetBlock(curr_res, curr_res // mult, dropout=dropout) for _ in range(num_res_blocks + 1)],
                AttnBlock(curr_res // mult) if curr_res // mult in attn_resolutions else nn.Identity(),
                upsample if i_level != 0 else nn.Identity()
            ))
            curr_res //= mult

        self.norm_out = nn.GroupNorm(32, curr_res)
        self.conv_out = nn.Conv2d(curr_res, out_ch, kernel_size=3, padding=1)

    def forward(self, x, t):
        temb = get_timestep_embedding(t, x.size(1))
        temb = nonlinearity(nn.Linear(temb.size(-1), temb.size(-1) * 4)(temb))
        temb = nonlinearity(nn.Linear(temb.size(-1), temb.size(-1) * 4)(temb))

        hs = [self.conv_in(x)]
        for layer in self.down:
            hs.append(layer(hs[-1], temb))

        h = self.middle[0](hs[-1], temb)
        h = self.middle[1](h, temb)
        h = self.middle[2](h, temb)

        for layer in self.up:
            h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb)

        h = nonlinearity(self.norm_out(h))
        h = self.conv_out(h)
        return h


def get_timestep_embedding(timesteps, embedding_dim):
    half_dim = embedding_dim // 2
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -math.log(10000) / (half_dim - 1))
    emb = timesteps[:, None].float() * emb[None, :]
    return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
