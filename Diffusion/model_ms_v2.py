import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F


# ------------------------
# Basic blocks
# ------------------------

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class ZeroLinear(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.out_features = out_features

    def forward(self, x):
        return torch.zeros(
            x.size(0), self.out_features,
            device=x.device, dtype=x.dtype
        )


# ------------------------
# Embeddings
# ------------------------

class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        super().__init__()
        assert d_model % 2 == 0

        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        emb = emb.view(T, d_model)

        self.net = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                init.zeros_(m.bias)

    def forward(self, t):
        return self.net(t)


class WindowEmbedding(nn.Module):
    def __init__(self, num_window, d_model, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Embedding(num_window + 1, d_model, padding_idx=0),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )

    def forward(self, w):
        return self.net(w)


# ------------------------
# UNet blocks
# ------------------------

class DownSample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, 2, 1)
        init.xavier_uniform_(self.conv.weight)
        init.zeros_(self.conv.bias)

    def forward(self, x, *_):
        return self.conv(x)


class UpSample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, 1, 1)
        init.xavier_uniform_(self.conv.weight)
        init.zeros_(self.conv.bias)

    def forward(self, x, *_):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class AttnBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(32, ch)
        self.q = nn.Conv2d(ch, ch, 1)
        self.k = nn.Conv2d(ch, ch, 1)
        self.v = nn.Conv2d(ch, ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

        for m in [self.q, self.k, self.v, self.proj]:
            init.xavier_uniform_(m.weight)
            init.zeros_(m.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)

        q = self.q(h).reshape(B, C, H * W).permute(0, 2, 1)
        k = self.k(h).reshape(B, C, H * W)
        v = self.v(h).reshape(B, C, H * W).permute(0, 2, 1)

        attn = torch.bmm(q, k) * (C ** -0.5)
        attn = attn.softmax(dim=-1)

        h = torch.bmm(attn, v)
        h = h.permute(0, 2, 1).reshape(B, C, H, W)
        return x + self.proj(h)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False, window=True):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
        )

        self.temb = nn.Sequential(Swish(), nn.Linear(tdim, out_ch))

        self.wemb = (
            nn.Sequential(Swish(), nn.Linear(tdim, out_ch))
            if window else ZeroLinear(out_ch)
        )

        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
        )

        self.shortcut = (
            nn.Conv2d(in_ch, out_ch, 1)
            if in_ch != out_ch else nn.Identity()
        )

        self.attn = AttnBlock(out_ch) if attn else nn.Identity()

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(m.weight)
                init.zeros_(m.bias)
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x, temb, wemb):
        h = self.block1(x)
        h += self.temb(temb)[:, :, None, None]
        h += self.wemb(wemb)[:, :, None, None]
        h = self.block2(h)
        h = h + self.shortcut(x)
        return self.attn(h)


# ------------------------
# UNet with dual conditioning
# ------------------------

class UNet(nn.Module):
    def __init__(
        self,
        T,
        ch,
        ch_mult,
        attn,
        num_res_blocks,
        dropout,
        n_window,
        window_embedding,
    ):
        super().__init__()

        tdim = ch * 4
        self.has_window_embedding = window_embedding
        self.time_emb = TimeEmbedding(T, ch, tdim)
        if self.has_window_embedding == 'categorical':
            self.window_embedding = WindowEmbedding(n_window, ch, tdim)
        elif self.has_window_embedding == 'spacial':
            self.window_embedding = TimeEmbedding(n_window,ch , tdim)
        else :
            self.window_embedding = ZeroLinear(ch)

        # separate heads
        self.x_head = nn.Conv2d(1, ch, 3, 1, 1)
        self.cond_head = nn.Conv2d(1, ch, 3, 1, 1)

        self.down = nn.ModuleList()
        self.cond_down = nn.ModuleList()

        chs = [ch]
        now_ch = ch

        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.down.append(
                    ResBlock(
                        now_ch, out_ch, tdim,
                        dropout, attn=(i in attn)
                    )
                )
                self.cond_down.append(
                    ResBlock(
                        now_ch, out_ch, tdim,
                        dropout, attn=False
                    )
                )
                now_ch = out_ch
                chs.append(now_ch)

            if i != len(ch_mult) - 1:
                self.down.append(DownSample(now_ch))
                self.cond_down.append(DownSample(now_ch))
                chs.append(now_ch)

        self.mid = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout),
        ])

        self.up = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.up.append(
                    ResBlock(
                        now_ch + chs.pop() * 2,
                        out_ch, tdim,
                        dropout, attn=(i in attn)
                    )
                )
                now_ch = out_ch
            if i != 0:
                self.up.append(UpSample(now_ch))

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv2d(now_ch, 1, 3, 1, 1),
        )

    def forward(self, x, t, cond, window):
        temb = self.time_emb(t)
        wemb = self.window_embedding(window)

        h = self.x_head(x)
        hc = self.cond_head(cond)

        hs, hcs = [h], [hc]

        for d, cd in zip(self.down, self.cond_down):
            h = d(h, temb, wemb)
            hc = cd(hc, temb, wemb)
            hs.append(h)
            hcs.append(hc)

        for m in self.mid:
            h = m(h, temb, wemb)

        for u in self.up:
            if isinstance(u, ResBlock):
                h = torch.cat([h, hs.pop(), hcs.pop()], dim=1)
            h = u(h, temb, wemb)

        return self.tail(h)
