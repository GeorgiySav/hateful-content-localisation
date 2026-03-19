"""
Core building blocks for the temporal localization backbone.

Ported from ActionFormer (https://github.com/happyharrycn/actionformer_release)
with minimal changes:
 - Removed registry decorators (standalone module)
 - Replaced custom trunc_normal_ with torch.nn.init.trunc_normal_
 - No other functional changes

Differences from vanilla ActionFormer blocks.py:
 - No dependency on .weight_init module
"""
import math
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn


class MaskedConv1D(nn.Module):
    """
    Masked 1D convolution. Interface remains the same as Conv1d.
    Only supports a subset of 1D convs (odd kernel sizes with matching padding).
    The mask is downsampled by nearest-neighbour when stride > 1.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode='zeros'
    ):
        super().__init__()
        # element must be aligned
        assert (kernel_size % 2 == 1) and (kernel_size // 2 == padding)
        self.stride = stride
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, bias, padding_mode
        )
        if bias:
            torch.nn.init.constant_(self.conv.bias, 0.)

    def forward(self, x, mask):
        # x:    (B, C, T)
        # mask: (B, 1, T) bool
        B, C, T = x.size()
        assert T % self.stride == 0

        out_conv = self.conv(x)
        if self.stride > 1:
            out_mask = F.interpolate(
                mask.to(x.dtype), size=out_conv.size(-1), mode='nearest'
            )
        else:
            out_mask = mask.to(x.dtype)

        out_conv = out_conv * out_mask.detach()
        out_mask = out_mask.bool()
        return out_conv, out_mask


class LayerNorm(nn.Module):
    """
    LayerNorm that supports inputs of size (B, C, T).
    Standard nn.LayerNorm expects the normalized axis last; this version
    normalizes along the channel (C) axis, matching ActionFormer convention.
    """
    def __init__(self, num_channels, eps=1e-5, affine=True, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(
                torch.ones([1, num_channels, 1], **factory_kwargs))
            self.bias = nn.Parameter(
                torch.zeros([1, num_channels, 1], **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        assert x.dim() == 3
        assert x.shape[1] == self.num_channels
        mu = torch.mean(x, dim=1, keepdim=True)
        res_x = x - mu
        sigma = torch.mean(res_x ** 2, dim=1, keepdim=True)
        out = res_x / torch.sqrt(sigma + self.eps)
        if self.affine:
            out *= self.weight
            out += self.bias
        return out


def get_sinusoid_encoding(n_position, d_hid):
    """Sinusoid position encoding table, returns (1, d_hid, n_position)."""
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid)
                for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i)
                                for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.FloatTensor(sinusoid_table).unsqueeze(0).transpose(1, 2)


class MaskedMHA(nn.Module):
    """
    Multi-Head Attention with mask (global, non-local).
    Input/output format: (B, C, T).
    """
    def __init__(self, n_embd, n_head, attn_pdrop=0.0, proj_pdrop=0.0):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_channels = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_channels)

        self.key   = nn.Conv1d(n_embd, n_embd, 1)
        self.query = nn.Conv1d(n_embd, n_embd, 1)
        self.value = nn.Conv1d(n_embd, n_embd, 1)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)
        self.proj = nn.Conv1d(n_embd, n_embd, 1)

    def forward(self, x, mask):
        B, C, T = x.size()

        k = self.key(x).view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        q = self.query(x).view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        v = self.value(x).view(B, self.n_head, self.n_channels, -1).transpose(2, 3)

        att = (q * self.scale) @ k.transpose(-2, -1)
        att = att.masked_fill(torch.logical_not(mask[:, :, None, :]), float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        out = att @ (v * mask[:, :, :, None].to(v.dtype))
        out = out.transpose(2, 3).contiguous().view(B, C, -1)
        out = self.proj_drop(self.proj(out)) * mask.to(out.dtype)
        return out, mask


class MaskedMHCA(nn.Module):
    """
    Multi-Head Conv Attention with mask.
    Adds depthwise conv before attention to encode relative position (or downsample).
    """
    def __init__(self, n_embd, n_head, n_qx_stride=1, n_kv_stride=1,
                 attn_pdrop=0.0, proj_pdrop=0.0):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_channels = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_channels)

        assert (n_qx_stride == 1) or (n_qx_stride % 2 == 0)
        assert (n_kv_stride == 1) or (n_kv_stride % 2 == 0)
        self.n_qx_stride = n_qx_stride
        self.n_kv_stride = n_kv_stride

        # query conv (depthwise)
        kernel_size = self.n_qx_stride + 1 if self.n_qx_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        self.query_conv = MaskedConv1D(
            n_embd, n_embd, kernel_size,
            stride=stride, padding=padding, groups=n_embd, bias=False
        )
        self.query_norm = LayerNorm(n_embd)

        # key/value conv (depthwise)
        kernel_size = self.n_kv_stride + 1 if self.n_kv_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        self.key_conv = MaskedConv1D(
            n_embd, n_embd, kernel_size,
            stride=stride, padding=padding, groups=n_embd, bias=False
        )
        self.key_norm = LayerNorm(n_embd)
        self.value_conv = MaskedConv1D(
            n_embd, n_embd, kernel_size,
            stride=stride, padding=padding, groups=n_embd, bias=False
        )
        self.value_norm = LayerNorm(n_embd)

        self.key   = nn.Conv1d(n_embd, n_embd, 1)
        self.query = nn.Conv1d(n_embd, n_embd, 1)
        self.value = nn.Conv1d(n_embd, n_embd, 1)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)
        self.proj = nn.Conv1d(n_embd, n_embd, 1)

    def forward(self, x, mask):
        B, C, T = x.size()

        q, qx_mask = self.query_conv(x, mask)
        q = self.query_norm(q)
        k, kv_mask = self.key_conv(x, mask)
        k = self.key_norm(k)
        v, _ = self.value_conv(x, mask)
        v = self.value_norm(v)

        q = self.query(q)
        k = self.key(k)
        v = self.value(v)

        k = k.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        q = q.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        v = v.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)

        att = (q * self.scale) @ k.transpose(-2, -1)
        att = att.masked_fill(torch.logical_not(kv_mask[:, :, None, :]), float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        out = att @ (v * kv_mask[:, :, :, None].to(v.dtype))
        out = out.transpose(2, 3).contiguous().view(B, C, -1)
        out = self.proj_drop(self.proj(out)) * qx_mask.to(out.dtype)
        return out, qx_mask


class LocalMaskedMHCA(nn.Module):
    """
    Local Multi-Head Conv Attention with mask (Longformer-style sliding window).
    window_size must be odd and > 1.

    NOTE: The input sequence length T (after the internal depthwise conv stride)
    must be divisible by (window_size // 2) * 2.  Ensure max_seq_len and all
    downsampled lengths satisfy this when configuring the model.
    """
    def __init__(self, n_embd, n_head, window_size,
                 n_qx_stride=1, n_kv_stride=1,
                 attn_pdrop=0.0, proj_pdrop=0.0, use_rel_pe=False):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_channels = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_channels)
        self.window_size = window_size
        self.window_overlap = window_size // 2
        assert self.window_size > 1 and self.n_head >= 1
        self.use_rel_pe = use_rel_pe

        assert (n_qx_stride == 1) or (n_qx_stride % 2 == 0)
        assert (n_kv_stride == 1) or (n_kv_stride % 2 == 0)
        self.n_qx_stride = n_qx_stride
        self.n_kv_stride = n_kv_stride

        # query conv (depthwise)
        kernel_size = self.n_qx_stride + 1 if self.n_qx_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        self.query_conv = MaskedConv1D(
            n_embd, n_embd, kernel_size,
            stride=stride, padding=padding, groups=n_embd, bias=False
        )
        self.query_norm = LayerNorm(n_embd)

        # key/value conv (depthwise)
        kernel_size = self.n_kv_stride + 1 if self.n_kv_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        self.key_conv = MaskedConv1D(
            n_embd, n_embd, kernel_size,
            stride=stride, padding=padding, groups=n_embd, bias=False
        )
        self.key_norm = LayerNorm(n_embd)
        self.value_conv = MaskedConv1D(
            n_embd, n_embd, kernel_size,
            stride=stride, padding=padding, groups=n_embd, bias=False
        )
        self.value_norm = LayerNorm(n_embd)

        self.key   = nn.Conv1d(n_embd, n_embd, 1)
        self.query = nn.Conv1d(n_embd, n_embd, 1)
        self.value = nn.Conv1d(n_embd, n_embd, 1)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)
        self.proj = nn.Conv1d(n_embd, n_embd, 1)

        if self.use_rel_pe:
            self.rel_pe = nn.Parameter(
                torch.zeros(1, 1, self.n_head, self.window_size))
            nn.init.trunc_normal_(self.rel_pe, std=(2.0 / n_embd) ** 0.5)

    @staticmethod
    def _chunk(x, window_overlap):
        """Convert into overlapping chunks of size 2w with overlap w."""
        x = x.view(x.size(0), x.size(1) // (window_overlap * 2),
                    window_overlap * 2, x.size(2))
        chunk_size = list(x.size())
        chunk_size[1] = chunk_size[1] * 2 - 1
        chunk_stride = list(x.stride())
        chunk_stride[1] = chunk_stride[1] // 2
        return x.as_strided(size=chunk_size, stride=chunk_stride)

    @staticmethod
    def _pad_and_transpose_last_two_dims(x, padding):
        x = F.pad(x, padding)
        x = x.view(*x.size()[:-2], x.size(-1), x.size(-2))
        return x

    @staticmethod
    def _mask_invalid_locations(input_tensor, affected_seq_len):
        beginning_mask_2d = input_tensor.new_ones(
            affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        beginning_mask = beginning_mask_2d[None, :, None, :]
        ending_mask = beginning_mask.flip(dims=(1, 3))
        beginning_input = input_tensor[:, :affected_seq_len, :, :affected_seq_len + 1]
        beginning_mask = beginning_mask.expand(beginning_input.size())
        beginning_input.masked_fill_(beginning_mask == 1, -float("inf"))
        ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1):]
        ending_mask = ending_mask.expand(ending_input.size())
        ending_input.masked_fill_(ending_mask == 1, -float("inf"))

    @staticmethod
    def _pad_and_diagonalize(x):
        total_num_heads, num_chunks, window_overlap, hidden_dim = x.size()
        x = F.pad(x, (0, window_overlap + 1))
        x = x.view(total_num_heads, num_chunks, -1)
        x = x[:, :, :-window_overlap]
        x = x.view(total_num_heads, num_chunks, window_overlap, window_overlap + hidden_dim)
        x = x[:, :, :, :-1]
        return x

    def _sliding_chunks_query_key_matmul(self, query, key, num_heads, window_overlap):
        bnh, seq_len, head_dim = query.size()
        batch_size = bnh // num_heads
        assert seq_len % (window_overlap * 2) == 0
        assert query.size() == key.size()
        chunks_count = seq_len // window_overlap - 1

        chunk_query = self._chunk(query, window_overlap)
        chunk_key = self._chunk(key, window_overlap)
        diagonal_chunked_attention_scores = torch.einsum(
            "bcxd,bcyd->bcxy", (chunk_query, chunk_key))
        diagonal_chunked_attention_scores = self._pad_and_transpose_last_two_dims(
            diagonal_chunked_attention_scores, padding=(0, 0, 0, 1))

        diagonal_attention_scores = diagonal_chunked_attention_scores.new_empty(
            (batch_size * num_heads, chunks_count + 1, window_overlap, window_overlap * 2 + 1))

        diagonal_attention_scores[:, :-1, :, window_overlap:] = \
            diagonal_chunked_attention_scores[:, :, :window_overlap, :window_overlap + 1]
        diagonal_attention_scores[:, -1, :, window_overlap:] = \
            diagonal_chunked_attention_scores[:, -1, window_overlap:, :window_overlap + 1]
        diagonal_attention_scores[:, 1:, :, :window_overlap] = \
            diagonal_chunked_attention_scores[:, :, -(window_overlap + 1):-1, window_overlap + 1:]
        diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = \
            diagonal_chunked_attention_scores[:, 0, :window_overlap - 1, 1 - window_overlap:]

        diagonal_attention_scores = diagonal_attention_scores.view(
            batch_size, num_heads, seq_len, 2 * window_overlap + 1).transpose(2, 1)
        self._mask_invalid_locations(diagonal_attention_scores, window_overlap)
        return diagonal_attention_scores

    def _sliding_chunks_matmul_attn_probs_value(self, attn_probs, value, num_heads, window_overlap):
        bnh, seq_len, head_dim = value.size()
        batch_size = bnh // num_heads
        assert seq_len % (window_overlap * 2) == 0
        assert attn_probs.size(3) == 2 * window_overlap + 1
        chunks_count = seq_len // window_overlap - 1

        chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
            batch_size * num_heads, seq_len // window_overlap, window_overlap, 2 * window_overlap + 1)

        padded_value = F.pad(value, (0, 0, window_overlap, window_overlap), value=-1)
        chunked_value_size = (batch_size * num_heads, chunks_count + 1, 3 * window_overlap, head_dim)
        chunked_value_stride = padded_value.stride()
        chunked_value_stride = (
            chunked_value_stride[0],
            window_overlap * chunked_value_stride[1],
            chunked_value_stride[1],
            chunked_value_stride[2],
        )
        chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        chunked_attn_probs = self._pad_and_diagonalize(chunked_attn_probs)
        context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        return context.view(batch_size, num_heads, seq_len, head_dim)

    def forward(self, x, mask):
        B, C, T = x.size()

        q, qx_mask = self.query_conv(x, mask)
        q = self.query_norm(q)
        k, kv_mask = self.key_conv(x, mask)
        k = self.key_norm(k)
        v, _ = self.value_conv(x, mask)
        v = self.value_norm(v)

        q = self.query(q)
        k = self.key(k)
        v = self.value(v)

        q = q.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        k = k.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        v = v.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)

        q = q.view(B * self.n_head, -1, self.n_channels).contiguous()
        k = k.view(B * self.n_head, -1, self.n_channels).contiguous()
        v = v.view(B * self.n_head, -1, self.n_channels).contiguous()

        q *= self.scale
        att = self._sliding_chunks_query_key_matmul(q, k, self.n_head, self.window_overlap)
        if self.use_rel_pe:
            att += self.rel_pe

        inverse_kv_mask = torch.logical_not(kv_mask[:, :, :, None].view(B, -1, 1))
        float_inverse_kv_mask = inverse_kv_mask.type_as(q).masked_fill(inverse_kv_mask, -1e4)
        diagonal_mask = self._sliding_chunks_query_key_matmul(
            float_inverse_kv_mask.new_ones(size=float_inverse_kv_mask.size()),
            float_inverse_kv_mask, 1, self.window_overlap)
        att += diagonal_mask

        att = F.softmax(att, dim=-1)
        att = att.masked_fill(
            torch.logical_not(kv_mask.squeeze(1)[:, :, None, None]), 0.0)
        att = self.attn_drop(att)

        out = self._sliding_chunks_matmul_attn_probs_value(att, v, self.n_head, self.window_overlap)
        out = out.transpose(2, 3).contiguous().view(B, C, -1)
        out = self.proj_drop(self.proj(out)) * qx_mask.to(out.dtype)
        return out, qx_mask


class TransformerBlock(nn.Module):
    """
    Pre-LayerNorm Transformer block with optional local attention and downsampling.
    Uses AffineDropPath for learnable per-channel scaling (from ActionFormer).
    """
    def __init__(
        self,
        n_embd,
        n_head,
        n_ds_strides=(1, 1),
        n_out=None,
        n_hidden=None,
        act_layer=nn.GELU,
        attn_pdrop=0.0,
        proj_pdrop=0.0,
        path_pdrop=0.0,
        mha_win_size=-1,
        use_rel_pe=False,
    ):
        super().__init__()
        assert len(n_ds_strides) == 2
        self.ln1 = LayerNorm(n_embd)
        self.ln2 = LayerNorm(n_embd)

        if mha_win_size > 1:
            self.attn = LocalMaskedMHCA(
                n_embd, n_head,
                window_size=mha_win_size,
                n_qx_stride=n_ds_strides[0],
                n_kv_stride=n_ds_strides[1],
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
                use_rel_pe=use_rel_pe,
            )
        else:
            self.attn = MaskedMHCA(
                n_embd, n_head,
                n_qx_stride=n_ds_strides[0],
                n_kv_stride=n_ds_strides[1],
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
            )

        if n_ds_strides[0] > 1:
            kernel_size = n_ds_strides[0] + 1
            stride = n_ds_strides[0]
            padding = (n_ds_strides[0] + 1) // 2
            self.pool_skip = nn.MaxPool1d(kernel_size, stride=stride, padding=padding)
        else:
            self.pool_skip = nn.Identity()

        if n_hidden is None:
            n_hidden = 4 * n_embd
        if n_out is None:
            n_out = n_embd

        self.mlp = nn.Sequential(
            nn.Conv1d(n_embd, n_hidden, 1),
            act_layer(),
            nn.Dropout(proj_pdrop, inplace=True),
            nn.Conv1d(n_hidden, n_out, 1),
            nn.Dropout(proj_pdrop, inplace=True),
        )

        if path_pdrop > 0.0:
            self.drop_path_attn = AffineDropPath(n_embd, drop_prob=path_pdrop)
            self.drop_path_mlp = AffineDropPath(n_out, drop_prob=path_pdrop)
        else:
            self.drop_path_attn = nn.Identity()
            self.drop_path_mlp = nn.Identity()

    def forward(self, x, mask, pos_embd=None):
        out, out_mask = self.attn(self.ln1(x), mask)
        out_mask_float = out_mask.to(out.dtype)
        out = self.pool_skip(x) * out_mask_float + self.drop_path_attn(out)
        out = out + self.drop_path_mlp(self.mlp(self.ln2(out)) * out_mask_float)
        if pos_embd is not None:
            out += pos_embd * out_mask_float
        return out, out_mask


class Scale(nn.Module):
    """Learnable scalar multiplier for regression range scaling."""
    def __init__(self, init_value=1.0):
        super().__init__()
        self.scale = nn.Parameter(
            torch.tensor(init_value, dtype=torch.float32), requires_grad=True)

    def forward(self, x):
        return x * self.scale


def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()
    return x.div(keep_prob) * mask


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class AffineDropPath(nn.Module):
    """
    Drop Path with a per-channel learnable scale (zero-initialized).
    Improves training stability for deep transformers.
    """
    def __init__(self, num_dim, drop_prob=0.0, init_scale_value=1e-4):
        super().__init__()
        self.scale = nn.Parameter(
            init_scale_value * torch.ones((1, num_dim, 1)), requires_grad=True)
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(self.scale * x, self.drop_prob, self.training)


# ──────────────────────────────────────────────────────────────────────────────
# TemporalMaxer block (from arXiv:2303.09055, https://github.com/TuanTNG/TemporalMaxer)
# ──────────────────────────────────────────────────────────────────────────────

class TemporalMaxerBlock(nn.Module):
    """
    Parameter-free temporal context block from TemporalMaxer (arXiv:2303.09055).

    Replaces self-attention with a local MaxPool1D operation. No learnable
    parameters in the pooling itself — only the downstream layers matter.
    Results in 2.8× fewer GMACs and 3× faster inference vs ActionFormer.

    Args:
        kernel_size: MaxPool kernel size (default 3).
        stride     : Downsampling stride (default 2 for pyramid downsampling).
        padding    : Padding to keep temporal length consistent (default 1).
        n_embd     : Feature channel dimension (unused by pooling itself).
    """

    def __init__(self, kernel_size, stride, padding, n_embd):
        super().__init__()
        self.ds_pooling = nn.MaxPool1d(kernel_size, stride=stride, padding=padding)
        self.stride = stride

    def forward(self, x, mask, **kwargs):
        # x:    (B, C, T)
        # mask: (B, 1, T) bool
        if self.stride > 1:
            out_mask = F.interpolate(
                mask.to(x.dtype),
                size=x.size(-1) // self.stride,
                mode='nearest',
            )
        else:
            out_mask = mask

        out = self.ds_pooling(x) * out_mask.to(x.dtype)
        return out, out_mask.bool()


# ──────────────────────────────────────────────────────────────────────────────
# SGP block (from TriDet, CVPR 2023, arXiv:2303.07347, https://github.com/dingfengshi/TriDet)
# ──────────────────────────────────────────────────────────────────────────────

class SGPBlock(nn.Module):
    """
    Scalable-Granularity Perception (SGP) layer from TriDet (CVPR 2023, arXiv:2303.07347).

    Replaces self-attention with a dual-branch depthwise convolutional structure:
      - Instant-level branch : depthwise conv at kernel_size (captures fine-grained features)
      - Window-level branch  : depthwise conv at a larger kernel (up_size ≈ k * kernel_size)
      - Global branch        : channel-wise global average pooling
    All three are combined multiplicatively/additively without cross-channel mixing,
    resolving the "rank loss problem" where attention collapses feature diversity.

    Args:
        n_embd          : Feature channel dimension.
        kernel_size     : Instant-level depthwise conv kernel size (must be odd).
        n_ds_stride     : Downsampling stride (1 = no downsampling).
        k               : Scale factor for window-level kernel size (default 1.5).
        group           : Groups for the FFN MLP conv (default 1 = standard conv).
        n_out           : Output dimension (default = n_embd).
        n_hidden        : Hidden dimension in the FFN MLP (default = 4 * n_embd).
        path_pdrop      : Drop-path rate.
        act_layer       : Activation function class.
        downsample_type : How to downsample when n_ds_stride > 1: 'max' or 'avg'.
        init_conv_vars  : Std of Gaussian init for depthwise conv weights.
    """

    def __init__(
        self,
        n_embd,
        kernel_size=3,
        n_ds_stride=1,
        k=1.5,
        group=1,
        n_out=None,
        n_hidden=None,
        path_pdrop=0.0,
        act_layer=nn.GELU,
        downsample_type='max',
        init_conv_vars=1,
    ):
        super().__init__()
        assert kernel_size % 2 == 1

        self.kernel_size = kernel_size
        self.stride = n_ds_stride

        if n_out is None:
            n_out = n_embd

        self.ln = LayerNorm(n_embd)
        self.gn = nn.GroupNorm(16, n_embd)

        # Window-level kernel size (larger, odd)
        up_size = round((kernel_size + 1) * k)
        up_size = up_size + 1 if up_size % 2 == 0 else up_size

        # Depthwise convs for the two branches
        self.psi      = nn.Conv1d(n_embd, n_embd, kernel_size, stride=1,
                                  padding=kernel_size // 2, groups=n_embd)
        self.fc       = nn.Conv1d(n_embd, n_embd, 1, stride=1,
                                  padding=0, groups=n_embd)
        self.convw    = nn.Conv1d(n_embd, n_embd, kernel_size, stride=1,
                                  padding=kernel_size // 2, groups=n_embd)
        self.convkw   = nn.Conv1d(n_embd, n_embd, up_size, stride=1,
                                  padding=up_size // 2, groups=n_embd)
        self.global_fc = nn.Conv1d(n_embd, n_embd, 1, stride=1,
                                   padding=0, groups=n_embd)

        # Downsampling
        if n_ds_stride > 1:
            if downsample_type == 'max':
                ds_kernel = n_ds_stride + 1
                ds_padding = (n_ds_stride + 1) // 2
                self.downsample = nn.MaxPool1d(ds_kernel, stride=n_ds_stride,
                                               padding=ds_padding)
                self.stride = n_ds_stride
            elif downsample_type == 'avg':
                self.downsample = nn.Sequential(
                    nn.AvgPool1d(n_ds_stride, stride=n_ds_stride, padding=0),
                    nn.Conv1d(n_embd, n_embd, 1, 1, 0),
                )
                self.stride = n_ds_stride
            else:
                raise NotImplementedError(f"downsample_type '{downsample_type}' not supported")
        else:
            self.downsample = nn.Identity()
            self.stride = 1

        # FFN MLP
        if n_hidden is None:
            n_hidden = 4 * n_embd

        self.mlp = nn.Sequential(
            nn.Conv1d(n_embd, n_hidden, 1, groups=group),
            act_layer(),
            nn.Conv1d(n_hidden, n_out, 1, groups=group),
        )

        # Drop-path regularisation
        if path_pdrop > 0.0:
            self.drop_path_out = AffineDropPath(n_embd, drop_prob=path_pdrop)
            self.drop_path_mlp = AffineDropPath(n_out,  drop_prob=path_pdrop)
        else:
            self.drop_path_out = nn.Identity()
            self.drop_path_mlp = nn.Identity()

        self.act = act_layer()
        self._reset_params(init_conv_vars)

    def _reset_params(self, init_conv_vars):
        for m in [self.psi, self.fc, self.convw, self.convkw, self.global_fc]:
            nn.init.normal_(m.weight, 0, init_conv_vars)
            nn.init.constant_(m.bias, 0)

    def forward(self, x, mask):
        # x:    (B, C, T)
        # mask: (B, 1, T) bool
        B, C, T = x.shape
        x = self.downsample(x)
        out_mask = F.interpolate(
            mask.to(x.dtype),
            size=torch.div(T, self.stride, rounding_mode='trunc'),
            mode='nearest',
        ).detach()

        out = self.ln(x)
        psi     = self.psi(out)
        fc      = self.fc(out)
        convw   = self.convw(out)
        convkw  = self.convkw(out)
        phi     = torch.relu(self.global_fc(out.mean(dim=-1, keepdim=True)))
        out     = fc * phi + (convw + convkw) * psi + out

        out = x * out_mask + self.drop_path_out(out)
        out = out + self.drop_path_mlp(self.mlp(self.gn(out)))

        return out, out_mask.bool()
