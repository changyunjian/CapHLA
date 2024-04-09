import torch
import math
from torch import nn

r'''
conformer model remove first feedforward model and second feedforward

convolution model(pointwise | depthwise | pointwise) +
multiself attention

flatten + feature selection (full connection)
'''

class DotProductionAttention(nn.Module):
    def __init__(self, dropout, **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        d = queries.shape[-1]
        score = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention = nn.functional.softmax(score, dim=-1)
        return torch.bmm(self.dropout(self.attention), values)


def transpose_qkv(X, num_heads):
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    X = X.permute(0, 2, 1, 3)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


class MultiHeadAttention(nn.Module):
    def __init__(self, vocab_size, num_heads,
                 dropout, bias=False, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductionAttention(dropout)
        self.wq = nn.Linear(vocab_size, vocab_size*num_heads, bias=bias)
        self.kw = nn.Linear(vocab_size, vocab_size*num_heads, bias=bias)
        self.wv = nn.Linear(vocab_size, vocab_size*num_heads, bias=bias)
        self.wo = nn.Linear(vocab_size*num_heads, vocab_size,  bias=bias)

    def forward(self, queries, keys, values):
        queries = transpose_qkv(self.wq(queries), self.num_heads)
        keys = transpose_qkv(self.kw(keys), self.num_heads)
        values = transpose_qkv(self.wv(values), self.num_heads)
        output = self.attention(queries, keys, values)
        output_concat = transpose_output(output, self.num_heads)
        return self.wo(output_concat)
    
class ConvolutionModule(nn.Module):
    def __init__(self, vocab_size, num_channels, depthwise_kernel_size,
                 dropout, bias, use_group_norm):
        super().__init__()
        if (depthwise_kernel_size - 1) % 2 != 0:
            raise ValueError("depthwise_kernel_size must be odd to achieve 'SAME' padding.")
        self.sequential = nn.Sequential(
            nn.Conv1d(
                vocab_size,
                2 * num_channels,
                1,
                stride=1,
                padding=0,
                bias=bias,
            ),
            nn.GLU(dim=1),
            nn.Conv1d(
                num_channels,
                num_channels,
                depthwise_kernel_size,
                stride=1,
                padding=(depthwise_kernel_size - 1) // 2,
                groups=num_channels,
                bias=bias,
            ),
            nn.GroupNorm(num_groups=1, num_channels=num_channels)
            if use_group_norm
            else nn.BatchNorm1d(num_channels),
            torch.nn.SiLU(),
            torch.nn.Conv1d(
                num_channels,
                vocab_size,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
            ),
            torch.nn.Dropout(dropout),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input.transpose(1, 2)
        x = self.sequential(x)
        return x.transpose(1, 2)


class FeatureSelection(torch.nn.Module):
    def __init__(self, input_dim, num_hiddens, dropout=0.2):
        super().__init__()
        self.sequential = torch.nn.Sequential(
            torch.nn.Linear(input_dim, num_hiddens, bias=True),
            torch.nn.SiLU(),
            torch.nn.BatchNorm1d(num_hiddens),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(num_hiddens, 64, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1, bias=True)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.sequential(input)


class CapHLA_BA(nn.Module):
    def __init__(
        self,
        vocab_size=21,
        num_hiddens=600,
        num_heads=9,
        num_step=59,
        num_channels=1600,
        depthwise_kernel_size=9,
        dropout=0.2,
        bias=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.norm = nn.LayerNorm(vocab_size)
        self.selfattention = MultiHeadAttention(vocab_size, num_heads, dropout)
        self.conv = ConvolutionModule(
            vocab_size=vocab_size,
            num_channels=num_channels,
            depthwise_kernel_size=depthwise_kernel_size,
            dropout=dropout,
            bias=True,
            use_group_norm=False,
        )
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.feature_selection = FeatureSelection(
            vocab_size * num_step, num_hiddens)

    def forward(self, pep, mhc):
        X = torch.cat((pep, mhc), dim=1).type(torch.float32)
        residual = X
        X = self.conv(X)
        X = self.norm(residual + X)
        residual = X
        X = self.selfattention(X, X, X)
        X = self.norm(residual + X)
        X = self.flatten(X)
        return self.feature_selection(X).squeeze(-1)