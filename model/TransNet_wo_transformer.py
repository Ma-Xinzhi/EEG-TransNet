import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange

def attention(query, key, value):
    dim = query.size(-1)
    scores = torch.einsum('bhqd,bhkd->bhqk', query, key) / dim**.5
    attn = F.softmax(scores, dim=-1)
    out = torch.einsum('bhqk,bhkd->bhqd', attn, value)
    return out, attn

class VarPoold(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
    def forward(self, x):
        t = x.shape[2]
        out_shape = (t - self.kernel_size) // self.stride + 1
        out = []

        for i in range(out_shape):
            index = i*self.stride
            input = x[:, :, index:index+self.kernel_size]
            output = torch.log(torch.clamp(input.var(dim=-1, keepdim=True), 1e-6, 1e6))
            out.append(output)

        out = torch.cat(out, dim=-1)

        return out

class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        self.d_k = d_model // n_head
        self.d_v = d_model // n_head
        self.n_head = n_head

        self.w_q = nn.Linear(d_model, n_head*self.d_k)
        self.w_k = nn.Linear(d_model, n_head*self.d_k)
        self.w_v = nn.Linear(d_model, n_head*self.d_v)
        self.w_o = nn.Linear(n_head*self.d_v, d_model)

        self.dropout = nn.Dropout(dropout)

    # [batch_size, n_channel, d_model]
    def forward(self, query, key, value):

        q = rearrange(self.w_q(query), "b n (h d) -> b h n d", h=self.n_head)
        k = rearrange(self.w_k(key), "b n (h d) -> b h n d", h=self.n_head)
        v = rearrange(self.w_v(value), "b n (h d) -> b h n d", h=self.n_head)
        
        out, _ = attention(q, k, v)
        
        out = rearrange(out, 'b h q d -> b q (h d)')
        out = self.dropout(self.w_o(out))

        return out

class FeedForward(nn.Module):
    def __init__(self, d_model, d_hidden, dropout):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_hidden)
        self.act = nn.GELU()
        self.w_2 = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)

        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, fc_ratio, attn_drop=0.5, fc_drop=0.5):
        super().__init__()
        self.multihead_attention = MultiHeadedAttention(embed_dim, num_heads, attn_drop)
        self.feed_forward = FeedForward(embed_dim, embed_dim*fc_ratio, fc_drop)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, data):
        res = self.layernorm1(data)
        out = data + self.multihead_attention(res, res, res)

        res = self.layernorm2(out)
        output = out + self.feed_forward(res)
        return output

class TransNet_wo_transformer(nn.Module):
    def __init__(self, num_classes=4, num_samples=1000, num_channels=22, embed_dim=32, pool_size=50, 
    pool_stride=15, num_heads=8, fc_ratio=4, depth=4, attn_drop=0.5, fc_drop=0.5):
        super().__init__()
        self.temp_conv1 = nn.Conv2d(1, embed_dim//4, (1, 15), padding=(0, 7))
        self.temp_conv2 = nn.Conv2d(1, embed_dim//4, (1, 25), padding=(0, 12))
        self.temp_conv3 = nn.Conv2d(1, embed_dim//4, (1, 51), padding=(0, 25))
        self.temp_conv4 = nn.Conv2d(1, embed_dim//4, (1, 65), padding=(0, 32))
        self.bn1 = nn.BatchNorm2d(embed_dim)
        self.spatial_conv = nn.Conv2d(embed_dim, embed_dim, (num_channels, 1))

        self.bn2 = nn.BatchNorm2d(embed_dim)
        self.elu = nn.ELU()

        self.var_pool = VarPoold(pool_size, pool_stride)
        self.avg_pool = nn.AvgPool1d(pool_size, pool_stride)

        temp_embedding_dim = (num_samples - pool_size) // pool_stride + 1

        self.dropout = nn.Dropout()

        self.conv_encoder = nn.Sequential(
            nn.Conv2d(temp_embedding_dim, temp_embedding_dim, (2, 1)),
            nn.BatchNorm2d(temp_embedding_dim),
            nn.ELU()
        )
        self.classify = nn.Linear(embed_dim*temp_embedding_dim, num_classes)

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x1 = self.temp_conv1(x)
        x2 = self.temp_conv2(x)
        x3 = self.temp_conv3(x)
        x4 = self.temp_conv4(x)
        x = torch.cat((x1,x2,x3,x4), dim=1)
        x = self.bn1(x)

        x = self.spatial_conv(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = x.squeeze()

        x1 = self.avg_pool(x)
        x2 = self.var_pool(x)

        x1 = self.dropout(x1)
        x2 = self.dropout(x2)

        x1 = rearrange(x1, 'b d n -> b n d')
        x2 = rearrange(x2, 'b d n -> b n d')
        
        x1 = x1.unsqueeze(dim=2)
        x2 = x2.unsqueeze(dim=2)

        x = torch.cat((x1, x2), dim=2)
        x = self.conv_encoder(x)

        x = x.reshape(x.size(0), -1)

        out = self.classify(x)

        return out
