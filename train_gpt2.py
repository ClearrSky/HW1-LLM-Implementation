from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

#-------------------------

class CasualSelfAttention(nn.Module):
    
    def __init__(self,config):
        super().__init__()
        # 相当于一个线性变换对应的矩阵同时记录了qkv，所以需要 3 * config.n_embd
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output matrix
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
    
    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x).chunk(3, dim=-1)
        q, k, v = qkv.split(self.n_embd,dim=2)
        k = k.view(B,T,self.n_head,C // self.n_head).transpose(1,2)
        q = q.view(B,T,self.n_head,C // self.n_head).transpose(1,2)
        v = v.view(B,T,self.n_head,C // self.n_head).transpose(1,2)

        att = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1,2).contiguous().view(B,T,self.n_embd)
        y = self.c_proj(y)
        return y
    
class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
    
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    def forward(self, x):                   # GPT2的修改，让残差连接在正则化之外（更加clean）
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
@dataclass
class GPT2Config:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    
class GPT(nn.Module): # GPT language model
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config.n_embd, config.n_head) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)