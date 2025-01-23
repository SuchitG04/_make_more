import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass

# hyperparameters
@dataclass
class hp:
    n_embd: int = 16
    n_heads: int = 4
    dropout: float = 0.2
    blk_sz: int = 6
    num_blks: int = 2


class CausalAttn(nn.Module):
    """An implementation Multi-Headed Self (and causal) Attention treating the heads as a batch dimension and processing them in parallel."""
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_embd = config.n_embd
        self.head_sz = self.n_embd // self.n_heads

        self.attn = nn.Linear(self.n_embd, 3*self.n_embd)
        self.register_buffer('tril', torch.tril(torch.ones((config.blk_sz, config.blk_sz))))
        self.proj = nn.Linear(self.n_embd, self.n_embd, bias=True)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B,T,C = x.shape
        # compute k,q,v at once and destructure it to have `n_embd` size
        k,q,v = self.attn(x).split(self.n_embd, dim=-1)
        # first view the tensors as (B, T, n_heads, head_sz), then transpose the middle dimensions to get (B, n_heads, T, head_sz).
        # think about [T, n_heads] to be a separate matrix and think about transposing it.
        # initially, you'll have T number of n_heads (have n_heads heads at each timestep) (T, n_heads)
        # after transposing, you'll have, at each head, T "blocks" or timestep elements    (n_heads, T)
        k = k.view(B, T, self.n_heads, self.head_sz).transpose(1, 2) # (B, n_heads, T, head_sz)
        q = q.view(B, T, self.n_heads, self.head_sz).transpose(1, 2) # (B, n_heads, T, head_sz)
        v = v.view(B, T, self.n_heads, self.head_sz).transpose(1, 2) # (B, n_heads, T, head_sz)
        
        # raw weights based on q, k affinity --> scaled dot product attn
        wei = q @ k.transpose(-2, -1) * self.head_sz**-0.5 # (B, n_heads, T, head_sz) @ (B, n_heads, head_sz, T) --> (B, n_heads, T, T)
        # mask past tokens and get a normalized distribution for affinities
        wei = wei.masked_fill(self.tril[:T, :T]==0, float('-inf')) # (B, n_heads, T, T)
        wei = wei.softmax(dim=-1) # (B, n_heads, T, T)
        # scale value vector with affinities
        out = wei @ v # (B, n_heads, T, T) @ (B, n_heads, T, head_sz) --> (B, n_heads, T, head_sz)
        # transpose(1, 2) --> (B, T, n_heads, head_sz)
        # contiguous --> transpose operations make the underlying memory non-contiguous. operations like view require contiguous memory representations.
        # view --> (B, T, n_heads, head_sz) -> (B, T, n_embd) (n_embd = n_heads * head_sz)
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_embd)
        out = self.proj(self.dropout(out))
        return out


class FeedForward(nn.Module):
    """An MLP to go at the end of a block."""
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4*config.n_embd),
            nn.ReLU(),
            nn.Linear(4*config.n_embd, config.n_embd),
            nn.Dropout(p=config.dropout)
        )

    def forward(self, x):
        return self.net(x)
    

class Block(nn.Module):
    """A transformer block with causal attention, an MLP, and layer norm with residual connections."""
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_heads = config.n_heads
        self.head_sz = self.n_embd // self.n_heads
        self.blk_sz = config.blk_sz
        self.causal_attn = CausalAttn(config)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(self.n_embd)
        self.ln2 = nn.LayerNorm(self.n_embd)

    def forward(self, x):
        x = x + self.causal_attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x # (B, T, n_embd)


class GPT(nn.Module):
    """
    Your GPT can talk. My GPT can add. We are not the same.
    Implementation of an unassuming GPT that can add two numbers.
    """
    def __init__(self, config):
        super().__init__()
        self.tok_emb_table = nn.Embedding(10, config.n_embd) # vocab_sz, n_embd
        self.pos_emb_table = nn.Embedding(config.blk_sz, config.n_embd) # blk_sz, n_embd

        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.num_blks)],
            nn.LayerNorm(config.n_embd)
        )
        self.lm_head = nn.Linear(config.n_embd, 10) # n_embd, vocab_sz

    
    @staticmethod
    def get_default_config():
        return hp()


    def forward(self, idx, targets=None):
        device = idx.device
        B,T = idx.shape
        tok_emb = self.tok_emb_table(idx)
        pos_emb = self.pos_emb_table(torch.arange(T, device=device))

        x = tok_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x)

        if targets is None: loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.contiguous().view(B*T)
            loss = F.cross_entropy(logits, targets, ignore_index=-1)
        return logits, loss
