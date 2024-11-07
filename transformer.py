import torch
import torch.nn as nn
from torch.nn import functional as F
import math


#########  HELPER FUNCTIONS ###########

# Functions to get slopes for ALiBi
def slopes_power_2(n_head):
    """
    Calculate slopes for attention heads in powers of 2.

    Args:
        n_head (int): Number of attention heads.

    Returns:
        list of float: List of slopes for each attention head.
    """
    slope = (2**(-2**-(math.log2(n_head)-3)))
    slope1 = slope
    return [slope*slope1**n for n in range(n_head)]

def encoder_slopes(n_head):
    """
    Calculate encoder slopes for ALiBi if number of heads is not a power of 2.

    Args:
        n_head (int): Number of attention heads.

    Returns:
        list of float: List of slopes for encoder.
    """
    if math.log2(n_head).is_integer():
        return slopes_power_2(n_head)
    else:
        n = 2**math.floor(math.log2(n_head))
        return slopes_power_2(n) + encoder_slopes(2*n)[0::2][:n_head-n]
    
def decoder_slopes(slope):
    """
    Calculate decoder slopes matrix for ALiBi in decoder.

    Args:
        slope (float): Base slope value for decoder ALiBi.

    Returns:
        torch.Tensor: Slope matrix for decoder ALiBi.
    """
    res = []
    for i in range(32):
        temp = []
        for j in range(0, i):
            temp.append(-j)
        temp = temp[::-1] + [0]*(32-i)
        res.append(temp)
    return slope*torch.Tensor(res)

def rotary_positional_encoding(x):
    """
    Applies Rotary Positional Encoding to the input tensor.
    Args:
        x (Tensor): Input tensor
    Returns:
        Tensor: Tensor with rotary positional encoding applied
    """
    cos_seq = torch.cos(x)
    sin_seq = torch.sin(x)
    return torch.stack([cos_seq, -sin_seq, sin_seq, cos_seq], dim=-1)

def apply_rope(q, k, rotary_dim):
    """
    Applies rotary positional encoding to query and key.
    Args:
        q (Tensor): Query tensor
        k (Tensor): Key tensor
        rotary_dim (int): Dimension to apply the encoding
    Returns:
        Tuple[Tensor, Tensor]: Updated query and key with ROPE applied
    """
    cos_pos = torch.cos(q[:, :, :rotary_dim])
    sin_pos = torch.sin(q[:, :, :rotary_dim])

    # Apply ROPE
    q_rot = (q[:, :, :rotary_dim] * cos_pos) + (q[:, :, rotary_dim:] * sin_pos)
    k_rot = (k[:, :, :rotary_dim] * cos_pos) + (k[:, :, rotary_dim:] * sin_pos)

    # Concatenate rotary and non-rotary dimensions
    q = torch.cat([q_rot, q[:, :, rotary_dim:]], dim=-1)
    k = torch.cat([k_rot, k[:, :, rotary_dim:]], dim=-1)

    return q, k



# Scaled Dot-Product Self-Attention Head
class AttentionHead(nn.Module):
    def __init__(self, head_size, n_embd, block_size, decoder=False, dropout=0.2, alibi=False, n_head=2, device="cpu", slope=0.5):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        self.decoder = decoder
        self.alibi = alibi
        self.n_head = n_head
        self.rotary_dim = head_size // 2  
        
        # ALiBi parameters (only used if alibi is True)
        if self.alibi and not self.decoder:
            self.rel_pos = torch.arange(block_size)[None, :].to(device) - torch.arange(block_size)[:, None].to(device)
            self.rel_pos = torch.abs(self.rel_pos).unsqueeze(0).expand(n_head, -1,-1)
            self.slopes = torch.Tensor(encoder_slopes(n_head)).to(device)*(-1)
            self.bias = self.slopes.unsqueeze(1).unsqueeze(1)*self.rel_pos
            self.bias = self.bias.view(1, n_head, block_size, block_size)
            
        if self.alibi and self.decoder:
            self.bias = decoder_slopes(slope).to(device)
            self.bias = self.bias.view(1, 1, block_size, block_size)

    def forward(self, x):
        batch, time, channels = x.shape

        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        if self.alibi:
            # Apply ROPE to query and key
            query, key = apply_rope(query, key, self.rotary_dim)
            mat_mul = query @ key.transpose(-2, -1)
            # Add ALiBi bias
            attn_weights = mat_mul.view(batch//self.n_head, self.n_head, 32, 32)
            attn_weights += self.bias[:,:,:32,:32].to(attn_weights)
            mat_mul = attn_weights.view(batch, 32, 32)
        else:
            # Standard positional encoding (if `alibi` is False)
            pos_enc = torch.arange(time, dtype=torch.float32, device=x.device).unsqueeze(0).unsqueeze(-1)
            query += pos_enc
            key += pos_enc
            mat_mul = query @ key.transpose(-2, -1)


        mat_mul /= channels**(0.5)
        if self.decoder:
            mat_mul = mat_mul.masked_fill(self.tril[:time, :time] == 0, float("-inf"))
        mat_mul = F.softmax(mat_mul, dim=-1)
        attention_maps = mat_mul
        mat_mul = self.dropout(mat_mul)

        res = mat_mul @ value

        return res, attention_maps


# Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, head_size, n_embd, block_size, decoder=False, dropout=0.2, alibi=False):
        super().__init__()
        # Initialize multiple attention heads with the ROPE and ALiBi configuration
        self.heads = nn.ModuleList([
            AttentionHead(
                head_size, n_embd, block_size, decoder=decoder, dropout=dropout, 
                alibi=alibi, slope=(0.5)**(i+1)
            ) for i in range(n_head)
        ])
        # Projection layer to merge the outputs from each head back to the embedding dimension
        self.projection_layer = nn.Linear(n_head * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        res = []
        attention_maps = []
        # Loop through each head and accumulate results
        for head in self.heads:
            head_res, attn_map = head(x)
            res.append(head_res)
            attention_maps.append(attn_map)

        # Concatenate along the last dimension for multi-head attention
        res = torch.cat(res, dim=-1)
        res = self.projection_layer(res)
        res = self.dropout(res)

        return res, attention_maps


# Feedforward Network
class FeedForward(nn.Module):
    def __init__(self, n_input, n_hidden, dropout=0.2):
        super().__init__()
        self.linear1 = nn.Linear(n_input, n_hidden)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)  # First dropout after linear1
        self.linear2 = nn.Linear(n_hidden, n_input)
        self.dropout2 = nn.Dropout(dropout) # Second dropout after linear2
        self.layer_norm = nn.LayerNorm(n_input)  # Layer normalization

    def forward(self, x):
        residual = x  # Save input for residual connection
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        x = self.layer_norm(x + residual)  # Apply residual connection and layer norm
        
        return x
    

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, n_head, n_embd, block_size, n_input, n_hidden, decoder=False, dropout=0.2, alibi=False):
        super().__init__()
        head_size = n_embd // n_head
        # Self-attention with MultiHeadAttention module
        self.self_attention = MultiHeadAttention(n_head, head_size, n_embd, block_size, decoder=decoder, dropout=dropout, alibi=alibi)
        # Layer norms applied after attention and feedforward layers
        self.layernorm1 = nn.LayerNorm(n_embd)
        self.feedforward = FeedForward(n_input, n_hidden, dropout=dropout)
        self.layernorm2 = nn.LayerNorm(n_input)

    def forward(self, x):
        x, attention_maps = x
        # Self-attention with residual connection and layer norm
        y, attention_maps = self.self_attention(x)
        x = self.layernorm1(x + y)
        # Feedforward with residual connection and layer norm
        res = self.layernorm2(x + self.feedforward(x))

        return res, attention_maps


# Encoder only Transformer Model
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, n_head, n_input, n_hidden, n_output, n_layer, device="cpu", dropout=0.2, alibi=False):
        super().__init__()
        self.device = device
        self.alibi = alibi

        # Embedding and positional encoding
        self.input_embedding = nn.Embedding(vocab_size, n_embd)
        self.positional_encoding = nn.Embedding(block_size, n_embd)
        
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(n_head, n_embd, block_size, n_input, n_hidden, dropout=dropout, alibi=alibi) for _ in range(n_layer)]
        )

        # Flatten and final classification layer
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(n_embd * block_size, n_output)
    
    def forward(self, x, targets=None):
        if len(x) == 2:
            x, attention_maps = x
        else:
            attention_maps = None
        #print(x)
        batch, time = x.shape

        # Input embedding
        input_embd = self.input_embedding(x)

        # Apply positional encoding if ALiBi is enabled
        if self.alibi:
            positional_embd = self.positional_encoding(torch.arange(time, device=self.device))
            y = input_embd + positional_embd
        else:
            y = input_embd
        
        # Apply transformer blocks
        y, attention_maps = self.transformer_blocks((y, None))

        # Flatten and apply final linear layer
        y = self.flatten(y)
        y = self.linear(y)

        # Compute loss if targets are provided
        loss = F.cross_entropy(y, targets) if targets is not None else None

        return y, loss, attention_maps
    

# Decoder only Transformer Model
class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, n_head, n_input, n_hidden, n_layer, device="cpu", decoder=True, alibi=False):
        super().__init__()
        self.device = device
        
        # Embedding and positional encoding layers
        self.input_embedding = nn.Embedding(vocab_size, n_embd)
        self.positional_encoding = nn.Embedding(block_size, n_embd)

        # Transformer blocks
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(n_head, n_embd, block_size, n_input, n_hidden, decoder=decoder, dropout=0.2, alibi=alibi) for _ in range(n_layer)]
        )
        
        # Final linear layer for logits over vocabulary
        self.linear = nn.Linear(n_input, vocab_size)

    def forward(self, x, targets=None):
        #x, attention_maps = x
        batch, time = x.shape

        # Input embedding and positional encoding
        input_embd = self.input_embedding(x)
        positional_embd = self.positional_encoding(torch.arange(time, device=self.device))
        y = input_embd + positional_embd

        # Pass through transformer blocks
        y, attention_maps = self.transformer_blocks((y, None))

        # Apply final linear layer to get logits for each token
        y = self.linear(y)

        loss = None
        if targets is not None:
            batch, time, channels = y.shape
            y = y.view(batch*time, channels)
            targets = targets.view(batch*time)
            loss = F.cross_entropy(y, targets)

        return y, loss, attention_maps
