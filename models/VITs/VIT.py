import torch
import torch.nn as nn
from torch.nn import functional as F
from ViT_config import ViT_Config

config = ViT_Config()

class Embedding(nn.Module):
    def __init__(self, config):
        super(Embedding, self).__init__()
        pass
    def forward(self):
        pass


class SelfAttention(nn.Module):
    def __init__(self, config, vis):
        super(SelfAttention, self).__init__()
        self.vis = vis
        self.num_key_value_head = config.transformer.num_key_value_head
        self.head_dim = int(config.hidden_size // config.head_num)

        self.softmax = nn.Softmax(dim=-1)
        self.attention_dropout = nn.Dropout(config.transformer.attention_dropout)
        self.dropoutout = nn.Dropout(config.transformer.dropoutout)
        self.out = nn.Linear(config.hidden_size, config.hidden_size)
        self.scale = config.head_dim ** -0.5

        self.Wk = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.Wq = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.Wv = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        
        if (self.head_dim * self.head_num) != config.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads}).")

    def forward(self, hidden_states):
        Q = self.Wq(hidden_states)
        K = self.Wk(hidden_states)
        V = self.Wv(hidden_states)

        # adjust shapes to be (batch, head_num, seq_len, head_dim)
        batch_size, seq_len, _ = hidden_states.size()
        Q = Q.view(batch_size, seq_len, config.head_num, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_key_value_head, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_key_value_head, self.head_dim).transpose(1, 2)

        # calculate attention scores and output
        weights = torch.matmul(Q, K.transpose(-1, -2))
        weights = self.softmax(weights * self.scale)
        vis_weights = weights if self.vis else None
        weights = self.attention_dropout(weights)
        output = torch.matmul(weights, V)

        # reshape to be (batch, seq_len, hidden_size)
        output = output.permute(0, 2, 1, 3).contiguous()
        output_shape = output.size()[:-2] + (self.all_head_size,)
        output = output.view(*output_shape)

        # output projection
        output = self.out(output)
        output = self.dropoutout(output)
        
        return output, vis_weights


class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        pass
    def forward(self):
        pass


class EncoderBlock(nn.Module):
    def __init__(self, config):
        super(EncoderBlock, self).__init__()
        pass
    def forward(self):
        pass


class TransformerEndocer(nn.Module):
    def __init__(self, config):
        super(TransformerEndocer, self).__init__()
        self.norm = nn.LayerNorm(config.hidden_size)

    def forward(self, embedd_in):

        embedd_in = self.selfattention(embedd_in)


class VisionTransformer(nn.Module):
    def __init__(self, config):
        super(VisionTransformer, self).__init__()
        pass
    def forward(self):
        pass