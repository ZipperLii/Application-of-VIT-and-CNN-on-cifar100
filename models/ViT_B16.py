import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair

class SelfAttention(nn.Module):
    def __init__(self, config, vis):
        super(SelfAttention, self).__init__()
        self.vis = vis
        self.num_key_value_head = config.transformer.num_key_value_head
        self.head_num = config.transformer.head_num
        self.head_dim = int(config.hidden_size // config.transformer.head_num)
        self.head_dim_sum = self.head_num * self.head_dim

        self.softmax = nn.Softmax(dim=-1)
        self.attention_dropout = nn.Dropout(config.transformer.attention_dropout)
        self.dropout = nn.Dropout(config.transformer.dropout)
        self.out = nn.Linear(config.hidden_size, config.hidden_size)
        self.scale = self.head_dim ** -0.5

        self.Wk = nn.Linear(config.hidden_size,
                            self.head_num * self.head_dim,
                            bias=config.transformer.qkv_bias)
        self.Wq = nn.Linear(config.hidden_size,
                            self.head_num * self.head_dim,
                            bias=config.transformer.qkv_bias)
        self.Wv = nn.Linear(config.hidden_size,
                            self.head_num * self.head_dim,
                            bias=config.transformer.qkv_bias)
        
        if (self.head_dim * self.head_num) != config.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.head_num}).")

    def forward(self, hidden_states):
        Q = self.Wq(hidden_states)
        K = self.Wk(hidden_states)
        V = self.Wv(hidden_states)

        # adjust shapes to be (batch, head_num, seq_len, head_dim)
        batch_size, seq_len, _ = hidden_states.size()
        Q = Q.view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)
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
        output_shape = output.size()[:-2] + (self.head_dim_sum,)
        output = output.view(*output_shape)

        # output projection
        output = self.out(output)
        output = self.dropout(output)
        
        return output, vis_weights


class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.input_dim = config.hidden_size
        self.hidden_dim = config.transformer.mlp_dim
        self.dropout_rate = config.transformer.dropout

        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.input_dim)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.activation = F.gelu

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, hidden_state):
        hidden_state = self.fc1(hidden_state)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.dropout(hidden_state)
        hidden_state = self.fc2(hidden_state)
        output = self.dropout(hidden_state)
        return output


class EncoderBlock(nn.Module):
    def __init__(self, config, vis):
        super(EncoderBlock, self).__init__()
        self.vis = vis
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.selfattention = SelfAttention(config, vis)
        self.mlp = MLP(config)

    def forward(self, embed_state):
        skip_state = embed_state
        hidden_state = self.layernorm(embed_state)
        hidden_state, weights = self.selfattention(hidden_state)
        hidden_state += skip_state
        
        skip_state = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = self.mlp(hidden_state)
        output = hidden_state + skip_state

        return output, weights


class TransformerEndocer(nn.Module):
    def __init__(self, config, vis):
        super(TransformerEndocer, self).__init__()
        self.vis = vis
        self.encoder_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.layer_num = config.transformer.layer_num
        self.encoder_layer = nn.ModuleList()
        for _ in range(self.layer_num):
            layer = EncoderBlock(config, vis)
            self.encoder_layer.append(copy.deepcopy(layer))

    def forward(self, hidden_state):
        attn_weights = []
        for layer in self.encoder_layer:
            hidden_state, weight = layer(hidden_state)
            if self.vis:
                attn_weights.append(weight)
        output = self.encoder_norm(hidden_state)

        return output, attn_weights


class Embedding(nn.Module):
    def __init__(self, config, img_size, img_channels=3):
        super(Embedding, self).__init__()

        # transfer an integer into (img_size, img_size)
        self.img_size = _pair(img_size)
        self.patch_size = _pair(config.patch['size'])
        self.patch_num = (img_size[0]//self.patch_size[0]) * (img_size[1]//self.patch_size[1])
        self.dropout = nn.Dropout(config.transformer.dropout)

        # patch embedding by filters
        self.patch_embed = nn.Conv2d(img_channels,
                                     config.hidden_size,
                                     kernel_size=self.patch_size,
                                     stride=self.patch_size)
        
        # setting positional encoding and <cls> as parameters for learning
        self.pos_encoding = nn.Parameter(torch.zeros(1, self.patch_num+1, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

    def forward(self, img):
        batch_num = img.shape[0]
        cls_token = self.cls_token.expand(batch_num, -1, -1)
        features = self.patch_embed(img)

        # flatten all dimesions accpet for the first(batch)
        features = features.flatten(2)
        features = features.transpose(-1, -2)
        features = torch.cat((cls_token, features), dim=1)

        # positional encoding
        features += self.pos_encoding
        features = self.dropout(features)

        return features


class VisionTransformer(nn.Module):
    def __init__(self, config, vis=False):
        super(VisionTransformer, self).__init__()
        self.img_size2d = config.input_size[1:]
        self.img_channel = config.input_size[0]
        self.embedding_layer = Embedding(config, self.img_size2d, self.img_channel)
        self.feature_layer = TransformerEndocer(config, vis)
        self.mlp_head = nn.Linear(config.hidden_size, config.num_classes)
        

    def forward(self, input_ids):
        embed_feature = self.embedding_layer(input_ids)
        features, attn_weights = self.feature_layer(embed_feature)
        
        # fc with first feature of Transformer output
        logits = self.mlp_head(features[:, 0])
        
        return logits, attn_weights
