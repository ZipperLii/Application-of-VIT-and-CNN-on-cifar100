import copy, logging
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from os.path import join
from scipy import ndimage

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"

def np2th(weights):
    if weights.ndim == 4:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

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
        self.hidden_size = config.hidden_size
        self.attn_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.mlp_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.selfattention = SelfAttention(config, self.vis)
        self.mlp = MLP(config)

    def forward(self, embed_state):
        skip_state = embed_state
        hidden_state = self.attn_norm(embed_state)
        hidden_state, weights = self.selfattention(hidden_state)
        hidden_state += skip_state
        
        skip_state = hidden_state
        hidden_state = self.mlp_norm(hidden_state)
        hidden_state = self.mlp(hidden_state)
        output = hidden_state + skip_state
        return output, weights
    
    def load_weights(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(
                weights[join(ROOT,ATTENTION_Q, "kernel").replace("\\","/")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(
                weights[join(ROOT, ATTENTION_K, "kernel").replace("\\","/")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(
                weights[join(ROOT, ATTENTION_V, "kernel").replace("\\","/")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(
                weights[join(ROOT, ATTENTION_OUT, "kernel").replace("\\","/")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(
                weights[join(ROOT, ATTENTION_Q, "bias").replace("\\","/")]).view(-1)
            key_bias = np2th(
                weights[join(ROOT, ATTENTION_K, "bias").replace("\\","/")]).view(-1)
            value_bias = np2th(
                weights[join(ROOT, ATTENTION_V, "bias").replace("\\","/")]).view(-1)
            out_bias = np2th(
                weights[join(ROOT, ATTENTION_OUT, "bias").replace("\\","/")]).view(-1)

            self.selfattention.Wq.weight.copy_(query_weight)
            self.selfattention.Wk.weight.copy_(key_weight)
            self.selfattention.Wv.weight.copy_(value_weight)
            self.selfattention.out.weight.copy_(out_weight)
            
            self.selfattention.Wq.bias.copy_(query_bias)
            self.selfattention.Wk.bias.copy_(key_bias)
            self.selfattention.Wv.bias.copy_(value_bias)
            self.selfattention.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(
                weights[join(ROOT, FC_0, "kernel").replace("\\","/")]).t()
            mlp_weight_1 = np2th(
                weights[join(ROOT, FC_1, "kernel").replace("\\","/")]).t()
            mlp_bias_0 = np2th(
                weights[join(ROOT, FC_0, "bias").replace("\\","/")]).t()
            mlp_bias_1 = np2th(
                weights[join(ROOT, FC_1, "bias").replace("\\","/")]).t()

            self.mlp.fc1.weight.copy_(mlp_weight_0)
            self.mlp.fc2.weight.copy_(mlp_weight_1)
            self.mlp.fc1.bias.copy_(mlp_bias_0)
            self.mlp.fc2.bias.copy_(mlp_bias_1)

            self.attn_norm.weight.copy_(np2th(
                weights[join(ROOT, ATTENTION_NORM, "scale").replace("\\","/")]))
            self.attn_norm.bias.copy_(np2th(
                weights[join(ROOT, ATTENTION_NORM, "bias").replace("\\","/")]))
            self.mlp_norm.weight.copy_(np2th(
                weights[join(ROOT, MLP_NORM, "scale").replace("\\","/")]))
            self.mlp_norm.bias.copy_(np2th(
                weights[join(ROOT, MLP_NORM, "bias").replace("\\","/")]))


class TransformerEndocer(nn.Module):
    def __init__(self, config, vis):
        super(TransformerEndocer, self).__init__()
        self.vis = vis
        self.encoder_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.layer_num = config.transformer.layer_num
        self.encoder_layer = nn.ModuleList()
        for _ in range(self.layer_num):
            layer = EncoderBlock(config, self.vis)
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
                                     stride=self.patch_size,
                                     bias=True)
        
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
    def __init__(self, config, load_head=True, vis=False):
        super(VisionTransformer, self).__init__()
        self.load_head = load_head
        self.vis = vis
        self.img_size2d = config.input_size[1:]
        self.img_channel = config.input_size[0]
        self.embedding_layer = Embedding(config, self.img_size2d, self.img_channel)
        self.feature_layer = TransformerEndocer(config, self.vis)
        self.mlp_head = nn.Linear(config.hidden_size, config.num_classes)
    
    def load_weights(self, weights):
        with torch.no_grad():
            if not self.load_head:
                nn.init.xavier_uniform_(self.mlp_head.weight)
                nn.init.zeros_(self.mlp_head.bias)
            else:
                self.mlp_head.weight.copy_(np2th(weights["head/kernel"]).t())
                self.mlp_head.bias.copy_(np2th(weights["head/bias"]).t())

            # load patch embedding filter weights
            self.embedding_layer.patch_embed.weight.copy_(np2th(weights["embedding/kernel"]))
            self.embedding_layer.patch_embed.bias.copy_(np2th(weights["embedding/bias"]))
            # load <cls> weights
            self.embedding_layer.cls_token.copy_(np2th(weights["cls"]))
            # load encoder layernorm weights
            self.feature_layer.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.feature_layer.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            # load positional embedding weights
            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_origin = self.embedding_layer.pos_encoding
            if posemb.size() == posemb_origin.size():
                self.embedding_layer.pos_encoding.copy_(posemb)
            else:
                logger = logging.getLogger(__name__)
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_origin.size()))
                ntok_new = posemb_origin.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            # nn.Module.named_children() → Iterator[Tuple[str, nn.Module]]
            for banme, block in self.feature_layer.named_children():
                for uname, unit in block.named_children():
                    # get each encoder block from block(12 layers)
                    unit.load_weights(weights, n_block=uname)

    def forward(self, input_ids):
        embed_feature = self.embedding_layer(input_ids)
        features, attn_weights = self.feature_layer(embed_feature)
        
        # fc with first feature of Transformer output
        logits = self.mlp_head(features[:, 0])
        
        return logits, attn_weights
    
    