import ml_collections

def ViT_Config():
    config = ml_collections.ConfigDict()
    config.patch = ml_collections.ConfigDict({'size': (16, 16)})
    config.input_size = (3, 224, 224)
    config.hidden_size = 768
    config.num_classes = 100
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.head_num = 12
    config.transformer.num_key_value_head = 12
    config.transformer.layer_num = 12
    config.transformer.qkv_bias = True
    config.transformer.attention_dropout = 0.0
    config.transformer.dropout = 0.1
    return config

def ViT_Config_Pretrained():
    config = ml_collections.ConfigDict()
    config.patch = ml_collections.ConfigDict({'size': (16, 16)})
    config.input_size = (3, 224, 224)
    config.hidden_size = 768
    config.num_classes = 100
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.head_num = 12
    config.transformer.num_key_value_head = 12
    config.transformer.layer_num = 12
    config.transformer.qkv_bias = True
    config.transformer.attention_dropout = 0.0
    config.transformer.dropout = 0.01
    return config