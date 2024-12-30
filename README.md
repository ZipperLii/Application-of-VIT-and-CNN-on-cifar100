# ViT-B16 on CIFAR-100

Using ViT-B16(Transformer structure) for classification on CIFAR-100 and comparing the performance of the model straightly trained on dataset and fine-tuned model with pre-trained weights.

## Requirement

Experiment Eviroument

- python3.10.9
- pytorch2.5.1+cu121

download repo:

```bash
$ git clone https://github.com/ZipperLii/ViT-B16-on-CIFAR100
```

## Usage

### Params

##### 1. ViT-B16 straightly trained on CIFAR-10

- Configuration

| Hyperparams               | Value |
|:-------------------------:|:-----:|
| patch_size                | 16×16 |
| hidden_size               | 768   |
| mlp_dim                   | 3072  |
| head_num                  | 12    |
| layer_num<br>(EncoderBlk) | 12    |
| attention_dropout         | 0.0   |
| fc_dropout                | 0.5   |

- batch_size: 64 (VRAM limited, you can try bigger one if possible)

- epoch: 50 (overfitting after about 45 epoch)

##### 2. ViT-B16 with pre-trained weights (fine-tuning on CIFAR-100)

- Configuration

| Hyperparams               | Value |
|:-------------------------:|:-----:|
| patch_size                | 16×16 |
| hidden_size               | 768   |
| mlp_dim                   | 3072  |
| head_num                  | 12    |
| layer_num<br>(EncoderBlk) | 12    |
| attention_dropout         | 0.0   |
| fc_dropout                | 0.1   |

- pre-trained weights: pre-trained on Image-Net 21k (released by Google)

- 

### Download model weights

- 

### Results

| dataset   | model                   | top1<br>acc | epoch<br>(lr=0.01) | epoch<br>(lr=0.02) | batch_size | weight<br>decay |
|:---------:|:-----------------------:|:-----------:|:------------------:|:------------------:|:----------:| --------------- |
| cifar-100 | ViT-B16                 |             | 50                 | -                  | 64         | 0               |
| cifar-100 | ViT-B16<br>(pretrained) |             |                    |                    |            |                 |
| cifar-100 | ResNet-50               |             |                    |                    |            |                 |

### Reference

1. [KingDomDom/ViT-CIFAR100-python](https://github.com/KingDomDom/ViT-CIFAR100-python)

2. 

### Author

ZipperLi (Zhepei Li)

Thanks for your time!