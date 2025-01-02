# Fine-tuning of ViT-B16 on CIFAR-100

Application of VisionTransformer(ViT-B/16) for image classification on CIFAR-10 and CIFAR-100 dataset. After using different fine-tuning methods on pre-trained model and trying to increase the performance, there are some explorations and findings about the performance of Transformer's architecture for image classification.

## Requirement

Experiment Eviroument

- python3.10.9
- pytorch2.5.1+cu121

Download repository:

```bash
$ git clone https://github.com/ZipperLii/ViT-B16-on-CIFAR100
```

## Implemented components

- **Vision Transformer architecture** implemented *from scratch* which includes Multihead Self-Attention, MLP, EncoderBlock and MLP Head(classification) module.

- **Fine-tuning model** by *feature extraction* and *full Fine-tuning* on CIFAR-10 and CIFAR-100 dataset.

- **Process Visualization** including *Attention Map*, *Positional Embedding* and *Patch Embedding filters*.

## Fine-tuning ViT-B/16 on CIFAR-100

### ViT architecture

Based on Transformer, images are divided into patches by filters and then flattend before embedding. After positional embedding, the model can recognize positional and various kinds of information between patches by multi-head self-attention mechanism.

<img title="" src="img/VIT.png" alt="VIT" width="835" data-align="inline">

### Dataset

[CIFAR-10 and CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)

These two datasets contain 50000 images for training and 10000 for testing respectively.

I have tried to train ViT-B16 on CIFAR-10 from the ground up, but the performance is far away from CNNs

- 50 epochs with SGD(Momentum=0.9) → 0.6327 acc on test set
- 60 epoch with Adam(L2-Regularization) → overfitting occurring after 20 epoch

It seems  hard to train on CIFAR dataset from the ground up. The architecture of ViT is more complex than other effective CNNs. So, it is easier for ViT to be overfitted on datasets of small scale like CIFAR-10 and CIFAR-100. It verified the conclusion of [2010.119290](https://arxiv.org/pdf/2010.11929).

### Pre-trained Model

The model is pre-trained on ImageNet-21k dataset which contains bigger amounts of images with more classes. So, there are more features learnt in pre-training process. 

In this task, we deploy ViT-B/16 locally and load [pre-trained weights](https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz) to address the training problem.

```python
!wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz 
```

### Params and Configuration

- architecture configuration(Params$\approx$86M)

| Hyperparams | patch_size | hidden_size | MLP<br>dim | Head<br>num | layer_num<br>(EncoderBlk) | Attention<br>Dropout | MLP<br>Dropout |
|:-----------:|:----------:|:-----------:|:----------:|:-----------:|:-------------------------:|:--------------------:|:--------------:|
| **Value**   | 16×16      | 768         | 3072       | 12          | 12                        | 0.0                  | 0.1            |

- Change the out dimension of MLP_head to be 100 for classification

#### 1. Feature extraction (fine-tuning mlp_head)

- **Freezing all parameters except output layer** (MLP_head)

- 50 Epochs with 256 batch_size

- Optimizer: SGD(Momentun=0.9, weight_decay=0.0001)

- lr: 1e-2(cosine decay)

##### Result＆Analysis

- Test Accuracy = 0.7726

- According to the following plot, if we fine-tune by just unfreezing the mlp_head, the model converges with 0.7726 test accuracy.

- It shows that the pre-trained model already have the capacity of capturing quantitative features even though we just rectified the output head.

<img title="" src="img/Fine-tuned-head-epoch50.jpg" alt="fig.2][Fine-tuning-of-ViT-B16-on-CIFAR-10-and-CIFAR-100/img/Fine-tuned-head-epoch50.jpg at main · ZipperLii/Fine-tuning-of-ViT-B16-on-CIFAR-10-and-CIFAR-100 · GitHub" width="475" data-align="inline">

#### 2. Layer fine-tuning (unfreeze mlp layers and mlp_head)

- **Freezing all parameters except mlp layers in all encoder bolck and output layer** (MLP_head)

- 20 Epoch with 64 batch_size

- Optimizer: SGD(Momentun=0.9, weight_decay=0.0001)

- lr: 1e-2(cosine decay)

##### Result＆Analysis

- Test Accuracy = 0.9248
- By fine-tuning mlp head and all mlp layers of encoder, the accuracy increased 0.1522.
- The transfer performance(top1 acc) is slightly better than the outcome of [2010.119290](https://arxiv.org/pdf/2010.11929) by SGD with 0.9 momentum and cosine decay learning rate.

<img title="" src="img/ViT-b16-CIFAR100-Epoch20-Finetuned-mlp.jpg" alt="" width="470">

#### 3. Full fine-tuning

- 20 Epoch with 64 batch_size

- Optimizer: SGD(Momentun=0.9, weight_decay=0.0001)

- lr: 1e-2(cosine decay)

### Visiualization

#### 1. Attention Map

The original images are in the upper row with groundtruth. The attention maps are in the lower row which predict how the regions weight.

The high value of greyscale(lighter) means that the model paied more attention to this region.

![](.\img\Attention_map.png)

### Download model weights

- [HEAD-FT-acc0.7726-Epoch50](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/u3637438_connect_hku_hk/EoDb8JWfCxFOh-e9ffIE_9MB2aZUTA4t8Ml3JMNwSJe73g?e=4RKzMg)

- [MLP-FT-acc0.9248-Epoch20](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/u3637438_connect_hku_hk/EoDb8JWfCxFOh-e9ffIE_9MB2aZUTA4t8Ml3JMNwSJe73g?e=4RKzMg)

- ### Results

| dataset   | model                                     | top1<br>acc | epoch<br> | batch_size | weight<br>decay |
|:---------:|:-----------------------------------------:|:-----------:|:---------:|:----------:| --------------- |
| cifar-100 | ViT-B16<br>(fine-tuned   <br> mlp head)   | 0.7266      | 50        | 256        | 0.0001          |
| cifar-100 | ViT-B16<br>(fine-tuned   <br> mlp layers) | 0.9248      | 20        | 64         | 0.0001          |
| cifar-100 | ViT-B16<br>(FFT)                          |             |           |            |                 |

### Reference

1. [KingDomDom/ViT-CIFAR100-python](https://github.com/KingDomDom/ViT-CIFAR100-python)

2. [ra1ph2/Vision-Transformer](https://github.com/ra1ph2/Vision-Transformer?tab=readme-ov-file)

3. 

### Author

ZipperLi (Zhepei Li)

Thanks for your time!