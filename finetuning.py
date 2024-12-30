import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from models.ViT_B16 import VisionTransformer
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm
from config.ViT_config import ViT_Config
from utils.train_utils import train_model, save_model, try_gpu


def np2th(weights):
    if weights.ndim == 4:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def main():
    # load pre-trained weights
    vit_config = ViT_Config()
    # change head for classification on CIFAR-100
    model = VisionTransformer(vit_config, 
                              load_head=False)
    model.load_weights(np.load('./checkpoints/ViT-B_16.npz'))

if __name__ == '__main__':
    main()
    
