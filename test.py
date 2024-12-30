import torch
import torchvision
import numpy as np
from models.ViT_B16 import VisionTransformer
from torch.utils import data
from torchvision import transforms
from config.ViT_config import ViT_Config
from utils.train_utils import cal_accuracy_gpu

def main():
    vit_config = ViT_Config()
    test_model = VisionTransformer(vit_config, 
                              load_head=False)
    MODEL_PATH = './checkpoints/ViT-B_16.npz'
    test_model.load_weights(np.load(MODEL_PATH))
    DATA_PATH = './data'
    batch_size = 64
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(0.5, 0.5)
    ])
    test_data = torchvision.datasets.CIFAR100(
        root=DATA_PATH,
        train=False,
        transform=trans,
        download=True)
    test_iter = data.DataLoader(test_data, batch_size, shuffle=True)
    
    test_acc = cal_accuracy_gpu(test_model, test_iter)
    print(test_acc)

if __name__ == '__main__':
    main()