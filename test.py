import torch
import torchvision
import torch.nn as nn
from models.ResNet import ResVGG
from models.ViT_B16 import VisionTransformer
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
from tqdm import tqdm
from config.ViT_config import ViT_Config
from Train import evaluate_accuracy_gpu

def main():
    config = ViT_Config()
    test_model = VisionTransformer(config)
    MODEL_PATH = './DeepLearning/Application-of-VIT-and-CNN-on-cifar100/checkpoints/CIFAR-10/ViT-b16-CIFAR10-Epoch10.pth'
    test_model.load_state_dict(torch.load(MODEL_PATH))
    batch_size = 64
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(0.5, 0.5)
    ])
    test_data = torchvision.datasets.CIFAR10(
        root="./DeepLearning/Application-of-VIT-and-CNN-on-cifar100/data",
        train=False,
        transform=trans,
        download=True)
    test_iter = data.DataLoader(test_data, batch_size, shuffle=True)
    
    test_acc = evaluate_accuracy_gpu(test_model, test_iter)
    print(test_acc)

if __name__ == '__main__':
    main()