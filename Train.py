import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
from models.ViT_B16 import VisionTransformer
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm
from config.ViT_config import ViT_Config
from utils.train_utils import train_model, save_model, try_gpu

def data_loader(batch_size, train_trans, test_trans, data_path):
    train_dataset = torchvision.datasets.CIFAR100(
        data_path,
        True,
        train_trans,
        download=True
    )
    test_dataset = torchvision.datasets.CIFAR100(
        data_path,
        False,
        test_trans,
        download=True
    )
    train_iter = data.DataLoader(train_dataset, batch_size, shuffle=True)
    test_iter = data.DataLoader(test_dataset, batch_size, shuffle=True)
    return train_iter, test_iter

def main():
    DATA_PATH = "./data"
    trans1 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224,scale=(0.64,1.0),ratio=(1.0,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(0.5, 0.25)
    ])
    trans2 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224,scale=(0.64,1.0),ratio=(1.0,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(0.5, 0.25)
    ])
    batch_size = 32
    train_iter, test_iter = data_loader(batch_size,
                                        trans1,
                                        trans2,
                                        DATA_PATH)
    
    config = ViT_Config()
    model = VisionTransformer(config)

    num_epochs = 60
    
    train_model(net=model,
            train_iter=train_iter,
            test_iter=test_iter,
            num_epochs=num_epochs,
            lr=0.1,
            device=try_gpu(),
            test=True,
            plot=True)
    
    # save model weights
    model_weights = 'ViT-b16-CIFAR100-Epoch50-SGD'
    datasets_name = 'CIFAR-100'
    PATH = f'./checkpoints/{datasets_name}/{model_weights}.pth'
    save_model(model, PATH)

if __name__ == '__main__':
    main()