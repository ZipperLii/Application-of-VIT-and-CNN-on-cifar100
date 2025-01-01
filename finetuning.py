import torchvision
import numpy as np
from models.ViT_B16 import VisionTransformer
from torch.utils import data
from torchvision import transforms
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
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])
    trans2 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224,scale=(0.64,1.0),ratio=(1.0,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])
    
    batch_size = 512
    train_iter, test_iter = data_loader(batch_size,
                                        trans1,
                                        trans2,
                                        DATA_PATH)
    
    # load pre-trained weights
    vit_config = ViT_Config()
    # change head for classification on CIFAR-100
    model = VisionTransformer(vit_config, 
                              load_head=False)
    model.load_weights(np.load('./checkpoints/ViT-B_16.npz'))
    
    # freeze parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # unfreeze head and mlp layers in encoder
    model.mlp_head.weight.requires_grad = True
    model.mlp_head.bias.requires_grad = True
    for i, layer in enumerate(model.feature_layer.encoder_layer):
        layer.mlp_norm.weight.requires_grad = True
        layer.mlp_norm.bias.requires_grad = True
        layer.mlp.fc1.weight.requires_grad = True
        layer.mlp.fc1.bias.requires_grad = True
        layer.mlp.fc2.weight.requires_grad = True
        layer.mlp.fc2.bias.requires_grad = True
    
    num_epochs = 100
    
    train_model(net=model,
            train_iter=train_iter,
            test_iter=test_iter,
            num_epochs=num_epochs,
            lr=1e-1,
            device=try_gpu(),
            test=True,
            plot=True)
    
    # save model weights
    model_weights = 'ViT-b16-CIFAR100-Epoch100-Finetuned-mlp'
    datasets_name = 'CIFAR-100'
    PATH = f'./checkpoints/{datasets_name}/{model_weights}.pth'
    save_model(model, PATH)
    
if __name__ == '__main__':
    main()
    
