import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from torch.nn import functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from models.ViT_B16 import np2th
from config.ViT_config import ViT_Config
from models.ViT_B16 import VisionTransformer

def transform_pca(filter):
    sc = StandardScaler()
    pca = PCA()

    # filter = np2th(weight["embedding/kernel"])
    filter = torch.reshape(filter, (768, -1))
    filter = torch.transpose(filter, 0, 1).detach()
    # filter = sc.fit_transform(filter)
    filter = pca.fit_transform(filter)
    filter = torch.from_numpy(filter)
    filter = torch.transpose(filter, 0, 1)
    filter = torch.reshape(filter, (768, 3, 16, 16))
    # print(filter.shape)

    return filter

def visTensor(tensor, ch=0, ncol=7, padding=1, allkernels=False): 
    # print(tensor.shape)
    n,c,h,w = tensor.shape

    if allkernels: tensor = tensor.view(n*c, -1, w, h)
    elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // ncol + 1, 64))    
    grid = torchvision.utils.make_grid(tensor, nrow=ncol, normalize=True, padding=padding)
    return grid, rows

def patch_embed():
    vit_config = ViT_Config()
    model1 = VisionTransformer(vit_config, 
                                load_head=False)
    MODEL_PATH1 = './checkpoints/CIFAR-100/ViT-b16-CIFAR100-Epoch100-Finetuned-mlp.pth'
    model1.load_state_dict(torch.load(MODEL_PATH1))
    filter1 = model1.embedding_layer.patch_embed.weight
    filter1 = transform_pca(filter1)

    model2 = VisionTransformer(vit_config, 
                                load_head=False)
    MODEL_PATH2 = './checkpoints/CIFAR-100/ViT-b16-CIFAR100-Epoch50-FFT.pth'
    model2.load_state_dict(torch.load(MODEL_PATH2))
    filter2 = model2.embedding_layer.patch_embed.weight
    filter2 = transform_pca(filter2)

    topk = 35
    ncol = 7
    grid1, row1 = visTensor(filter1[0:topk], ncol=ncol)
    grid2, row2 = visTensor(filter2[0:topk], ncol=ncol)

    fig, axs = plt.subplots(1, 2)

    axs[0].imshow(grid1.numpy().transpose((1, 2, 0)), cmap='gray')
    axs[0].axis('off')
    axs[1].imshow(grid2.numpy().transpose((1, 2, 0)), cmap='gray')
    axs[1].axis('off')

    axs[0].set_title(f'first {topk} principal filters\nof PEFT model', fontsize=18)
    axs[1].set_title(f'first {topk} principal filters\nof FFT model', fontsize=18)
    plt.show()

def pos_encoding():
    vit_config = ViT_Config()
    model = VisionTransformer(vit_config, 
                                load_head=False)
    MODEL_PATH1 = './checkpoints/CIFAR-100/ViT-b16-CIFAR100-Epoch100-Finetuned-mlp.pth'
    model.load_state_dict(torch.load(MODEL_PATH1))
    
    posembd_filter = model.embedding_layer.pos_encoding
    a = posembd_filter[0][:]
    a = F.normalize(a, p=2, dim=-1)
    b = torch.transpose(a, 0, 1)
    a = torch.matmul(a, b)

    grid = torchvision.utils.make_grid(a, nrow=1, normalize=True, padding=1)

    grid = grid.numpy().transpose((1, 2, 0))
    plt.figure(figsize=(15,15))
    plt.pcolormesh(a.detach().numpy().transpose(0,1))
    plt.imshow(grid)
    plt.title('Positional Encoding', fontsize=25)
    plt.show()
    
if __name__ == '__main__':
    pos_encoding()