import torch
import cv2
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from train import data_loader
from config.ViT_config import ViT_Config
from models.ViT_B16 import VisionTransformer
from utils.evaluation_utils import CIFAR100_Num2Class

def draw_attn_map(img, model):
    # plugging in another dimension as attn layer
    x = img.unsqueeze(0)
    
    logits, attn_mat = model(x)
    prediction = torch.argmax(logits).unsqueeze(0)
    
    # transfer to torch.Size([layer_num, head_num, seq_len, seq_len])
    attn_mat = torch.stack(attn_mat).squeeze(1)

    # average among all heads
    attn_mat = torch.mean(attn_mat, dim=1)
    residual_attn = torch.eye(attn_mat.size(1))
    # broadcasting mechanism
    aug_attn_mat = attn_mat + residual_attn
    # normalization along last dimension(normalize among heads in each layer)
    # same works as attention normalization(unsqueeze(-1) for broadcasting mechanism)
    aug_attn_mat = aug_attn_mat / aug_attn_mat.sum(dim=-1).unsqueeze(-1)
    
    joint_attentions = torch.zeros(aug_attn_mat.size())
    joint_attentions[0] = aug_attn_mat[0]
    
    # calculate joint attention
    # according to the forward process(the latter attention mat = the former matmul itself)
    for n in range(1, aug_attn_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_attn_mat[n], joint_attentions[n-1])
    
    # the last matrix layer
    v = joint_attentions[-1]
    # img_size(224) // patch_size(16) = grid_size(14)
    grid_size = int(np.sqrt(aug_attn_mat.size(-1)))
    # output mask: projection on output space
    mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
    
    target_size = (img.shape[1], img.shape[2])
    mask = cv2.resize(mask / mask.max(), target_size)[..., np.newaxis]
    to_pil = transforms.ToPILImage()
    result = (mask * to_pil(img)).astype("uint8")
    return result, prediction

def img_std(img, destandardize=False):
    npimg = img.detach().cpu().numpy()
    if destandardize:
        # necessary if we standardize the dataset
        npimg = (npimg * np.array([0.2675, 0.2565, 0.2761])[:, None, None] 
                 + np.array([0.5071, 0.4867, 0.4408])[:, None, None])
    
    npimg = np.transpose(npimg, (1, 2, 0))  # from CHW to HWC
    npimg = np.clip(npimg, 0, 1) # standardize image
    # transfor to tensor for inputting to model
    img = torch.from_numpy(npimg).permute(2, 0, 1).float()
    return img
    
def attnmap_show(image_list, ncols=5):
    fig, axes = plt.subplots(2, ncols, figsize=(15, 9))
    for j in range(ncols):
        for i in range(2):
            npimg = image_list[i][j][0]
            axes[i, j].imshow(npimg, cmap='viridis')
            axes[i, j].set_title(f"{image_list[i][j][1]}", fontsize=16)
            axes[i, j].axis('off')
            
    plt.tight_layout()
    plt.show()

def main():
    DATA_PATH = "./data"
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224))
        # transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])
    batch_size = 4
    _, test_iter = data_loader(batch_size,
                                        trans,
                                        trans,
                                        DATA_PATH
                                        )
    data_iter = iter(test_iter)
    images, label = next(data_iter)
    vit_config = ViT_Config()
    model = VisionTransformer(vit_config, 
                            load_head=False,
                            vis=True)
    MODEL_PATH = './checkpoints/CIFAR-100/ViT-b16-CIFAR100-Epoch100-Finetuned-mlp.pth'
    model.load_state_dict(torch.load(MODEL_PATH))
    image_list = [[],[]]
    for i in range(batch_size):
        org_map = torchvision.utils.make_grid(images[i])
        std_map = img_std(org_map)
        attn_map, pred = draw_attn_map(std_map, model)
        org_map = std_map.numpy().transpose((1, 2, 0))
        truth = CIFAR100_Num2Class(label[i])
        image_list[0].append([org_map, f'Ground truth: {truth}'])
        prediction = CIFAR100_Num2Class(pred)
        image_list[1].append([attn_map, f'Prediction: {prediction}'])
    
    attnmap_show(image_list, 4)
    
if __name__ == '__main__':
    main()
    