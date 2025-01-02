import torch
import torchvision
from models.ViT_B16 import VisionTransformer
from torch.utils import data
from torchvision import transforms
from config.ViT_config import ViT_Config
from utils.evaluation_utils import evaluation

def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def main():
    vit_config = ViT_Config()
    test_model = VisionTransformer(vit_config, 
                              load_head=False)
    MODEL_PATH = './checkpoints/CIFAR-100/ViT-b16-CIFAR100-Epoch100-Finetuned-mlp.pth'
    test_model.load_state_dict(torch.load(MODEL_PATH))
    # test_model.load_weights(np.load(MODEL_PATH))
    DATA_PATH = './data'
    batch_size = 64
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        # transforms.RandomResizedCrop(224,scale=(0.64,1.0),ratio=(1.0,1.0)),
        # transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])
    test_data = torchvision.datasets.CIFAR100(
        root=DATA_PATH,
        train=False,
        transform=trans,
        download=True)
    test_iter = data.DataLoader(test_data, batch_size, shuffle=True)
    
    test_acc = evaluation(test_model, test_iter, try_gpu())
    print(f"Test Accuracy: {test_acc:.2%}")

if __name__ == '__main__':
    main()