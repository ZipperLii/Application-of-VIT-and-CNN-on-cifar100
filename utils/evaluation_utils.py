import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

def CIFAR100_Num2Class(index_tensor):
    cifar100_classes = [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
        'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
        'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
        'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
        'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
        'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
        'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
        'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
        'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
    ]
    
    if not isinstance(index_tensor, torch.Tensor):
        raise ValueError("Input must be a tensor")
    
    index = index_tensor.item()
    
    if index < 0 or index >= len(cifar100_classes):
        raise ValueError("Index out of range for CIFAR-100 classes")
    
    return cifar100_classes[index]

def cal_correct_num(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluation(net, data_iter, device):
    if isinstance(net, nn.Module):
        net.eval()
        net.to(device)
    print(f"evaluation on: {device}")
    metric = [0.0] * 3
    with torch.no_grad():
        for X, y in tqdm(data_iter, desc="Testing Progress", leave=True):
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            correct_list = [cal_correct_num(net(X)[0], y), y.numel()]
            metric = [a + float(b) for a, b in zip(metric, correct_list)]
    return metric[0] / metric[1]