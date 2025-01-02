import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt


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