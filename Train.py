import torch
import torchvision
import torch.nn as nn
from models.ResNet import ResVGG
from models.ViT_B16 import VisionTransformer
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm
from config.ViT_config import ViT_Config
import matplotlib.pyplot as plt

def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def get_optimizer(net, lr):
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    return optimizer

def cal_correct_num(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def cal_accuracy_gpu(net, data_iter, device=None):
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    metric = [0.0] * 3
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            correct_list = [cal_correct_num(net(X), y), y.numel()]
            metric = [a + float(b) for a, b in zip(metric, correct_list)]
    return metric[0] / metric[1]

def save_model(net, path):
    torch.save(net.state_dict(), path)

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)

def plot_curves(epoch_train_loss, epoch_train_acc, epoch_test_acc, path='./img/Fig.jpg'):

    num_epochs = len(epoch_test_acc)
    batch_size = len(epoch_train_acc) / num_epochs

    batch_indices = [i / batch_size for i in range(len(epoch_train_loss))]
    epoch_indices = range(1, num_epochs + 1)

    plt.rcParams['font.family'] = 'Arial'
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(batch_indices, epoch_train_loss, label="Train Loss", color="#8A2BE2", marker="")
    ax1.set_xlabel("Epoch", fontsize=14)
    ax1.set_ylabel("Loss", color="black", fontsize=13)
    ax1.tick_params(axis="y", labelcolor="black", labelsize=12)
    ax1.set_xticks(range(num_epochs + 1))

    ax2 = ax1.twinx()
    ax2.plot(batch_indices, epoch_train_acc, label="Train Accuracy", color="#6495ED", marker="")
    ax2.plot(epoch_indices, epoch_test_acc, label="Test Accuracy", color="#FF4500", marker="")
    ax2.set_ylabel("Accuracy", color="black", fontsize=14)
    ax2.tick_params(axis="y", labelcolor="black", labelsize=12)

    plt.title("Training and Testing Loss and Accuracy Curves", fontsize=15)
    fig.tight_layout()
    fig.legend(ncol=3, bbox_to_anchor=(0.5, -0.06), loc='lower center',
            edgecolor='w', fontsize=15)

    plt.savefig(path, dpi=300,bbox_inches='tight')

def train_model(net, train_iter, test_iter, num_epochs, lr, device, test=False, plot=False):
    # initialize weights
    net.apply(init_weights)
    # train on gpu?
    print('Training on', device)
    net.to(device)
    # optimizer
    optimizer = get_optimizer(net, lr)
    # criterion
    loss = nn.CrossEntropyLoss()
    if plot == True:
        train_record = [[], []]
        if test == True:
            test_record = []

    # start training
    for epoch in range(num_epochs):
        metric = [0.0] * 3
        net.train()
        with tqdm(total=len(train_iter), desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
            for i, (X, y) in enumerate(train_iter):
                optimizer.zero_grad()
                X, y = X.to(device), y.to(device)
                y_hat = net(X)[0]
                l = loss(y_hat, y)
                l.backward()
                optimizer.step()
                with torch.no_grad():
                    metrics_list = [l * X.shape[0], cal_correct_num(y_hat, y), X.shape[0]]
                    metric = [a + float(b) for a, b in zip(metric, metrics_list)]
                    train_l = metric[0] / metric[2]
                    train_acc = metric[1] / metric[2]
                    if plot == True:
                        train_record[0].append(train_l)
                        train_record[1].append(train_acc)
                # update bar
                pbar.set_postfix({"Loss": train_l, "Accuracy": train_acc})
                pbar.update(1)

        if test == True:
            test_acc = cal_accuracy_gpu(net, test_iter)
            print(f"Epoch {epoch + 1}/{num_epochs} -> Train Accuracy: {train_acc:.4f}, Train Loss: {train_l:.4f}, Test Accuracy: {test_acc:.4f}")
        else:
            print(f"Epoch {epoch + 1}/{num_epochs} -> Train Accuracy: {train_acc:.4f}, Train Loss: {train_l:.4f}")
        
        if (plot == True) and (test == True):
            test_record.append(test_acc)
        
    print(f'Final loss {train_l:.3f}, Final Train acc {train_acc:.3f}',end='')
    print(f',Final Test acc {test_acc:.3f}') if test == True else None
    if plot == True:
        PATH = './img/Fig.jpg'
        plot_curves(train_record[0], train_record[1], test_record, PATH)
    

def main():
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224,scale=(0.64,1.0),ratio=(1.0,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(0.5, 0.5)
    ])
    train_dataset = torchvision.datasets.CIFAR10(
        "./DeepLearning/Application-of-VIT-and-CNN-on-cifar100/data",
        True,
        trans,
        download=True
    )
    test_dataset = torchvision.datasets.CIFAR10(
        "./DeepLearning/Application-of-VIT-and-CNN-on-cifar100/data",
        False,
        trans,
        download=True
    )

    batch_size = 64
    train_iter = data.DataLoader(train_dataset, batch_size, shuffle=True)
    test_iter = data.DataLoader(test_dataset, batch_size, shuffle=True)

    config = ViT_Config()
    model = VisionTransformer(config)

    num_epochs = 50

    train_model(net=model,
            train_iter=train_iter,
            test_iter=test_iter,
            num_epochs=num_epochs,
            lr=0.01,
            device=try_gpu())
    # save model weights
    model_weights = 'ViT-b16-CIFAR10-Epoch50'
    datasets_name = 'CIFAR-10'
    PATH = f'./DeepLearning/Application-of-VIT-and-CNN-on-cifar100/checkpoints/{datasets_name}/{model_weights}.pth'
    save_model(model, PATH)

if __name__ == '__main__':
    main()