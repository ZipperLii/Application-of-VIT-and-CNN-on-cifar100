import torch
import torchvision
import torch.nn as nn
from ImageNet import AlexNet
from ImageNet import VGG
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
from tqdm import tqdm

def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def get_optimizer(net, lr):
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    return optimizer

def evaluate_accuracy_gpu(net, data_iter, device=None):
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def save_model(net, path):
    torch.save(net.state_dict(), path)

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)

def train_model(net, train_iter, test_iter, num_epochs, lr, device, plot=False):
    # initialize weights
    net.apply(init_weights)
    # train on gpu?
    print('Training on', device)
    net.to(device)
    # optimizer
    optimizer = get_optimizer(net, lr)
    # criterion
    loss = nn.CrossEntropyLoss()
    if plot==True:
        # For plotting data in animation
        animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                                legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)

    # start training
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)
        net.train()
        with tqdm(total=len(train_iter), desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
            for i, (X, y) in enumerate(train_iter):
                timer.start()
                optimizer.zero_grad()
                X, y = X.to(device), y.to(device)
                y_hat = net(X)
                l = loss(y_hat, y)
                l.backward()
                optimizer.step()
                with torch.no_grad():
                    metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
                    timer.stop()
                    train_l = metric[0] / metric[2]
                    train_acc = metric[1] / metric[2]
                    if plot == True:
                        if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                            animator.add(epoch + (i + 1) / num_batches,
                                        (train_l, train_acc, None))
                # update bar
                pbar.set_postfix({"Loss": train_l, "Accuracy": train_acc})
                pbar.update(1)

        test_acc = evaluate_accuracy_gpu(net, test_iter)
        if plot == True:
            animator.add(epoch + 1, (None, None, test_acc))
        print(f"Epoch {epoch + 1}/{num_epochs} -> Train Accuracy: {train_acc:.4f}, Train Loss: {train_l:.4f}, Test Accuracy: {test_acc:.4f}")
        
    print(f'Final loss {train_l:.3f}, Final Train acc {train_acc:.3f}, '
          f'Final Test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')

def main(train = None):
    if train == 'Train':
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224))
        ])
        train_dataset = torchvision.datasets.CIFAR10(
            "./DeepLearning/CV/data",
            True,
            trans,
            download=True
        )
        test_dataset = torchvision.datasets.CIFAR10(
            "./DeepLearning/CV/data",
            False,
            trans,
            download=True
        )

        batch_size = 64
        train_iter = data.DataLoader(train_dataset, batch_size, shuffle=True)
        test_iter = data.DataLoader(test_dataset, batch_size, shuffle=True)

        # VGG16 arichitecture: 2+2+3+3+3 = 13(conv) , 13 + 3(fc) = 16
        architecture = ((2,64), (2,128), (3,256), (3,512), (3,512))
        model = VGG(architecture, 10)
        num_epochs = 5
        train_model(net=model,
                train_iter=train_iter,
                test_iter=test_iter,
                num_epochs=num_epochs,
                lr=0.01,
                device=try_gpu())
        # save model weights
        model_weights = 'VGG16-CIFAR10-epoch20'
        PATH = f'./DeepLearning/CV/checkpoints/CIFAR-10/{model_weights}.pth'
        save_model(model, PATH)

    else:
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224))
        ])

        test_dataset = torchvision.datasets.CIFAR10(
            "./DeepLearning/CV/data",
            False,
            trans,
            download=True
        )

        batch_size = 64
        test_iter = data.DataLoader(test_dataset, batch_size, shuffle=True)

        MODEL_PATH = './DeepLearning/CV/checkpoints/CIFAR-10/VGG16-CIFAR10-epoch20.pth'
        architecture = ((2,64), (2,128), (3,256), (3,512), (3,512))
        test_model = VGG(architecture, 10)
        test_model.load_state_dict(torch.load(MODEL_PATH))
        test_acc = evaluate_accuracy_gpu(test_model, test_iter)
        print(test_acc)

if __name__ == '__main__':
    main()