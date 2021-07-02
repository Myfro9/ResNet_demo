import torch

from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from resnet import ResNet18

batchsz = 32
cifar_train = datasets.CIFAR10('cifar', True, transform=transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(0.5),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
]), download=True)
cifar_train = DataLoader(cifar_train, batch_size=batchsz, shuffle=True)

cifar_test = datasets.CIFAR10('cifar', False, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]), download=True)
cifar_test = DataLoader(cifar_test, batch_size=batchsz, shuffle=True)

x, label = iter(cifar_train).next()
print('x:', x.shape, 'label:', label.shape)

device = torch.device('cuda')
print('The device:', device)
ResNet18_model = ResNet18().to(device)

criteon = nn.CrossEntropyLoss().to(device)   # 已经包含了softmax操作
optimizer = optim.Adam(ResNet18_model.parameters(),lr=1e-3)
print(ResNet18_model)

for epoch in range(1000):
    ResNet18_model.train()
    for batchidx, (x,label) in enumerate(cifar_train):
        #print('training batchidx: ',batchidx)
        x,label =x.to(device),label.to(device)
        logits = ResNet18_model(x)
        loss = criteon(logits, label)

        # training
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('epoch: ',epoch, ' loss:', loss.item() )

    ResNet18_model.eval()
    with torch.no_grad():
        total_correct = 0
        total_num = 0
        for x,label in cifar_test:
            x,label = x.to(device), label.to(device)
            logits = ResNet18_model(x)
            pred = logits.argmax(dim=1)
            correct = torch.eq(pred,label).float().sum().item()
            total_correct = total_correct + correct
            total_num = total_num + x.size(0)
        acc = total_correct / total_num
        print('epoch: ', epoch, ' test acc: ',acc)


if __name__ == '__main__':
    main()