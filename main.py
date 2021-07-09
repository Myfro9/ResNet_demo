import torch

from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from resnet import ResNet18


#### 1) Data loader
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


##### 2) 定义Model
device = torch.device('cuda')
print('The device:', device)
ResNet18_model = ResNet18().to(device)


##### 3) 定义Optimizer and Criteon
criteon = nn.CrossEntropyLoss().to(device)   # 已经包含了softmax操作
optimizer = optim.Adam(ResNet18_model.parameters(),lr=1e-3)
print(ResNet18_model)


##### 4）epoch训练及验证
for epoch in range(1000):
    ResNet18_model.train()  # 指定当前模型为训练
    for batchidx, (x,label) in enumerate(cifar_train):   # 每个Batch循环训练
        #print('training batchidx: ',batchidx)
        x,label =x.to(device),label.to(device)
        logits = ResNet18_model(x)
        loss = criteon(logits, label)

        # training
        optimizer.zero_grad()  #优化器清空梯度
        loss.backward()        #误差反向传播
        optimizer.step()       #更新优化器

    print('epoch: ',epoch, ' loss:', loss.item() )

    ResNet18_model.eval()  #指定当前模型为验证
    with torch.no_grad():  #验证没有梯度访问
        total_correct = 0
        total_num = 0
        for x,label in cifar_test:   #每个Batch循环验证
            x,label = x.to(device), label.to(device)
            logits = ResNet18_model(x)
            pred = logits.argmax(dim=1)
            correct = torch.eq(pred,label).float().sum().item()
            total_correct = total_correct + correct
            total_num = total_num + x.size(0)
        acc = total_correct / total_num  #计算当前epoch里所有Batch的累计准确率
        print('epoch: ', epoch, ' test acc: ',acc)


if __name__ == '__main__':
    main()