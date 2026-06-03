import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
train_set=torchvision.datasets.CIFAR10("./data",train=True,transform=transform,download=True)
test_set=torchvision.datasets.CIFAR10("./data",train=False,transform=transform,download=True)
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(3,16,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16,32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier=nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*8*8,128),
            nn.ReLU(),
            nn.Linear(128,10)
        )
    def forward(self,x):
        x=self.features(x)
        x=self.classifier(x)
        return x

def train_and_validata(batch_size=32,epochs=20,lr=0.001,optimizer_name="adam",weight_decay=0.0):
    model=SimpleCNN().to(device)
    trainloader=torch.utils.data.DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=0)
    testloader=torch.utils.data.DataLoader(test_set,batch_size=batch_size,shuffle=False,num_workers=0)
    criterion=nn.CrossEntropyLoss()
    if optimizer_name=="adam":
        optimizer=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)
    elif optimizer_name=="sgd":
        optimizer=torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9,weight_decay=weight_decay)

    for epoch in range(epochs):
        model.train()                              # ← 切训练模式
        loss_total=0.0
        for images,labels in trainloader:          # ← 变量名改 labels
            images,labels=images.to(device),labels.to(device)
            output=model(images)
            loss=criterion(output,labels)           # ← 统一拼写
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_total+=loss.item()                # ← .item() 转 python 数字
        print(f"epoch {epoch+1}/{epochs} 训练 loss: {loss_total/len(trainloader):.4f}")

    # ── 验证阶段 ──
    model.eval()
    correct=0
    total=0
    with torch.no_grad():
        for images,labels in testloader:
            images,labels=images.to(device),labels.to(device)
            output=model(images)
            _,predicted=torch.max(output,1)
            total+=labels.size(0)
            correct+=(predicted==labels).sum().item()   # ← .sum().item()
    return 100*correct/total
if __name__=="__main__":
    print("=" * 55)
    print("实验1：不同 batch_size 对比（lr=0.001, epochs=20, Adam）")
    print("-" * 55)
    for batch in [32,64,128]:
        acc=train_and_validata(batch_size=batch)
        print(f"batch_size={batch:<5} → 测试准确率 {acc:.1f}%")
    print("结论：batch_size 适中最好，太小噪声大，太大泛化差\n")

    print("=" * 55)
    print("实验2：不同学习率对比（batch=64, epochs=20, Adam）")
    print("-" * 55)
    for lr in [0.1,0.01,0.001]:
        acc=train_and_validata(lr=lr)
        print(f"lr={lr:<8} → 测试准确率 {acc:.1f}%")
    print("结论：lr=0.001 通常是最稳的起点\n")

    print("=" * 55)
    print("实验3：不同 epochs 对比（batch=64, lr=0.001, Adam）")
    print("-" * 55)
    for ep in [20,40,60]:
        acc=train_and_validata(epochs=ep)
        print(f"epochs={ep:<5} → 测试准确率 {acc:.1f}%")
    print("结论：epochs 增加到一定程度后收益递减\n")

    print("=" * 55)
    print("实验4：Adam vs SGD（batch=64, lr=0.001, epochs=20）")
    print("-" * 55)
    for opt in ["adam","sgd"]:
        acc=train_and_validata(optimizer_name=opt)
        print(f"optimizer={opt:<5} → 测试准确率 {acc:.1f}%")
    print("结论：Adam 收敛快，SGD 需要更大 lr\n")

    print("=" * 55)
    print("实验5：weight_decay 对比（batch=64, lr=0.001, epochs=20, Adam）")
    print("-" * 55)
    for wd in [0.0, 1e-4, 1e-3]:
        acc=train_and_validata(batch_size=64, lr=0.001, epochs=20, weight_decay=wd)
        print(f"weight_decay={wd:<6} → 测试准确率 {acc:.1f}%")
    print("结论：weight_decay 不是越大越好，适度（1e-4）可能提升泛化\n")
    print("=" * 55)
    print("调参核心原则：一次只改一个参数，否则不知道是哪个起的作用")    