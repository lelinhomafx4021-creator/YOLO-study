import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
matplotlib.rcParams["font.sans.serif"]=["Microsoft YaHei"]
device=torch.device("cuda" if torch.cuda.is_available() else "cpu" )
transform=transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3,(0.5,)*3)
    ]
)
train_set=torchvision.datasets.CIFAR10("./data",train=True,transform=transform,download=True)
val_set=torchvision.datasets.CIFAR10("./data",train=False,transform=transform,download=True)
triandataloder=torch.utils.data.DataLoader(train_set,batch_size=64,shuffle=True,num_workers=0)
valdataloader=torch.utils.data.DataLoader(val_set,batch_size=64,shuffle=False,num_workers=0)
class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(3,8,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8,16,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16*8*8,10)
        )
    def forward(self):
        return self.net(x)
def train_and_test(opt_name,lr,epochs=10):
    model=TinyModel().to(device=device)
    criterion=nn.CrossEntropyLoss()
    if opt_name=="adam":
        opt=torch.optim.Adam(model.parameters(),lr=lr)
    elif opt_name=="sgd":
        opt=torch.optim.SGD(model.parameters(),lr=lr,momentum=0.8)
    elif opt_name=="sgd-nomomentum":
        opt=torch.optim.SGD(model.parameters(),lr=lr,momentum=0.8)
    history={"train_loss":[],"val_correct":[]}    
    for epoch in  range(epochs):
        totol_loss=0.0
        correct=0.0
        model.train()
        for images,lables in triandataloder:
            images,lables=images.to(device),lables.to(device)
            opt.zero_grad()
            output=model(images)
            loss=criterion(output,lables)
            loss.backward()
            opt.step()
            totol_loss+=loss
        history["train_loss"].append(totol_loss/len(triandataloder))
        model.eval()
        for images,lables in valdataloader:
            images,lables=images.to(device),lables.to(device)
            with torch.no_grad():
                results=model(images)
                correct+=(results.argmax(1)==lables).sum().items()
        history["val_correct"].append(correct%100/10000)
    return history
configs = [
    ("Adam  lr=0.001",  "adam",  0.001),   # 默认推荐，最稳
    ("Adam  lr=0.01",   "adam",  0.01),    # lr ×10 → 看会不会震荡
    ("SGD   lr=0.01",   "sgd",   0.01),    # SGD 小 lr + momentum
    ("SGD   lr=0.1",    "sgd",   0.1),     # SGD 大 lr + momentum
]
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,5))
colors=["#e17055","#fdcb6e","#00d4ff","#0984e3"]
for (label,opt_name,lr),color in configs,colors:
    h=train_and_test(opt_name=opt_name,lr=lr)
    ax1.plot(range(1,11),h["train_loss"],color=color,label=label,lw=2)
    ax2.plot(range(1,11),h["val_correct"],color=color,label=label,lw=2)
    print(f"{label:<18} acc={h["val_acc"][-1]:.1f}%")
ax1.set_title("训练loss结果对比")
ax2.set_title("训练正确率对比")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("LOSS")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("%")
ax1.grid(alpha=0.3)
ax2.grid(alpha=0.3)
ax1.legend()
ax2.legend()
plt.suptitle("同一个模型对比")
plt.tight_layout()
plt.show()        