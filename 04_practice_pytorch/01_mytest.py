import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3,(0.5,)*3)
])
train_set=torchvision.datasets.CIFAR10("./data",train=True,transform=transform,download=True)
test_set=torchvision.datasets.CIFAR10("./data",train=False,transform=transform,download=True)
trainloader=torch.utils.data.DataLoader(train_set,batch_size=64,shuffle=True)
evalloader=torch.utils.data.DataLoader(test_set,batch_size=64,shuffle=False)
class TinyCNN(nn.Module):
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
    def forward(self,x):
        return self.net(x)
model=TinyCNN().to(device)
before_weight=model.net[0].weight.data.clone()
print(f"训练前的conv1weight[0.0.0.0]={before_weight[0,0,0,0]:.6f}")
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
for epoch in range(3):
    model.train()
    for imgs,labels in trainloader:
        imgs,labels=imgs.to(device),labels.to(device)
        optimizer.zero_grad()
        loss=criterion(model(imgs),labels)
        loss.backward()
        optimizer.step()
    model.eval()
    correct=0
    with torch.no_grad():
        for imags,labels in evalloader:
            imags,labels=imags.to(device),labels.to(device)
            results=model(imags)
            correct+=(results.argmax(1)==labels).sum().item()
    print(f"Epoch{epoch+1}:acc={100*correct/10000:.1f}%")
after_weight=model.net[0].weight.data
print(f"训练后的参数{after_weight[0,0,0,0]:.6f}")
torch.save(model.state_dict(),"./cnndata.pth")
new_model=TinyCNN().to(device)
new_model.load_state_dict(torch.load("./cnndata.pth",map_location=device))
new_model.eval()
loaded_weight=new_model.net[0].weight.data
print("加载后的参数")
imgs,labels=next(iter(evalloader))
imgs=imgs.to(device)
with torch.no_grad():
    preds=new_model(imgs).argmax(1)
classes = ["飞机","汽车","鸟","猫","鹿","狗","青蛙","马","船","卡车"]
for i  in range(5):
    print(f"图{i+1},预测值{classes[preds[i]]},实际值{classes[labels[i]]}")
