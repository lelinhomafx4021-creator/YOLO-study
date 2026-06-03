import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
x=torch.rand(100,1)
y=2*x+1+x*0.1
class MyModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear=nn.Linear(1,1)
    def forward(self,x):
        return self.linear(x)
model=MyModel()
class MyDataset(Dataset):
    def __init__(self,x,y):
        self.x=x
        self.y=y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx],self.y[idx]
criterion=nn.MSELoss()
dataset=MyDataset(x,y)
dataloader=DataLoader(dataset,batch_size=16,shuffle=True)
optimizer=torch.optim.SGD(model.parameters(),lr=0.1)
for epoch in range(100):
    for x_batch,y_batch in dataloader:
        pred=model(x_batch)
        loss=criterion(pred,y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
print(f"学到的w:{model.linear.weight.item()}")
print(f"学到的w:{model.linear.bias.item()}")
