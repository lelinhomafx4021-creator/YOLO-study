import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader,random_split
print(torch.rand(100,1))
x=torch.rand(200,1)
y=2*x+1+0.1*torch.randn(100,1)
class My_Dataset(Dataset,x,y):
    def __init__(self):
        super().__init__()
        self.x=x
        self.y=y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx],self.y[idx]
class My_Model(nn.Module):
    def __init__(self):
        super.__init__()
        self.linear=nn.Linear(1,1)
    def forward(self,x):
        return self.linear(x)

def train_one_epoch(model,dataloader,creterion,optimizer):
    model.train()
    train_loss_total=0.0
    for x_batch,y_batch in dataloader:
        optimizer.zero_grad()
        pred=model(x_batch)
        loss=creterion(pred,y_batch)
        loss.backward()
        optimizer.step()
        train_loss_total+=loss
    return train_loss_total/len(dataloader)

def validata(model,dataloder,creterion):
    model.eval()
    total_loss=0.0
    with torch.no_grad():
        for x_batch,y_batch in dataloder:
            answer=model(x_batch)
            loss=creterion(anwser,y_batch)
            total_loss+=loss
        return total_loss/len(dataloder)
def predict_samples(model):
    model.eval()
    test_x=torch.tensor([[0.1],[0.5],[0.9]])
    with torch.no_grad():
        pred_y=model(test_x)
    for i in range(len(test_x)):
        x_value=test_x[i].item
        pred_value=pred_y[i].item()
        true_value = 2 * x_value + 1
        print(
            f"x={x_value:.2f} | 预测 y={pred_value:.3f} | 理论 y={true_value:.3f}"
        )

def main():
    dataset=My_Dataset(x,y)
    train_size=int(0.8*len(dataset))
    val_size=len(dataset)-train_size
    train_dataset,val_dataset=random_split(dataset,train_size,val_size)
    train_loader=DataLoader(train_dataset,batch_size=16,shuffle=True)
    val_loader=DataLoader(val_dataset,batch_size=16,shuffle=False)
    model=My_Model()
    criterion=nn.MSELoss()
    optimizer=torch.optim.SGD(model.parameters(),lr=0.1)
    num_epochs=30
    for epoch in range(num_epochs):
        train_loss=train_one_epoch(model,train_dataset,criterion,optimizer)
        validata_loss=validata(model,val_dataset,creterion)
        print(f"Epoch:{epoch+1:02d}/{num_epochs}|"
              f"train:loss={train_loss}|"
              f"validate:loss={validata_loss}"
              )
    print(f"最后的偏重为：{model.linear.weight.item():.4f}")
    print(f"最后的偏置为：{model.linear.bias.item():.4f}")
    predict_samples(model)
if __name__=="__main__":
    main




    