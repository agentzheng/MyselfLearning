import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        # 一个图像输入通道，六个输出通道 5*5卷积核
        self.conv1=nn.Conv2d(1,6,5)

        # 定义卷积层，输入6张特征图，输出16张特征图
        self.conv2=nn.Conv2d(6,16,5)

        #全连接层
        #线性链接(y=Wx+b),16*5*5个节点到120个节点上
        self.fc1=nn.Linear(16*5*5,120)

        self.fc2=nn.Linear(120,84)

        self.fc3=nn.Linear(84,10)

    def forward(self, x):
        x=F.max_pool2d(F.relu(self.conv1(x)),(2,2))

        x=F.max_pool2d(F.relu(self.conv2(x)),2)
        x=x.view(-1,self.num_flat_features(x))
        x=F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x=self.fc3(x)
        return x


    def num_flat_features(self,x):
        size=x.size()[1:]
        num_features=1
        for s in size:
            num_features*=s
        return num_features

net=Net()
print(net)

params=list(net.parameters())
print(len(params))
print(params[0].size())

input=torch.randn(1,1,32,32)
out=net(input)
print(out)



net.zero_grad()

out.backward(torch.randn(1,10))

output=net(input)
target=torch.arange(1,11)
target=target.view(1,-1)

criterion=nn.MSELoss()

loss=criterion(output,target)


print(loss)

net.zero_grad()
print(net.conv1.bias.grad)

loss.backward()

print(net.conv1.bias.grad)

