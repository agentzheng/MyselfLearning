import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y=x.pow(3)+0.2*torch.rand(x.size())

# torch can only train on Variable, so convert them to Variable
x,y = Variable(x), Variable(y)

plt.scatter(x.data.numpy(), y.data.numpy())
plt.show()

class Net(torch.nn.Module):
    def __init__(self,n_input_dim,n_hidden_dim,n_output_dim):
        super(Net,self).__init__()
        self.hidden=torch.nn.Linear(n_input_dim,n_hidden_dim)
        self.out=torch.nn.Linear(n_hidden_dim,n_output_dim)

    def forward(self, x):
        x=F.relu(self.hidden(x))
        x=self.out(x)
        return x



net2=torch.nn.Sequential(
    torch.nn.Linear(1,10),
    torch.nn.ReLU(),
    torch.nn.Linear(10,1)
)

print(net2)

optimizer = torch.optim.SGD(net2.parameters(), lr=0.5)

loss_func = torch.nn.MSELoss()


plt.ion()

for i in range(200):
    out=net2(x)
    loss=loss_func(out,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i%10 ==0:
        print('loss=%.2f'%loss)
        plt.plot(x.data.numpy(), out.data.numpy(), 'r-', lw=5)
        plt.pause(0.1)

plt.ioff()
plt.show()


torch.save(net2, 'net.pkl')  # 保存整个网络
torch.save(net2.state_dict(), 'net_params.pkl') #只保存参数


def restore_net():
    # restore entire net1 to net2
    net2 = torch.load('net.pkl')
    prediction = net2(x)

def restore_params():
    # 新建 net3
    net3 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )

    # 将保存的参数复制到 net3
    net3.load_state_dict(torch.load('net_params.pkl'))
    prediction = net3(x)