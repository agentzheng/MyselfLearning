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

net=Net(1,10,1)

print(net)

optimizer = torch.optim.SGD(net.parameters(), lr=0.02)

loss_func = torch.nn.MSELoss()


plt.ion()
for i in range(1000):
    out=net(x)
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