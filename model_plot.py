import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from lib import *
from train import *
from net import *

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = torch.load('net_model.pkl')
al = 0.5
tau = 0.05
te = 1
xe = 1
ye = 1
t_range = torch.linspace(0, te, 100).to(device=dev)
data = net(t_range.reshape(-1, 1))
plt.plot(t_range.detach().numpy(), data.detach().numpy())
plt.show()