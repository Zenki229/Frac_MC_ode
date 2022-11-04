from Frac_MCPINN import *
from net import *
from train import *


dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Net(NL=2, NN=5).to(device=dev)
dtp = torch.float32
al = 0.5
te = 1
xe = 1
model = FracMCPINN(net, al, xe, te, dev)
train = Train(net, model, BATCH_SIZE=100)
train.train(epoch=10**5, lr=1e-3)