from lib import *
from torch.distributions import Beta


class FracMCPINN():

    def __init__(self, net, al, xe, te, dev):
        self.te = te
        self.xe = xe
        self.al = al
        self.net = net
        self.dev = dev

    def sample(self, size):
        t = torch.rand([size, 1], device=self.dev)
        return t

    def loss_func(self, size):
        al = self.al
        t = self.sample(size=size)
        sample_num = 20
        tau_dist = Beta(torch.tensor([1-self.al]), torch.tensor([1.0]))
        epsi_t = 1e-6

        loss_eqn = 0
        t_init = torch.tensor([0.0], device=self.dev)
        t_all1 = torch.zeros([size, 1+sample_num], device=self.dev)
        t_all2 = torch.zeros([size, 1+sample_num], device=self.dev)
        lower = torch.ones(sample_num, device=self.dev) * epsi_t
        tau1 = tau_dist.sample(sample_shape=torch.tensor([size, sample_num])).to(device=self.dev).reshape(size, sample_num)
        tau2 = tau_dist.sample(sample_shape=torch.tensor([size, sample_num])).to(device=self.dev).reshape(size, sample_num)
        t_1 = tau1*t
        t_2 = tau2*t
        t_1[t_1<epsi_t] = epsi_t
        t_2[t_2<epsi_t] = epsi_t
        t_aux = t.repeat((1, sample_num))
        t_sample1 = t_aux-t_1
        t_sample2 = t_aux-t_2
        aux_part2 = (self.net(t) - self.net(torch.zeros_like(t, device=self.dev))) / t ** al
        aux_part2 = aux_part2.repeat((1, sample_num))
        aux_part3 = self.net(t)
        aux_part3 = aux_part3.repeat((1, sample_num))
        U1 = self.net(t_aux.reshape(-1, 1))-self.net(t_sample1.reshape(-1, 1))
        U1 = U1.reshape(size, sample_num)/t_1*al/(1-al)*t_aux**(1-al)+aux_part2+aux_part3
        U2 = self.net(t_aux.reshape(-1, 1)) - self.net(t_sample2.reshape(-1, 1))
        U2 = U2.reshape(size, sample_num)/t_2*al/(1-al)*t_aux**(1-al)+aux_part2+aux_part3
        loss_eqn = torch.sum((U1*U2).flatten())/(sample_num*size)
        loss_init = (self.net(t_init)-torch.tensor([1.0], device=self.dev)).flatten()**2
        return loss_eqn+loss_init




