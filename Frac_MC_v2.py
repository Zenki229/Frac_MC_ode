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
        sample_num = 2
        tau_dist = Beta(torch.tensor([1-self.al]), torch.tensor([1.0]))
        epsi_t = 1e-6
        loss_eqn = 0
        t_init = torch.tensor([0.0], device=self.dev)
        t_all1 = torch.zeros([size, 1+sample_num], device=self.dev)
        t_all2 = torch.zeros([size, 1+sample_num], device=self.dev)
        lower = torch.ones(sample_num, device=self.dev) * epsi_t
        for enum in range(size):
            t_all1[enum, 0] = t[enum]
            t_all2[enum, 0] = t[enum]
            tau1 = tau_dist.sample(sample_shape=torch.tensor([sample_num])).to(device=self.dev).flatten()
            tau1 = torch.max(torch.stack((tau1, lower/t[enum]), dim=1), dim=1)[0]
            tau2 = tau_dist.sample(sample_shape=torch.tensor([sample_num])).to(device=self.dev).flatten()
            tau2 = torch.max(torch.stack((tau2, lower/t[enum]), dim=1), dim=1)[0]
            t_all1[enum, 1:-1] = t[enum]*(1-tau1)
            t_all2[enum, 1:-1] = t[enum]*(1-tau2)


        # for enum, _t in enumerate(t):
        #     lower = torch.ones(sample_num, device=self.dev)*epsi_t/_t
        #     tau1 = tau_dist.sample(sample_shape=torch.tensor([sample_num])).to(device=self.dev).flatten()
        #     tau1 = torch.max(torch.stack((tau1, lower), dim=1),dim=1)[0]
        #     tau2 = tau_dist.sample(sample_shape=torch.tensor([sample_num])).to(device=self.dev).flatten()
        #     tau2 = torch.max(torch.stack((tau2, lower), dim=1), dim=1)[0]
        #     aux_part2 = (self.net(_t) - self.net(t_init)) / _t ** al
        #     t_now = torch.ones_like(tau1, device=self.dev)*_t
        #     t_sample1 = t_now*(1-tau1)
        #     t_sample2 = t_now*(1-tau2)
        #     ode1 = al/(1-al)*_t**(1-al)*torch.sum((self.net(t_now.reshape(-1, 1))-self.net(t_sample1.reshape(-1, 1)))/(tau1*_t))
        #     ode2 = al/(1-al)*_t**(1-al)*torch.sum((self.net(t_now.reshape(-1, 1))-self.net(t_sample2.reshape(-1, 1)))/(tau2*_t))
        #     loss_eqn += (ode1+sample_num*(aux_part2+self.net(_t)))*(ode2+sample_num*(aux_part2+self.net(_t)))
        # loss_eqn = loss_eqn/(sample_num*size)
        loss_eqn = loss_eqn**2
        loss_init = torch.sum((self.net(t_init)-torch.tensor([1.0], device=self.dev))**2)
        return loss_eqn+loss_init




