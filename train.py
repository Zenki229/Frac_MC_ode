from lib import *


class Train():
    def __init__(self, net, subdiffusion, BATCH_SIZE):
        self.errors = []
        self.BATCH_SIZE = BATCH_SIZE
        self.net = net
        self.model = subdiffusion

    def train(self, epoch, lr):
        std = 1
        optimizer = optim.Adam(self.net.parameters(), lr=lr)
        schedule = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10000, gamma=0.5)
        avg_loss = 0
        for e in range(epoch):
            optimizer.zero_grad()
            loss = self.model.loss_func(self.BATCH_SIZE)
            avg_loss = avg_loss + float(loss.item())
            loss.backward()
            optimizer.step()
            if e % 50 == 49:
                loss = avg_loss/50
                if loss < std:
                    torch.save(self.net, 'net_model.pkl')
                    std = loss
                print("Epoch {} - lr {} -  loss: {}".format(e, schedule.get_last_lr(), loss))
                avg_loss = 0
                # error = self.model.loss_func(2**8)
                # self.errors.append(error.detach())
            schedule.step()

    def get_errors(self):
        return self.errors
