import torch

class GAE():
    def __init__(self, gamma, tau, device):
        self.gamma = gamma
        self.tau = tau
        self.device = device

    def calc_GAE(self, rewards, values, masks):
        gae = 0
        values.append(torch.zeros((1, 1)).to(self.device))
        returns = []
        for iter in reversed(range(len(rewards))):
            delta = rewards[iter] + self.gamma * values[iter + 1] * masks[iter] - values[iter]
            gae = delta + self.gamma * self.tau * masks[iter] * gae
            returns.insert(0, gae + values[iter])
        values.pop()
        return returns