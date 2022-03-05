import torch

class memory():
    def __init__(self, device):
        self.states = []
        self.directions = []
        self.actions = []
        self.values = []
        self.rewards = []
        self.masks = []
        self.log_probs = []
        self.hunter_states = []

        self.memory_lenght = 0
        self.device = device

    def reset(self):
        self.states = []
        self.directions = []
        self.actions = []
        self.values = []
        self.rewards = []
        self.masks = []
        self.log_probs = []
        self.hunter_states = []

        self.memory_lenght = 0
    
    def add_to_memory(self, state, direction, hunter_state, action, value, reward, mask, log_prob):
        self.states.append(state)
        self.directions.append(direction)
        self.actions.append(action)
        self.values.append(value)
        self.rewards.append(reward)
        self.masks.append(mask)
        self.log_probs.append(log_prob)
        self.hunter_states.append(hunter_state)

        self.memory_lenght += 1

    def make_data(self, returns):
        returns = torch.cat(returns).squeeze(dim=1).detach()

        return torch.cat(self.states).to(self.device).detach(), \
                torch.cat(self.directions).to(self.device).detach(), \
                torch.cat(self.hunter_states).to(self.device).detach(), \
                torch.tensor(self.actions).to(self.device).detach(), \
                torch.tensor(self.log_probs).to(self.device).detach(), \
                returns, returns - torch.cat(self.values).flatten(start_dim=0).to(self.device).detach()