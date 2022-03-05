import torch
import torch.optim as optim
import numpy as np

class Agent_ghost():
    def __init__(self, lr, model, epsilon, GAE, checkpoint):
        self.lr = lr
        self.model = model
        self.epsilon = epsilon
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.gae = GAE
        if checkpoint is not None:
            self.optimizer.load_state_dict(checkpoint['ghost_optimizer_state_dict'])

    def get_returns(self, rewards, values, masks):
        return self.gae.calc_GAE(rewards, values, masks)
    
    def get_action(self, state, direction, hunter_state):
        value, dist = self.model(state, direction, hunter_state)
        return value, dist
        
    def get_minibatch(self, mini_batch_size, states, directions, hunter_states, actions, log_probs, returns, advantages):
        batch_size = len(states)
        if batch_size // mini_batch_size == 0:
            rand_idx = np.random.randint(0, batch_size, batch_size)
            yield states[rand_idx], directions[rand_idx], hunter_states[rand_idx], actions[rand_idx], log_probs[rand_idx], advantages[rand_idx], returns[rand_idx]
        else:
            for _ in range(batch_size // mini_batch_size):
                rand_idx = np.random.randint(0, batch_size, mini_batch_size)
                yield states[rand_idx], directions[rand_idx], hunter_states[rand_idx], actions[rand_idx], log_probs[rand_idx], advantages[rand_idx], returns[rand_idx]

    def train(self, epochs, mini_batch_size, states, directions, hunter_states, actions, log_probs, returns, advantages):
        loss_temp = 0
        for _ in range(epochs):
            for state, direction, hunter_state, action, old_log_probs, advantage, return_ in self.get_minibatch(mini_batch_size, states, directions, hunter_states, actions, log_probs, returns, advantages):
                value, dist = self.model(state, direction, hunter_state)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action)

                ratio = (new_log_probs / old_log_probs).exp()
                surr_1 = ratio * advantage
                surr_2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage

                actor_loss = - torch.min(surr_1, surr_2).mean()
                critic_loss = ((return_ - value) ** 2).mean()

                loss = 0.5 * critic_loss + actor_loss - 0.1 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_temp += loss.item()
        return loss_temp                

class Agent_player():
    def __init__(self, lr, model, epsilon, GAE, checkpoint):
        self.lr = lr
        self.model = model
        self.epsilon = epsilon
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.gae = GAE
        if checkpoint is not None:
            self.optimizer.load_state_dict(checkpoint['player_optimizer_state_dict'])

    def get_returns(self, rewards, values, masks):
        return self.gae.calc_GAE(rewards, values, masks)
    
    def get_action(self, state, direction, hunter_state):
        value, dist = self.model(state, direction, hunter_state)
        return value, dist
        
    def get_minibatch(self, mini_batch_size, states, directions, hunter_states, actions, log_probs, returns, advantages):
        batch_size = len(states)
        if batch_size // mini_batch_size == 0:
            rand_idx = np.random.randint(0, batch_size, batch_size)
            yield states[rand_idx], directions[rand_idx], hunter_states[rand_idx], actions[rand_idx], log_probs[rand_idx], advantages[rand_idx], returns[rand_idx]
        else:
            for _ in range(batch_size // mini_batch_size):
                rand_idx = np.random.randint(0, batch_size, mini_batch_size)
                yield states[rand_idx], directions[rand_idx], hunter_states[rand_idx], actions[rand_idx], log_probs[rand_idx], advantages[rand_idx], returns[rand_idx]

    def train(self, epochs, mini_batch_size, states, directions, hunter_states, actions, log_probs, returns, advantages):
        loss_temp = 0
        for _ in range(epochs):
            for state, direction, hunter_state, action, old_log_probs, advantage, return_ in self.get_minibatch(mini_batch_size, states, directions, hunter_states, actions, log_probs, returns, advantages):
                value, dist = self.model(state, direction, hunter_state)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action)

                ratio = (new_log_probs / old_log_probs).exp()
                surr_1 = ratio * advantage
                surr_2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage

                actor_loss = - torch.min(surr_1, surr_2).mean()
                critic_loss = ((return_ - value) ** 2).mean()

                loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_temp += loss.item()
        return loss_temp