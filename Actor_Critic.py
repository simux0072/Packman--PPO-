import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch
torch.set_default_tensor_type(torch.DoubleTensor)

class Actor_Critic(nn.Module):
    def __init__(self, device, num_layer):
        super(Actor_Critic, self).__init__()

        self.device = device

        self.conv1_1 = nn.Conv2d(in_channels=num_layer, out_channels=num_layer, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=num_layer, out_channels=num_layer, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(in_channels=num_layer, out_channels=num_layer, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=num_layer, out_channels=num_layer, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(in_channels=num_layer, out_channels=num_layer, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=num_layer, out_channels=num_layer, kernel_size=3, padding=1)

        self.conv_res1 = nn.Conv2d(in_channels=num_layer, out_channels=num_layer, kernel_size=1)
        self.conv_res2 = nn.Conv2d(in_channels=num_layer, out_channels=num_layer, kernel_size=1)
        self.conv_res3 = nn.Conv2d(in_channels=num_layer, out_channels=num_layer, kernel_size=1)


        self.critic = nn.Sequential(
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=1)
        )

        self.actor = nn.Sequential(
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=4),
            nn.Softmax(dim=-1)
        )

    def forward(self, state, direction, hunter):

        state = state.to(self.device)

        res = self.conv_res1(state)
        state = F.relu(self.conv1_1(state))
        state = F.relu(self.conv1_2(state) + res)

        state = F.max_pool2d(state, kernel_size=3, stride=2)

        res = self.conv_res2(state)
        state = F.relu(self.conv2_1(state))
        state = F.relu(self.conv2_2(state) + res)

        state = F.max_pool2d(state, kernel_size=3, stride=2)

        res = self.conv_res3(state)
        state = F.relu(self.conv3_1(state))
        state = F.relu(self.conv3_2(state) + res)

        state = torch.cat((state.flatten(start_dim=1), direction.to(self.device), hunter.to(self.device)), dim=1)

        policy_prob = self.actor(state)
        dist = Categorical(policy_prob)
        value = self.critic(state)

        return value, dist