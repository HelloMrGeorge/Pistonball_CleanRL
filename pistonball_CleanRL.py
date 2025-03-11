"""Basic code which shows what it's like to run PPO on the Pistonball env using the parallel API, this code is inspired by CleanRL.

This code is exceedingly basic, with no logging or weights saving.
The intention was for users to have a (relatively clean) ~200 line file to refer to when they want to design their own learning algorithm.

Author: Jet (https://github.com/jjshoots)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

__all__ = ["Agent", "Agent_ADG", "batchify_obs", "batchify", "unbatchify"]

class Agent(nn.Module):
    def __init__(self, num_actions):
        super().__init__()

        self.network = nn.Sequential(
            self._layer_init(nn.Conv2d(4, 32, 3, padding=1)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            self._layer_init(nn.Conv2d(32, 64, 3, padding=1)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            self._layer_init(nn.Conv2d(64, 128, 3, padding=1)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
            self._layer_init(nn.Linear(128 * 8 * 8, 512)),
            nn.ReLU(),
        )
        self.actor = self._layer_init(nn.Linear(512, num_actions), std=0.01)
        self.critic = self._layer_init(nn.Linear(512, 1))

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
    
    def forward(self, x, action=None):
        return self.get_action_and_value(x, action)
    

class Agent_ADG_deprecated(nn.Module):
    def __init__(self, num_actions, embedding_dim=32, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        
        self.device = device
        self.network = nn.Sequential(
            self._layer_init(nn.Conv2d(4, 32, 3, padding=1)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            self._layer_init(nn.Conv2d(32, 64, 3, padding=1)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            self._layer_init(nn.Conv2d(64, 128, 3, padding=1)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
            self._layer_init(nn.Linear(128 * 8 * 8, 512)),
            nn.ReLU(),
        )
        
        self.action_embedding1 = nn.Embedding(num_actions, embedding_dim)
        self.action_embedding2 = nn.Embedding(num_actions, embedding_dim)

        self.additional_layer = nn.Sequential(
            self._layer_init(nn.Linear(512 + 2 * embedding_dim, 128)),
            nn.ReLU(),
        )

        self.actor = self._layer_init(nn.Linear(128, num_actions), std=0.01)
        self.critic = self._layer_init(nn.Linear(128, 1))
    
    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer
    
    def get_value(self, x):
        return self.critic(self.network(x / 255.0))
    
    def get_action_and_value(self, x, actions):
        action1, action2 = actions
        hidden = self.network(x / 255.0)
        action1_embedded = self.action_embedding1(action1).unsqueeze(0)
        action2_embedded = self.action_embedding2(action2).unsqueeze(0)
        hidden = torch.cat((hidden, action1_embedded, action2_embedded), dim=1)
        hidden = self.additional_layer(hidden)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

    def forward(self, x, action=None):
        num_agents = x.shape[0]
        x = x.unsqueeze(1)
        action_base = torch.ones(num_agents + 2, dtype=torch.int).to(self.device)
        log_prob = torch.zeros(num_agents).to(self.device)
        entropy = torch.zeros(num_agents).to(self.device)
        value = torch.zeros(num_agents).to(self.device)
        if action is not None:
            action_base[:-2] = action
            for ind in range(num_agents - 1, -1, -1):
                depend_actions = action_base[ind+1:ind+3]
                _, log_prob[ind], entropy[ind], value[ind] = self.get_action_and_value(x[ind], depend_actions)
        else:
            for ind in range(num_agents - 1, -1, -1):
                depend_actions = action_base[ind+1:ind+3]
                action_base[ind], log_prob[ind], entropy[ind], value[ind] = self.get_action_and_value(x[ind], depend_actions)
        return action_base[:-2], log_prob, entropy, value


class Agent_ADG(nn.Module):
    def __init__(self, num_actions, embedding_dim=64):
        super().__init__()
        
        self.network = nn.Sequential(
            self._layer_init(nn.Conv2d(4, 32, 3, padding=1)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            self._layer_init(nn.Conv2d(32, 64, 3, padding=1)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            self._layer_init(nn.Conv2d(64, 128, 3, padding=1)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
            self._layer_init(nn.Linear(128 * 8 * 8, 512)),
            nn.ReLU(),
        )
        
        self.action_embedding1 = nn.Embedding(num_actions + 1, embedding_dim)
        self.action_embedding2 = nn.Embedding(num_actions + 1, embedding_dim)

        self.additional_layer = nn.Sequential(
            self._layer_init(nn.Linear(512 + 2 * embedding_dim, 128)),
            nn.ReLU(),
        )

        self.actor = self._layer_init(nn.Linear(128, num_actions), std=0.01)
        self.critic = self._layer_init(nn.Linear(128, 1))
    
    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer
    
    def forward(self, x, depend_actions, action=None):
        action1, action2 = depend_actions[:, 0], depend_actions[:, 1]
        action1_embedded = self.action_embedding1(action1)
        action2_embedded = self.action_embedding2(action2)

        hidden = self.network(x / 255.0)
        hidden = torch.cat((hidden, action1_embedded, action2_embedded), dim=1)
        hidden = self.additional_layer(hidden)

        logits = self.actor(hidden)
        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


def batchify_obs(obs, device):
    """Converts PZ style observations to batch of torch arrays."""
    # convert to list of np arrays
    obs = np.stack([obs[a] for a in obs], axis=0)
    # transpose to be (batch, channel, height, width)
    obs = obs.transpose(0, -1, 1, 2)
    # convert to torch
    obs = torch.tensor(obs).to(device)

    return obs


def batchify(x, device):
    """Converts PZ style returns to batch of torch arrays."""
    # convert to list of np arrays
    x = np.stack([x[a] for a in x], axis=0)
    # convert to torch
    x = torch.tensor(x).to(device)

    return x


def unbatchify(x, env):
    """Converts np array to PZ style arguments."""
    x = x.cpu().numpy()
    x = {a: x[i] for i, a in enumerate(env.possible_agents)}

    return x
