"""
sac_model.py

This module implements the core components of the Soft Actor-Critic (SAC)
algorithm for continuous control tasks. It includes implementations of:

- PrioritizedReplayBuffer: A replay buffer with prioritized sampling.
- ReplayBuffer: A simple uniform replay buffer.
- GaussianPolicy: The actor network that outputs a Gaussian distribution over actions,
  with layer normalization to stabilize training.
- QNetwork: The critic network for estimating Q-values, also with layer normalization.
- SACAgent: The SAC agent that aggregates the actor, critics, target networks, and 
  automatic entropy tuning.

All components run on the available device (CPU or GPU).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PrioritizedReplayBuffer:
    """
    A prioritized replay buffer that samples transitions with probabilities 
    proportional to their TD error (raised to a power alpha).

    Attributes:
        capacity (int): Maximum number of transitions to store.
        buffer (list): List of transitions.
        pos (int): Current insertion index.
        priorities (np.array): Array storing the priority of each transition.
        alpha (float): Exponent that determines how much prioritization is used.
    """
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha

    def push(self, state, action, reward, next_state, done):
        """
        Add a new transition to the buffer.
        
        Args:
            state (np.array): Current state.
            action (np.array): Action taken.
            reward (float): Reward received.
            next_state (np.array): Next state.
            done (bool): Whether the episode terminated.
        """
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        """
        Sample a batch of transitions, weighted by priority.
        
        Args:
            batch_size (int): Number of transitions to sample.
            beta (float): Importance-sampling exponent to correct for bias.
            
        Returns:
            tuple: Tensors for states, actions, rewards, next_states, dones,
                   weights (for IS correction), and the indices of the samples.
        """
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:len(self.buffer)]
        probs = prios ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)
        state, action, reward, next_state, done = map(np.stack, zip(*samples))
        return (torch.FloatTensor(state).to(device),
                torch.FloatTensor(action).to(device),
                torch.FloatTensor(reward).unsqueeze(1).to(device),
                torch.FloatTensor(next_state).to(device),
                torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device),
                torch.FloatTensor(weights).unsqueeze(1).to(device),
                indices)

    def update_priorities(self, indices, priorities):
        """
        Update the priorities for the given indices.
        
        Args:
            indices (list or np.array): Indices of transitions.
            priorities (list or np.array): New priority values.
        """
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio

    def __len__(self):
        """Return the current number of stored transitions."""
        return len(self.buffer)

class ReplayBuffer:
    """
    A uniform (non-prioritized) replay buffer for storing transitions.

    Attributes:
        capacity (int): Maximum number of transitions.
        buffer (list): List of transitions.
        position (int): Current index for insertion.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer.
        
        Args:
            state: The current state.
            action: The action taken.
            reward: The reward received.
            next_state: The next state.
            done: Whether the episode terminated.
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Uniformly sample a batch of transitions.
        
        Args:
            batch_size (int): Number of transitions to sample.
            
        Returns:
            tuple: Tensors for states, actions, rewards, next_states, dones.
        """
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return (torch.FloatTensor(state).to(device),
                torch.FloatTensor(action).to(device),
                torch.FloatTensor(reward).unsqueeze(1).to(device),
                torch.FloatTensor(next_state).to(device),
                torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device))

    def __len__(self):
        """Return the number of stored transitions."""
        return len(self.buffer)

class GaussianPolicy(nn.Module):
    """
    A Gaussian policy network that produces a distribution over actions.
    
    The network uses two hidden layers with layer normalization for stability.
    
    Attributes:
        fc1, fc2: Fully connected layers.
        ln1, ln2: Layer normalization layers.
        mean_layer: Final layer producing the mean of the Gaussian.
        log_std_layer: Final layer producing the log standard deviation.
    """
    def __init__(self, num_inputs, num_actions, hidden_size, log_std_min=-20, log_std_max=2):
        super(GaussianPolicy, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.mean_layer = nn.Linear(hidden_size, num_actions)
        self.log_std_layer = nn.Linear(hidden_size, num_actions)
    
    def forward(self, state):
        """
        Perform a forward pass through the network.
        
        Args:
            state (Tensor): The input state.
            
        Returns:
            tuple: The mean and log standard deviation.
        """
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state):
        """
        Sample an action from the policy using the reparameterization trick.
        
        Args:
            state (Tensor): The input state.
            
        Returns:
            tuple: Sampled action, its log probability, and the squashed mean.
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t
        log_prob = normal.log_prob(x_t)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        mean = torch.tanh(mean)
        return action, log_prob, mean

class QNetwork(nn.Module):
    """
    A critic network (Q-network) that estimates the Q-value for a state-action pair.
    
    The network concatenates the state and action, passes them through two hidden layers 
    with layer normalization, and outputs a single Q-value.
    """
    def __init__(self, num_inputs, num_actions, hidden_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
    
    def forward(self, state, action):
        """
        Perform a forward pass through the network.
        
        Args:
            state (Tensor): The input state.
            action (Tensor): The action.
            
        Returns:
            Tensor: The estimated Q-value.
        """
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        return self.fc3(x)

class SACAgent:
    """
    Soft Actor-Critic (SAC) Agent.
    
    This agent encapsulates the actor (policy network), two critics (Q-networks),
    target critics, and automatic entropy tuning (alpha). Optimizers are set
    externally during training.
    
    Attributes:
        policy (GaussianPolicy): The actor network.
        q1, q2 (QNetwork): The critic networks.
        q1_target, q2_target (QNetwork): Target critic networks.
        log_alpha (Tensor): Logarithm of the entropy temperature.
        target_entropy (float): Desired target entropy.
    """
    def __init__(self, state_dim, action_dim, hidden_size, init_alpha=0.2):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.policy = GaussianPolicy(state_dim, action_dim, hidden_size).to(device)
        self.policy_optimizer = None  # Set externally during training

        self.q1 = QNetwork(state_dim, action_dim, hidden_size).to(device)
        self.q2 = QNetwork(state_dim, action_dim, hidden_size).to(device)
        self.q1_optimizer = None
        self.q2_optimizer = None

        self.q1_target = QNetwork(state_dim, action_dim, hidden_size).to(device)
        self.q2_target = QNetwork(state_dim, action_dim, hidden_size).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.target_entropy = -action_dim
        self.log_alpha = torch.tensor(np.log(init_alpha), requires_grad=True, device=device)
        self.alpha_optimizer = None  # Set externally during training

    @property
    def alpha(self):
        """
        Return the current temperature (alpha) value computed as exp(log_alpha).
        
        Returns:
            Tensor: The current alpha value.
        """
        return self.log_alpha.exp()

    def select_action(self, state, evaluate=False):
        """
        Select an action for a given state.
        
        Args:
            state (np.array): The current state.
            evaluate (bool): If True, use deterministic action selection.
            
        Returns:
            np.array: The selected action.
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action, _, _ = self.policy.sample(state)
        return action.cpu().numpy()[0]
