import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Standard tuple: (state, action_idx, reward, next_state, done_flag)"""
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
        
    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, state_dim: int):
        super(QNetwork, self).__init__()
        # State describes the Request (Src, Dst) + Network Utilization + Target Path length + Target SNR
        # Network outputs precisely the Expected Q-Value for taking this Action (allocating this exact route/slot)
        # Because Action space candidates vary dynamically, we feed (State + Action_Features) -> Q value
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 1) # Single Q-value score
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)


class DQNAgent:
    def __init__(
        self, 
        state_dim: int, 
        learning_rate: float = 1e-3, 
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 2000,
        buffer_capacity: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 50
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0
        
        self.policy_net = QNetwork(state_dim).to(self.device)
        self.target_net = QNetwork(state_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(buffer_capacity)
        
    def select_action(self, condensed_candidate_states: np.ndarray) -> int:
        """
        Takes an array of shape (num_candidates, state_dim).
        Returns the index of the selected candidate mapping to a physical Path/Slot assignment.
        """
        self.steps_done += 1
        
        # Anneal epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon -= (1.0 - self.epsilon_end) / self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_end)
            
        # Epsilon-Greedy Exploration
        if random.random() < self.epsilon:
            return random.randrange(len(condensed_candidate_states))
            
        # Exploitation via Q-Network
        with torch.no_grad():
            states_tensor = torch.FloatTensor(condensed_candidate_states).to(self.device)
            q_values = self.policy_net(states_tensor)
            return int(q_values.argmax().item())
            
    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return 0.0 # Loss is 0
            
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        state_batch = torch.FloatTensor(states).to(self.device)
        reward_batch = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(next_states).to(self.device)
        done_batch = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Current Q value (for the actual action taken) is specifically the forward pass
        # Note: Since the agent maps (State+Action) -> Q-value directly, state_batch is already (State+Action)
        current_q_values = self.policy_net(state_batch)
        
        # Compute Next Q-Value iteratively (or max of candidates)
        # Because action space changes size per step realistically, tracking NEXT state transitions requires 
        # either passing the optimal 'next_state' max directly from the environment or averaging.
        # For this optical sim, we operate as purely episodic sequential states with assumed 0 Future if action is taken,
        # but to keep it mathematically pure we map the next optimal action if not done.
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch)
            max_next_q = next_q_values.max(1)[0].unsqueeze(1)
            target_q_values = reward_batch + (self.gamma * max_next_q * (1 - done_batch))
            
        loss = F.mse_loss(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping prevents explosion
        for param in self.policy_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        # Target Sync
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        return loss.item()
