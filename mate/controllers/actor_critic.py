from mate.controllers.controller import Controller
from mate.utils import assertEquals
from torch.distributions import Categorical
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy

def preprocessing_module(nr_input_features, nr_hidden_units):
    return nn.Sequential(
            nn.Linear(nr_input_features, nr_hidden_units),
            nn.ELU(),
            nn.Linear(nr_hidden_units, nr_hidden_units),
            nn.ELU()
        )

class ActorNet(nn.Module):
    def __init__(self, input_dim, nr_actions, nr_hidden_units, learning_rate):
        super(ActorNet, self).__init__()
        self.nr_input_features = input_dim
        self.nr_hidden_units = nr_hidden_units
        self.fc_net = preprocessing_module(self.nr_input_features, self.nr_hidden_units)
        self.action_head = nn.Linear(self.nr_hidden_units, nr_actions)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.fc_net(x)
        return F.softmax(self.action_head(x), dim=-1)

class CriticNet(nn.Module):
    def __init__(self, input_dim, nr_hidden_units, learning_rate):
        super(CriticNet, self).__init__()
        self.nr_input_features = input_dim
        self.nr_hidden_units = nr_hidden_units
        self.fc_net = preprocessing_module(self.nr_input_features, self.nr_hidden_units)
        self.value_head = nn.Linear(self.nr_hidden_units, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.fc_net(x)
        return self.value_head(x)

class ActorCritic(Controller):

    def __init__(self, params):
        super(ActorCritic, self).__init__(params)
        self.nr_update_iterations = 1
        self.actor_nets = []
        self.critic_nets = []
        for _ in range(self.nr_agents):
            self.actor_nets.append(ActorNet(self.input_dim, self.nr_actions, params["nr_hidden_units"], self.learning_rate))
            self.critic_nets.append(CriticNet(self.input_dim, params["nr_hidden_units"], self.learning_rate))

    def sample_comm_failure(self):
        if self.current_epoch < self.failure_start_epoch:
            return False
        sample = numpy.random.rand()
        return sample < self.comm_failure_prob

    def sample_no_comm_failure(self):
        return not self.sample_comm_failure()
    
    def local_probs(self, history, agent_id):
        history = torch.tensor([history], dtype=torch.float32, device=self.device)
        return self.actor_nets[agent_id](history).detach().numpy()[0]

    def preprocess(self):
        return None

    def update_step(self):
        preprocessed_data = self.preprocess()
        for i, memory, actor_net, critic_net in\
            zip(range(self.nr_agents), self.memories, self.actor_nets, self.critic_nets):
            self.local_update(i, memory, actor_net, critic_net, preprocessed_data)
            memory.clear()
    
    def policy_loss(self, advantage, probs, action, old_probs):
        m1 = Categorical(probs)
        return -m1.log_prob(action)*advantage

    def local_update(self, agent_id, memory, actor_net, critic_net, preprocessed_data):
        training_data = memory.get_training_data()
        for _ in range(self.nr_update_iterations):
            self.update_critic(agent_id, training_data, critic_net, preprocessed_data)
            self.update_actor(agent_id, training_data, actor_net, preprocessed_data)
        return True

    def update_critic(self, agent_id, training_data, critic_net, preprocessed_data):
        histories, _, _, _, returns, _, _, _ = training_data
        values = critic_net(histories).squeeze()
        assertEquals(values.size(), returns.size())
        critic_loss = F.mse_loss(returns.detach(), values)
        critic_net.optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(critic_net.parameters(), self.clip_norm)
        critic_net.optimizer.step()

    def get_values(self, agent_id, histories):
        return self.critic_nets[agent_id](histories)

    def update_actor(self, agent_id, training_data, actor_net, preprocessed_data):
        histories, _, actions, _, returns, old_probs, _, _ = training_data
        values = self.get_values(agent_id, histories).squeeze().detach()
        action_probs = actor_net(histories)
        advantages = returns.detach() - values.detach()
        policy_losses = []
        for action, old_prob, probs, advantage in zip(actions, old_probs, action_probs, advantages):
            policy_losses.append(self.policy_loss(advantage.item(), probs, action, old_prob))
        actor_loss = torch.stack(policy_losses).sum()
        actor_net.optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(actor_net.parameters(), self.clip_norm)
        actor_net.optimizer.step()
