import torch
import numpy

"""
 Experience memory for each agent.
"""
class ExperienceMemory:
    def __init__(self, params, agent_id, device):
        self.device = device
        self.agent_id = agent_id
        self.episode_buffer = []
        self.nr_episodes = params["episodes_per_epoch"]
        self.episode_count = 0
        self.episode_time_limit = params["time_limit"]
        self.gamma = params["gamma"]
        self.histories = []
        self.next_histories = []
        self.rewards = []
        self.returns = []
        self.extrinsic_returns = []
        self.incentive_rewards = []
        self.actions = []
        self.old_probs = []
        self.dones = []
        self.reward_actions = []
        self.joint_actions = []
        self.abs_incentive_cost = torch.zeros(1, dtype=torch.float32, device=self.device)
        self.eps = numpy.finfo(numpy.float32).eps.item()

    def save(self, new_transition):
        self.episode_buffer.append(new_transition)
        assert len(self.episode_buffer) <= self.episode_time_limit
        if new_transition["done"]:
            self.episode_count += 1
            return_value = 0
            extrinsic_return_value = 0
            local_abs_incentive_cost = torch.zeros(1, dtype=torch.float32, device=self.device)
            for transition in reversed(self.episode_buffer):
                return_value = transition["rewards"][self.agent_id] + self.gamma*return_value
                extrinsic_return_value = transition["extrinsic_rewards"][self.agent_id] + self.gamma*extrinsic_return_value
                incentive_reward = transition["incentive_rewards"][self.agent_id]
                local_abs_incentive_cost = incentive_reward.exp().abs().sum() + self.gamma*local_abs_incentive_cost
                self.returns.append(return_value)
                self.incentive_rewards.append(incentive_reward)
                self.extrinsic_returns.append(extrinsic_return_value)
                self.histories.append(transition["joint_histories"][self.agent_id])
                self.next_histories.append(transition["next_joint_histories"][self.agent_id])
                self.rewards.append(transition["rewards"][self.agent_id])
                self.actions.append(transition["joint_action"][self.agent_id])
                self.old_probs.append(transition["joint_old_probs"][self.agent_id])
                self.dones.append(transition["done"])
                self.joint_actions.append(transition["joint_action"])
                if "reward_actions" in transition:
                    self.reward_actions.append(transition["reward_actions"])
            self.episode_buffer.clear()
            self.abs_incentive_cost += local_abs_incentive_cost
    
    def get_training_data(self):
        return torch.tensor(numpy.array(self.histories), dtype=torch.float32, device=self.device),\
            torch.tensor(numpy.array(self.next_histories), dtype=torch.float32, device=self.device),\
            torch.tensor(numpy.array(self.actions), dtype=torch.long, device=self.device),\
            torch.tensor(numpy.array(self.rewards), dtype=torch.float32, device=self.device),\
            torch.tensor(numpy.array(self.returns), dtype=torch.float32, device=self.device),\
            torch.tensor(numpy.array(self.old_probs), dtype=torch.float32, device=self.device),\
            torch.tensor(numpy.array(self.dones), dtype=torch.float32, device=self.device),\
            torch.tensor(numpy.array(self.reward_actions), dtype=torch.float32, device=self.device)

    def get_extrinsic_returns(self):
        return torch.tensor(self.extrinsic_returns, dtype=torch.float32, device=self.device)

    def get_joint_actions(self):
        return self.joint_actions

    def get_incentive_rewards_for(self, j):
        return torch.stack(self.incentive_rewards)[:,j].view(-1)

    def is_full(self):
        return self.episode_count >= self.nr_episodes

    def clear(self):
        self.episode_count = 0
        self.abs_incentive_cost = torch.zeros(1, dtype=torch.float32, device=self.device)
        self.histories.clear()
        self.next_histories.clear()
        self.rewards.clear()
        self.returns.clear()
        self.extrinsic_returns.clear()
        self.incentive_rewards.clear()
        self.actions.clear()
        self.old_probs.clear()
        self.dones.clear()
        self.reward_actions.clear()
        self.episode_buffer.clear()
        self.joint_actions.clear()
    