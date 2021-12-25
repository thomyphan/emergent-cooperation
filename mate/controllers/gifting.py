from mate.utils import get_param_or_default, assertContains
from mate.controllers.actor_critic import ActorCritic
import torch
import numpy

ZERO_SUM_MODE = "zero_sum"
BUDGET_MODE = "budget"

GIFTING_MODES = [ZERO_SUM_MODE, BUDGET_MODE]

class Gifting(ActorCritic):

    def __init__(self, params):
        params["nr_actions"] += 1
        self.gifting_action = params["nr_actions"] - 1
        super(Gifting, self).__init__(params)
        self.gift_reward = get_param_or_default(params, "gift_reward", 1.0)
        self.gifting_mode = get_param_or_default(params, "gifting_mode", BUDGET_MODE)
        self.gifting_budgets = numpy.zeros(self.nr_agents)
        assertContains(GIFTING_MODES, self.gifting_mode)
        self.last_probs = numpy.ones((self.nr_agents, self.nr_actions))

    def local_probs(self, history, agent_id):
        history = torch.tensor([history], dtype=torch.float32, device=self.device)
        probs = self.actor_nets[agent_id](history).detach().numpy()[0]
        if self.gifting_mode == BUDGET_MODE and self.gifting_budgets[agent_id] < self.gift_reward:
            probs[self.gifting_action] = 0
            probs /= numpy.sum(probs)
            self.last_probs[agent_id] = probs
        return probs

    def prepare_transition(self, joint_histories, joint_action, rewards, next_joint_histories, done, info):
        transition = super(Gifting, self).prepare_transition(joint_histories, joint_action, rewards, next_joint_histories, done, info)
        receive_enabled = [self.sample_no_comm_failure() for _ in range(self.nr_agents)]
        for i, action, reward in zip(range(self.nr_agents), joint_action, rewards):
            if reward > 0:
                self.gifting_budgets[i] += reward
            if action == self.gifting_action:
                neighborhood = info["neighbor_agents"][i]
                nr_neighbors = len(neighborhood)
                gifting_enabled = self.sample_no_comm_failure()
                if gifting_enabled and nr_neighbors > 0: # Gifting only possible when receivers are around
                    if self.gifting_mode == ZERO_SUM_MODE:
                        transition["rewards"][i] -= self.gift_reward
                        for j in neighborhood:
                            assert i != j
                            if receive_enabled[j]:
                                transition["rewards"][j] += self.gift_reward/nr_neighbors
                                transition["request_messages_sent"] += 1
                    if self.gifting_mode == BUDGET_MODE:
                        assert self.gifting_budgets[i] >= self.gift_reward, "budget is {} and action prob was {}".format(self.gifting_budgets[i], self.last_probs[i])
                        self.gifting_budgets[i] -= self.gift_reward
                        for j in neighborhood:
                            assert i != j
                            if receive_enabled[j]:
                                transition["rewards"][j] += self.gift_reward/nr_neighbors
                                transition["request_messages_sent"] += 1
        if done:
            self.gifting_budgets[:] = 0
        return transition
