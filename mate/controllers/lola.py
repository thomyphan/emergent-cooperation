from mate.utils import get_param_or_default
from mate.controllers.actor_critic import ActorCritic
import torch
from torch.distributions import Categorical
from torch.autograd import grad

class LOLA(ActorCritic):

    def __init__(self, params):
        super(LOLA, self).__init__(params)
        self.second_order_learning_rate = get_param_or_default(params, "second_order_learning_rate", 1)

    def preprocess(self):
        all_gradients = []
        for i in range(self.nr_agents):
            histories_i, _, actions_i, _, Ri, old_probs_i, _, _ = self.memories[i].get_training_data()
            probs_i = self.actor_nets[i](histories_i)
            Ri -= self.critic_nets[i](histories_i).squeeze().detach()
            gradients = []
            for weights in self.actor_nets[i].parameters():
                gradients.append(torch.zeros_like(weights))
            for j in range(self.nr_agents):
                if j != i:
                    histories_j, _, actions_j, _, Rj, old_probs_j, _, _ = self.memories[j].get_training_data()
                    probs_j = self.actor_nets[j](histories_j)
                    Rj -= self.critic_nets[j](histories_j).squeeze().detach()
                    losses_V_i = []
                    losses_V_j = []
                    for R_i, R_j, p_i, op_i, a_i, p_j, op_j, a_j in zip(Ri, Rj, probs_i, old_probs_i, actions_i, probs_j, old_probs_j, actions_j): 
                        log_prob_i = Categorical(p_i).log_prob(a_i)
                        log_prob_j = Categorical(p_j).log_prob(a_j)
                        losses_V_i.append(log_prob_i*R_i*log_prob_j)
                        losses_V_j.append(log_prob_i*R_j*log_prob_j)
                    total_loss_V_i = torch.stack(losses_V_i).sum()
                    total_loss_V_j = torch.stack(losses_V_j).sum()

                    for grads, param_i, param_j in zip(gradients, self.actor_nets[i].parameters(), self.actor_nets[j].parameters()):
                        D1Ri = grad(total_loss_V_i, (param_i, param_j), create_graph=True)
                        D1Rj = grad(total_loss_V_j, (param_i, param_j), create_graph=True)
                        D2Rj = [grad(g, param_i, create_graph=True)[0].view(-1) for g in D1Rj[1].view(-1)]
                        D2Rj = torch.stack([D2Rj[x] for x,_ in enumerate(param_i.view(-1))])
                        naive_grad = D1Ri[0].view(-1)
                        second_order_grad = torch.matmul(D2Rj, D1Ri[1].view(-1))
                        lola_grad = naive_grad + self.second_order_learning_rate*second_order_grad
                        grads += lola_grad.view_as(grads)

            all_gradients.append(gradients)
        return all_gradients

    def update_actor(self, agent_id, training_data, actor_net, critic_net, preprocessed_data):
        actor_net.optimizer.zero_grad()
        for params, lola_grad in zip(actor_net.parameters(), preprocessed_data[agent_id]):
            params.grad = lola_grad
        actor_net.optimizer.step()
