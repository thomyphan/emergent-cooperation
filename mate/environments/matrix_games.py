from mate.environments.environment import Environment
import numpy

"""
 Iterated Two-Player Matrix Game with some predefined payoff matrices.
"""
class MatrixGameEnvironment(Environment):

    def __init__(self, params):
        params["domain_value_labels"] = ["time_steps", "desired_equilibria", "gifting_1", "gifting_2"]
        super(MatrixGameEnvironment, self).__init__(params)
        self.nr_agents = 2
        self.payoff_matrices = params["payoff_matrices"]
        self.domain_value_function = params["domain_value_function"]
        self.last_performed_action = None
        assert len(self.payoff_matrices) == self.nr_agents,\
            "2-player matrix games require 2 matrices, but got {}".format(len(self.payoff_matrices))

    def perform_step(self, joint_action):
        self.last_performed_action = joint_action
        rewards, infos = super(MatrixGameEnvironment, self).perform_step(joint_action)
        assert not self.is_done(), "Episode terminated at time step {}. Please, reset before calling 'step'."\
            .format(self.time_step)
        action_1, normed1 = self.norm_action(joint_action[0])
        action_2, normed2 = self.norm_action(joint_action[1])
        for i, payoff_matrix in\
            zip(range(self.nr_agents), self.payoff_matrices):
            rewards[i] = payoff_matrix[action_1][action_2]
        if normed1 or normed2:
            if normed1:
                self.domain_counts[2] += 1
            if normed2:
                self.domain_counts[3] += 1
        elif self.domain_value_function([action_1, action_2]):
            self.domain_counts[1] += 1
        return rewards, infos

    def get_metric_indices(self, metric):
        if metric == "cooperation_rate":
            return self.get_index("desired_equilibria"), self.get_index("time_steps")
        return None, self.get_index("time_steps")

    def domain_value_debugging_indices(self):
        return self.get_index("desired_equilibria"), self.get_index("time_steps")

    def norm_action(self, action):
        if action >= 0 and action < self.nr_actions:
            return action, False
        else:
            return numpy.random.randint(0, self.nr_actions), True

    def local_observation(self, agent_id):
        a1, a2 = self.last_joint_action
        obs = numpy.zeros(self.observation_dim)
        if self.last_performed_action is not None:
            if agent_id == 0:
                obs[a2] = 1
            else:
                obs[a1] = 1
        return obs

    def reset(self):
        self.last_performed_action = None
        return super(MatrixGameEnvironment, self).reset()

def make(params):
    domain_name = params["domain_name"]
    params["gamma"] = 0.95
    params["time_limit"] = 150
    params["nr_agents"] = 2
    params["nr_actions"] = 2
    params["history_length"] = 1
    if domain_name in ["Matrix-IPD", "Matrix-PrisonersDilemma"]:
        params["payoff_matrices"] = [
            numpy.array([
                [-1, -3],
                [ 0, -2]
            ]),
            numpy.array([
                [-1,  0],
                [-3, -2]
            ])
        ]
        params["domain_value_function"] = lambda x: tuple(x) == (0,0)
    if domain_name in ["Matrix-ISH", "Matrix-StagHunt"]:
        params["payoff_matrices"] = [
            numpy.array([
                [ 4, 1],
                [ 3, 2]
            ]),
            numpy.array([
                [ 4, 3],
                [ 1, 2]
            ])
        ]
        params["domain_value_function"] = lambda x: tuple(x) in [(0,0), (1,1)]
    if domain_name in ["Matrix-ICG", "Matrix-Coordination"]:
        params["payoff_matrices"] = [
            numpy.array([
                [ 1, 0],
                [ 0, 1]
            ]),
            numpy.array([
                [ 1, 0],
                [ 0, 1]
            ])
        ]
        params["domain_value_function"] = lambda x: tuple(x) in [(0,0), (1,1)]
    if domain_name in ["Matrix-IMP", "Matrix-MatchingPennies"]:
        params["payoff_matrices"] = [
            numpy.array([
                [ 1, -1],
                [-1,  1]
            ]),
            numpy.array([
                [-1, 1],
                [ 1,-1]
            ])
        ]
        params["domain_value_function"] = lambda x: tuple(x) in [(0,0), (1,1)]
    if domain_name in ["Matrix-IC", "Matrix-Chicken"]:
        params["payoff_matrices"] = [
            numpy.array([
                [ 0, -1],
                [ 1, -1000]
            ]),
            numpy.array([
                [ 0,  1],
                [-1, -1000]
            ])
        ]
        params["domain_value_function"] = lambda x: tuple(x) in [(0,0), (1,0), (0,1)]
    assert "payoff_matrices" in params and "domain_value_function" in params,\
        "Unknown matrix game '{}'".format(domain_name)
    params["observation_dim"] = int(params["nr_actions"])
    return MatrixGameEnvironment(params)
