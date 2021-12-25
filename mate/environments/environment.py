import numpy

class Environment:

    def __init__(self, params) -> None:
        self.domain_value_labels = params["domain_value_labels"]
        self.observation_dim = params["observation_dim"]
        self.nr_agents = params["nr_agents"]
        self.nr_actions = params["nr_actions"]
        self.time_limit = params["time_limit"]
        self.gamma = params["gamma"]
        self.time_step = 0
        self.sent_gifts = numpy.zeros(self.nr_agents)
        self.discounted_returns = numpy.zeros(self.nr_agents)
        self.undiscounted_returns = numpy.zeros(self.nr_agents)
        self.domain_counts = numpy.zeros(len(self.domain_value_labels))
        self.last_joint_action = -numpy.ones(self.nr_agents, dtype=numpy.int)

    """
     Performs the joint action in order to change the environment.
     Returns the reward for each agent in a list sorted by agent ID.
    """
    def perform_step(self, joint_action):
        assert not self.is_done(), "Episode terminated at time step {}. Please, reset before calling 'step'."\
            .format(self.time_step)
        return numpy.zeros(self.nr_agents), {}

    """
     Indicates if an episode is done and the environments needs
     to be reset.
    """
    def is_done(self):
        return self.time_step >= self.time_limit

    def action_as_vector(self, action):
        if action < self.nr_actions:
            vector = numpy.zeros(self.nr_actions)
            if action >= 0:
                vector[action] = 1
        else:
            vector = numpy.ones(self.nr_actions)
        return vector

    """
     Performs a joint action to change the state of the environment.
     Returns the joint observation, the joint reward, a done flag,
     and other optional information (e.g., logged data).
     Note: The joint action must be a list ordered according to the agent ID!.
    """
    def step(self, joint_action):
        assert len(joint_action) == self.nr_agents, "Length of 'joint_action' is {}, expected {}"\
            .format(len(joint_action), self.nr_agents)
        assert not self.is_done(), "Episode terminated at time step {}. Please, reset before calling 'step'."\
            .format(self.time_step)
        rewards, infos = self.perform_step(joint_action)
        for i, a in enumerate(joint_action):
            self.last_joint_action[i] = a
            if a >= self.nr_actions:
                self.sent_gifts[i] += 1
        assert len(rewards) == self.nr_agents, "Length of 'rewards' is {}, expected {}"\
            .format(len(rewards), self.nr_agents)
        observations = self.joint_observation()
        assert len(observations) == self.nr_agents, "Length of 'observations' is {}, expected {}"\
            .format(len(observations), self.nr_agents)
        self.time_step += 1
        self.domain_counts[0] += 1.0
        self.undiscounted_returns += rewards
        self.discounted_returns += (self.gamma**self.time_step)*rewards
        if "neighbor_agents" not in infos:
            infos["neighbor_agents"] = [[j for j in range(self.nr_agents) if j != i] for i in range(self.nr_agents)]
        return observations, rewards, self.is_done(), infos

    def get_index(self, label):
        return self.domain_value_labels.index(label)

    """
     The local observation for a specific agent. Only visible for
     the corresponding agent and private to others.
    """
    def local_observation(self, agent_id):
        pass

    """
     Returns the observations of all agents in a listed sorted by agent ids.
    """
    def joint_observation(self):
        return [numpy.array(self.local_observation(i)).reshape(self.observation_dim) for i in range(self.nr_agents)]

    """
     Returns a high-level value which is domain-specific.
    """
    def domain_values(self):
        return self.domain_counts

    def domain_value_debugging_indices(self):
        return 0,1

    """
     Re-Setup of the environment for a new episode.
    """
    def reset(self):
        self.time_step = 0
        self.discounted_returns[:] = 0
        self.undiscounted_returns[:] = 0
        self.last_joint_action[:] = -1
        self.domain_counts[:] = 0
        self.sent_gifts[:] = 0
        return self.joint_observation()

