from mate.utils import get_param_or_default
from mate.environments.environment import Environment
import numpy
import random
import copy

NOOP = 0
MOVE_NORTH = 1
MOVE_SOUTH = 2
MOVE_WEST = 3
MOVE_EAST = 4
TAG_NORTH = 5
TAG_SOUTH = 6
TAG_WEST = 7
TAG_EAST = 8

HARVEST_ACTIONS = [NOOP, MOVE_NORTH, MOVE_SOUTH, MOVE_WEST, MOVE_EAST, TAG_NORTH, TAG_SOUTH, TAG_WEST, TAG_EAST]
TAGGING_ACTIONS = [TAG_NORTH, TAG_SOUTH, TAG_WEST, TAG_EAST]

class MovableAgent:

    def __init__(self, agent_id, width, height, view_range):
        self.agent_id = agent_id
        self.position = None
        self.width = width
        self.height = height
        self.view_range = view_range
        self.tagged_timeout = 0

    def move(self, action):
        if not self.is_tagged():  
            x, y = self.position
            if action == MOVE_NORTH and y + 1 < self.height:
                self.position = (x, y + 1)
            if action == MOVE_SOUTH and y - 1 >= 0:
                self.position = (x, y - 1)
            if action == MOVE_EAST and x + 1 < self.width:
                self.position = (x + 1, y)
            if action == MOVE_WEST and x - 1 >= 0:
                self.position = (x - 1, y)

    def reset(self, position):
        self.position = position
        self.tagged_timeout = 0

    def is_tagged(self):
        return self.tagged_timeout > 0

    def reduce_tagged_timeout(self):
        self.tagged_timeout = max(0, self.tagged_timeout - 1)

    def tag(self, tagged_timeout):
        assert tagged_timeout > 0, "Expected positive tagging timeout but got {}".format(tagged_timeout)
        if not self.is_tagged(): # Tagged agents temporarily "disappear" thus are not taggable.
            self.tagged_timeout = tagged_timeout

    def visible_positions(self):
        x0, y0 = self.position
        x_center = int(self.view_range/2)
        y_center = int(self.view_range/2)
        positions = [(x,y) for x in range(-x_center+x0, x_center+1+x0)\
            for y in range(-y_center+y0, y_center+1+y0)]
        return positions

    def relative_position_to(self, other_position):
        dx = other_position[0] - self.position[0]
        dy = other_position[1] - self.position[1]
        return dx, dy

    def distance_to(self, other_position):
        dx, dy = self.relative_position_to(other_position)
        return max(abs(dx), abs(dy))

class HarvestEnvironment(Environment):

    def __init__(self, params):
        params["domain_value_labels"] = ["time_steps", "nr_tags", "untagged_agents", "nr_collected_apples"]
        super(HarvestEnvironment, self).__init__(params)
        self.width = params["width"]
        self.height = params["height"]
        self.view_range = params["view_range"]
        self.time_step_penalty = get_param_or_default(params, "time_step_penalty", 0.01)
        self.observation_shape = (4, self.view_range, self.view_range)
        self.agents = [MovableAgent(i, self.width, self.height, self.view_range) for i in range(self.nr_agents)]
        self.tagged_timeout = get_param_or_default(params, "tagged_timeout", 25)
        self.original_apple_map = params["apple_map"]
        self.apple_map = copy.deepcopy(self.original_apple_map)
        self.agent_initial_positions = params["agent_initial_positions"]

    def perform_step(self, joint_action):
        rewards, infos = super(HarvestEnvironment, self).perform_step(joint_action)
        assert not self.is_done(), "Episode terminated at time step {}. Please, reset before calling 'step'."\
            .format(self.time_step)
        neighbor_agents = [[] for _ in range(self.nr_agents)]
        agent_actions = list(zip(self.agents, joint_action))
        random.shuffle(agent_actions)
        neighborhood_radius = int(self.view_range/2)
        for agent, action in agent_actions:
            neighborhood = []
            if not agent.is_tagged():
                rewards[agent.agent_id] -= self.time_step_penalty
                neighborhood = [other for other in self.agents if\
                    (other.agent_id != agent.agent_id) and \
                    (agent.distance_to(other.position) <= neighborhood_radius)]
                if action in TAGGING_ACTIONS:
                    self.domain_counts[1] += 1
                    rel_positions = [agent.relative_position_to(other.position) for other in neighborhood]
                    if action == TAG_NORTH:
                        victims = [neighborhood[i] for i, rel_pos in enumerate(rel_positions)\
                            if abs(rel_pos[0]) <= 2 and rel_pos[1] >= 0]
                    if action == TAG_SOUTH:
                        victims = [neighborhood[i] for i, rel_pos in enumerate(rel_positions)\
                            if abs(rel_pos[0]) <= 2 and rel_pos[1] <= 0]
                    if action == TAG_EAST:
                        victims = [neighborhood[i] for i, rel_pos in enumerate(rel_positions)\
                            if rel_pos[0] >= 0 and abs(rel_pos[1]) <= 2]
                    if action == TAG_WEST:
                        victims = [neighborhood[i] for i, rel_pos in enumerate(rel_positions)\
                            if rel_pos[0] <= 0 and abs(rel_pos[1]) <= 2]
                    for victim in victims:
                        victim.tag(self.tagged_timeout)
                else:
                    agent.move(action)
                    x, y = agent.position
                    if self.apple_map[x][y] > 0:
                        rewards[agent.agent_id] += 1
                        self.apple_map[x][y] = 0
                        self.domain_counts[3] += 1
            neighbor_agents[agent.agent_id] = [a.agent_id for a in neighborhood]
            agent.reduce_tagged_timeout()
        self.apple_regrowth()
        infos["neighbor_agents"] = neighbor_agents
        self.domain_counts[2] += len([a for a in self.agents if not a.is_tagged()])
        return rewards, infos

    def get_metric_indices(self, metric):
        if metric == "peace":
            return self.get_index("untagged_agents"), self.get_index("time_steps")
        if metric == "sustainability":
            return self.get_index("nr_collected_apples"), self.get_index("time_steps")
        return None, self.get_index("time_steps")

    def domain_value_debugging_indices(self):
        return self.get_index("untagged_agents"), self.get_index("time_steps")

    def is_done(self):
        done = super(HarvestEnvironment, self).is_done()
        done = numpy.sum(self.apple_map) <= 0 or done
        return done

    def apple_regrowth(self):
        for x in range(self.width):
            for y in range(self.height):
                if self.apple_map[x][y] == 0:
                    nr_neighbor_apples = 0
                    for dx in range(-2, 3):
                        for dy in range(-2, 3):
                            x1 = x + dx
                            y1 = y + dy
                            distance = abs(dx) + abs(dy)
                            within_bounds = x1 >= 0 and y1 >= 0 and x1 < self.width and y1 < self.height 
                            if within_bounds and distance <= 2 and self.apple_map[x1][y1] > 0:
                                nr_neighbor_apples += 1
                    P = 0
                    if nr_neighbor_apples in [1,2]:
                        P = 0.01
                    if nr_neighbor_apples in [3,4]:
                        P = 0.05
                    if nr_neighbor_apples > 4:
                        P = 0.1
                    new_apple = numpy.random.choice([0,1], p=[1-P, P])
                    self.apple_map[x][y] = new_apple

    def local_observation(self, agent_id):
        observation = numpy.zeros(self.observation_shape)
        focus_agent = self.agents[agent_id]
        if not focus_agent.is_tagged():
            x_center = int(self.view_range/2)
            y_center = int(self.view_range/2)
            observation[0][x_center][y_center] = 1
            visible_positions = focus_agent.visible_positions()
            for x, y in visible_positions:
                dx, dy = focus_agent.relative_position_to((x,y))
                if x < 0 or y < 0 or x >= self.width or y >= self.height:
                    observation[1][x_center+dx][y_center+dy] += 1
                else:
                    observation[2][x_center+dx][y_center+dy] = self.apple_map[x][y]
            for agent in self.agents:
                if agent.agent_id != agent_id and agent.position in visible_positions and not agent.is_tagged():
                    dx, dy = focus_agent.relative_position_to(agent.position)
                    observation[3][x_center+dx][y_center+dy] += 1
        return observation

    def reset(self):
        self.tag_count = 0
        self.apple_map = copy.deepcopy(self.original_apple_map)
        positions = random.sample(self.agent_initial_positions, k=self.nr_agents)
        for pos, agent in zip(positions, self.agents):
            agent.reset(pos)
        return super(HarvestEnvironment, self).reset()

HARVEST_LAYOUTS = {
    "Harvest-6": (6,
            """
             . . . . . . . . . . . . . . . . . . . . . . . . .
             . . . . . A . . . . A . . . . A . . . . A . . . .
             . . . . A A A . . A A A . . A A A . . A A A . . .
             . . . . . A . . . . A . . . . A . . . . A . . . .
             . . . . . . . . . . . . . . . . . . . . . . . . .
             . . A . . . . A . . . . A . . . . A . . . . A . .
             . A A A . . A A A . . A A A . . A A A . . A A A .
             . . A . . . . A . . . . A . . . . A . . . . A . .
             . . . . . . . . . . . . . . . . . . . . . . . . .
            """),
    "Harvest-12": (12,
            """
             . . . . . . . . . . . . . . . . . . . . . . . . .
             . . . . . A . . . . A . . . . A . . . . A . . . .
             . . . . A A A . . A A A . . A A A . . A A A . . .
             . . . . . A . . . . A . . . . A . . . . A . . . .
             . . . . . . . . . . . . . . . . . . . . . . . . .
             . . A . . . . A . . . . A . . . . A . . . . A . .
             . A A A . . A A A . . A A A . . A A A . . A A A .
             . . A . . . . A . . . . A . . . . A . . . . A . .
             . . . . . . . . . . . . . . . . . . . . . . . . .
            """)
}

def make(params):
    domain_name = params["domain_name"]
    params["gamma"] = 0.99
    params["nr_agents"], raw_layout = HARVEST_LAYOUTS[domain_name]
    params["time_limit"] = 250
    params["nr_actions"] = len(HARVEST_ACTIONS)
    params["history_length"] = 1
    params["view_range"] = 7
    params["height"] = 0
    params["width"] = 0
    params["agent_initial_positions"] = []
    params["observation_dim"] = int(4*params["view_range"]*params["view_range"])
    for _,line in enumerate(raw_layout.splitlines()):
        splitted_line = line.strip().split()
        if splitted_line:
            for x,cell in enumerate(splitted_line):
                position = (x,params["height"])
                if cell == '.':
                    params["agent_initial_positions"].append(position)
                params["width"] = x
            params["height"] += 1
            params["width"] += 1
    params["apple_map"] = numpy.zeros((params["width"], params["height"]))
    y = 0
    for _,line in enumerate(raw_layout.splitlines()):
        splitted_line = line.strip().split()
        if splitted_line:
            for x,cell in enumerate(splitted_line):
                position = (x,y)
                if cell == 'A':
                    params["apple_map"][x][y] = 1
            y += 1
    return HarvestEnvironment(params)