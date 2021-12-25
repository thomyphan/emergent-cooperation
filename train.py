from settings import params
import mate.domains as domains
import mate.algorithms as algorithms
import mate.experiments as experiments
import mate.data as data
import sys

params["domain_name"] = sys.argv[1]
params["algorithm_name"] = sys.argv[2]

env = domains.make(params)
env.reset()
controller = algorithms.make(params)

params["directory"] = params["output_folder"] + "/" + params["data_prefix_pattern"].\
    format(
        params["nr_agents"],\
        params["domain_name"],\
        params["algorithm_name"])
params["directory"] = data.mkdir_with_timestap(params["directory"])

experiments.run_training(env, controller, params)
