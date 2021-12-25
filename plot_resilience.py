from settings import params
import mate.plotting as plotting
import mate.domains as domains
import sys
import matplotlib.pyplot as plot
from os.path import join

ALGORITHM_NAMES = {
    "IAC_": "Naive Learning",
    "Gifting-ZEROSUM_": "Gifting (Zero-Sum)",
    "Gifting-BUDGET_": "Gifting (Budget)",
    "MATE-TD_": "MATE",
    "MATE-REWARD_": "MATE (reward-based)",
    "LIO_": "LIO",
    "MATE-TD-DEFECT_COMPLETE_": "MATE (defect=Complete)",
    "MATE-TD-DEFECT_REQUEST_": "MATE (defect=Request)",
    "MATE-TD-DEFECT_RESPONSE_": "MATE (defect=Response)",
    "Random_": "Random"
}

Y_LABEL = {
    "undiscounted_returns": "Efficiency (U)",
    "discounted_returns": "collective discounted return",
    "own_coin_prob": "P(own coin)",
    "peace": "Peace (P)",
    "sustainability": "Sustainability (S)"
}

LEGEND_CONFIG = {
    "Matrix-IPD":4,
    "CoinGame-4":'best',
    "Harvest-12":10
}
path = "plots"
plot.figure(figsize=(4, 3))
ax = plot.gca()
params["domain_name"] = sys.argv[1]
env = domains.make(params)
params["filter_size"] = 10
params["nr_runs"] = 20
params["directory"] = "output"
metric = "undiscounted_returns"
params["metric_index"] = None
params["norm_index"] = None
params["stats_label"] = "domain_values"
params["y_label"] = Y_LABEL["undiscounted_returns"]
params["data_length"] = 5000
if len(sys.argv) > 2:
    params["stats_label"] = "domain_values"
    metric = sys.argv[2]
if metric == "equality":
    params["stats_label"] = metric
    params["y_label"] = "Equality (E)"
elif "_returns" not in metric:
    params["metric_index"], params["norm_index"] = env.get_metric_indices(metric)
    params["norm_nr_agents"] = 1
    params["y_label"] = Y_LABEL[metric]
if len(sys.argv) > 3:
    params["norm_nr_agents"] = env.nr_agents
    params["y_label"] += " per agent"
if params["domain_name"].startswith("Matrix-"):
    params["stats_label"] = "domain_values"
    params["norm_index"] = 0
    params["y_label"] = Y_LABEL["undiscounted_returns"]
    params["norm_nr_agents"] = 1
params["x_label"] = "communication failure rate"
filename_prefix = "{}_resilience_{}".format(params["domain_name"].lower(), metric.lower())
filename = filename_prefix + ".svg"
filename_png = filename_prefix + ".png"
filename_pdf = filename_prefix + ".pdf"

data_prefix_pattern = params["data_prefix_pattern"]

params["comm_failure_rates"] = ["0", "0.1", "0.2", "0.4", "0.8"]
params["offset"] = 0.25
params["bar_width"] = 0.5

offset_factor = 1
for color, algorithm_name in [("b","MATE-TD_"), ("r", "LIO_"), ("k","IAC_"), ("darkblue", "Random_")]:
    params["data_prefix_pattern"] = data_prefix_pattern.format(
        params["nr_agents"],\
        params["domain_name"],\
        algorithm_name)
    params["label"] = ALGORITHM_NAMES[algorithm_name]
    params["color"] = color
    params["offset"] *= offset_factor
    params["algorithm_name"] = algorithm_name
    params["plot_mean"] = algorithm_name == "Random_"
    params["plot_mean_length"] = 5
    plotting.plot_run_end(params)
    offset_factor += 1
legend_position = None
if params["domain_name"] in LEGEND_CONFIG and params["y_label"] == Y_LABEL["undiscounted_returns"]:
    legend_position = LEGEND_CONFIG[params["domain_name"]]
    legend = plot.legend(loc=legend_position)
plot.tight_layout()
ax.grid(which='both', linestyle='--')
plot.savefig(join(path, filename), bbox_inches='tight')
plot.savefig(join(path, filename_png), bbox_inches='tight')
plot.savefig(join(path, filename_pdf), bbox_inches='tight')