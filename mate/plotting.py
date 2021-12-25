from scipy import stats
from mate.utils import get_param_or_default, assertEquals
from mate.data import list_directories, list_files, load_json
import matplotlib.pyplot as plot
import numpy

def bootstrap(data, n_boot=10000, ci=95):
    boot_dist = []
    for _ in range(int(n_boot)):
        resampler = numpy.random.randint(0, data.shape[0], data.shape[0])
        sample = data.take(resampler, axis=0)
        boot_dist.append(numpy.mean(sample, axis=0))
    b = numpy.array(boot_dist)
    s1 = numpy.apply_along_axis(stats.scoreatpercentile, 0, b, 50.-ci/2.)
    s2 = numpy.apply_along_axis(stats.scoreatpercentile, 0, b, 50.+ci/2.)
    return (s1,s2)

def tsplot(data, params, alpha=0.12, **kw):
    data = numpy.array(data)
    default_x = list(range(data.shape[1]))
    x = get_param_or_default(params, "x_axis_values", default_x)[:len(default_x)]
    indices = list(range(0, len(default_x), params["filter_size"]))
    data_ = []
    for d,x0 in zip(data,x):
        data_.append([d[i] for i in indices])
    data = numpy.array(data_)
    x = [x[i] for i in indices]
    est = numpy.mean(data, axis=0)
    ci = get_param_or_default(params, "ci", 95)
    cis = bootstrap(data, ci=ci)
    color = get_param_or_default(params, "color", None)
    label = params["label"]
    x_label = params["x_label"]
    y_label = params["y_label"]
    plot.title(get_param_or_default(params, "title"))
    if color is not None:
        plot.fill_between(x,cis[0],cis[1],alpha=alpha, color=color, **kw)
        handle = plot.plot(x,est,label=label,color=color,**kw)
    else:
        plot.fill_between(x,cis[0],cis[1],alpha=alpha, **kw)
        handle = plot.plot(x,est,label=label, **kw)
    plot.xlabel(x_label)
    plot.ylabel(y_label)
    plot.margins(x=0)
    return handle

def meanplot(data, params, alpha=0.12, **kw):
    data = numpy.array(data)
    x = [0, params["plot_mean_length"]]
    y_data = []
    for y_dat in data:
        y_data += list(y_dat)
    data = numpy.array([[y] for y in y_data])
    est = numpy.mean(data)
    ci = get_param_or_default(params, "ci", 95)
    cis = bootstrap(data, ci=ci)
    color = get_param_or_default(params, "color", None)
    label = params["label"]
    x_label = params["x_label"]
    y_label = params["y_label"]
    plot.title(get_param_or_default(params, "title"))
    if color is not None:
        plot.fill_between(x,[cis[0][0], cis[0][0]],[cis[1][0], cis[1][0]],alpha=alpha, color=color, **kw)
        handle = plot.plot(x,[est, est],label=label,color=color,linestyle="dashed",**kw)
    else:
        plot.fill_between(x,[cis[0][0], cis[0][0]],[cis[1][0], cis[1][0]],alpha=alpha, **kw)
        handle = plot.plot(x,[est, est],label=label,linestyle="dashed",**kw)
    plot.xlabel(x_label)
    plot.ylabel(y_label)
    plot.margins(x=0)
    return handle

def plot_runs(params):
    data = []
    directory_count = 0
    filter_size = params["filter_size"]
    path = params["directory"]
    filename = get_param_or_default(params, "filename", "results.json")
    data_prefix_pattern = params["data_prefix_pattern"]
    stats_label = params["stats_label"]
    data_length = get_param_or_default(params, "data_length", None)
    nr_runs = get_param_or_default(params, "nr_runs", None)
    metric_index = get_param_or_default(params, "metric_index", None)
    norm_index = get_param_or_default(params, "norm_index", None)
    norm_nr_agents = get_param_or_default(params, "norm_nr_agents", 1)
    particular_folder = get_param_or_default(params, "particular_folder", None)
    plot_mean = get_param_or_default(params, "plot_mean", False)
    for directory in list_directories(path, lambda x: x.startswith(data_prefix_pattern)):
        if (nr_runs is None or directory_count < nr_runs) and (particular_folder is None or particular_folder in directory):
            for json_file in list_files(directory, lambda x: x == filename):
                json_data = load_json(json_file)
                if stats_label == "domain_values":
                    if metric_index is None:
                        return_values = numpy.sum(json_data["undiscounted_returns"], axis=0)
                    else:  
                        return_values = [value[metric_index] for value in json_data[stats_label]]
                    if norm_index is not None:
                        old_length = len(return_values)
                        norm = numpy.array([value[norm_index] for value in json_data[stats_label]])
                        return_values = numpy.array(return_values)/norm
                        assertEquals(old_length, len(return_values))
                elif stats_label in ["messages_sent", "response_messages_sent", "request_messages_sent"]:
                    return_values = json_data[stats_label]
                elif stats_label == "equality":
                    undiscounted_returns = json_data["undiscounted_returns"]
                    def compute_equality(joint_return):
                        N = len(joint_return)
                        abs_diff_sum = sum([sum([abs(joint_return[i] - joint_return[j]) for j in range(N)]) for i in range(N)])
                        norm = 2*N*sum(joint_return)
                        return 1 - abs_diff_sum*1.0/norm
                    return_values = []
                    for i in range(len(undiscounted_returns[0])):
                        joint_return = [returns[i] for returns in undiscounted_returns]
                        return_values.append(compute_equality(joint_return))
                else:
                    return_values = numpy.sum(json_data[stats_label], axis=0)
                if data_length is not None:
                    return_values = return_values[:data_length]
                kernel = numpy.ones(filter_size)/filter_size
                return_values = numpy.convolve(return_values, kernel, mode='valid')
                return_values /= norm_nr_agents
                data.append(return_values)
                directory_count += 1
    if len(data) > 0:
        print(data_prefix_pattern, "{} runs".format(directory_count))
        if plot_mean:
            return meanplot(data, params)
        else:
            params["filter_size"] = filter_size
            return tsplot(data, params)
    return None

def end_plot(data, params):
    y = [numpy.mean(d) for d in data]
    plot_mean = get_param_or_default(params, "plot_mean", False)
    confidence_intervals = [bootstrap(numpy.array(d)) for d in data]
    cis_min, cis_max = zip(*confidence_intervals)
    cis_min = numpy.array(cis_min)
    cis_max = numpy.array(cis_max)
    label = params["label"]
    x_label = params["x_label"]
    y_label = params["y_label"]
    x = numpy.arange(len(y))
    if plot_mean:
        plot.plot(x, y, color=params["color"], linestyle="dashed", label=label)
    else:
        plot.plot(x, y, color=params["color"], label=label, marker="+")
    plot.fill_between(x,cis_min,cis_max,alpha=0.12, color=params["color"])
    plot.xlabel(x_label)
    plot.ylabel(y_label)
    plot.margins(x=0)
    plot.xticks([r for r in range(len(y))], params["comm_failure_rates"])


def plot_run_end(params):
    data = []
    path = params["directory"]
    filename = get_param_or_default(params, "filename", "results.json")
    data_prefix_pattern = params["data_prefix_pattern"]
    comm_failure_rates = params["comm_failure_rates"]
    stats_label = params["stats_label"]
    data_length = get_param_or_default(params, "data_length", None)
    metric_index = get_param_or_default(params, "metric_index", None)
    norm_index = get_param_or_default(params, "norm_index", None)
    plot_mean = get_param_or_default(params, "plot_mean", False)
    for failure_rate in comm_failure_rates:
        nr_runs = get_param_or_default(params, "nr_runs", None)
        directory_count = 0
        bar_data = []
        data_prefix_pattern_ = data_prefix_pattern
        if failure_rate != "0" and params["algorithm_name"] != "IAC_" and params["algorithm_name"] != "Random_":
            suffix = "-{}".format(failure_rate)
            data_prefix_pattern_ = data_prefix_pattern[:-1] + suffix
        for directory in list_directories(path, lambda x: x.startswith(data_prefix_pattern_)):
            if nr_runs is None or directory_count < nr_runs:
                for json_file in list_files(directory, lambda x: x == filename):
                    json_data = load_json(json_file)
                    if stats_label == "domain_values":
                        if metric_index is None:
                            return_values = numpy.sum(json_data["undiscounted_returns"], axis=0)
                        else:  
                            return_values = [value[metric_index] for value in json_data[stats_label]]
                        if norm_index is not None:
                            old_length = len(return_values)
                            norm = numpy.array([value[norm_index] for value in json_data[stats_label]])
                            return_values = numpy.array(return_values)/norm
                            assertEquals(old_length, len(return_values))
                    elif stats_label in ["messages_sent", "response_messages_sent", "request_messages_sent"]:
                        return_values = json_data[stats_label]
                    elif stats_label == "equality":
                        undiscounted_returns = json_data["undiscounted_returns"]
                        def compute_equality(joint_return):
                            N = len(joint_return)
                            abs_diff_sum = sum([sum([abs(joint_return[i] - joint_return[j]) for j in range(N)]) for i in range(N)])
                            norm = 2*N*sum(joint_return)
                            return 1 - abs_diff_sum*1.0/norm
                        return_values = []
                        for i in range(len(undiscounted_returns[0])):
                            joint_return = [returns[i] for returns in undiscounted_returns]
                            return_values.append(compute_equality(joint_return))
                    else:
                        return_values = numpy.sum(json_data[stats_label], axis=0)
                    if data_length is not None:
                        return_values = return_values[:data_length]
                    if plot_mean:
                        bar_data += list(return_values)
                    else:
                        bar_data.append(numpy.mean(list(return_values)[-1]))
                    directory_count += 1
        data.append(bar_data)
    if len(data) > 0:
        print(data_prefix_pattern, "{} runs".format([len(b) for b in data]))
        return end_plot(data, params)
    return None

def show(showgrid=True, legend_position=None):
    if showgrid:
        plot.grid()
    if legend_position is not None:
        plot.legend(loc=legend_position)
    plot.show()