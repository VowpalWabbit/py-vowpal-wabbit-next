import matplotlib.pyplot as plt
import numpy as np

# Benchmark results are stored in a csv file with the following format:
# command,mean,stddev,median,user,system,min,max
def load_benchmark_results(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        lines = lines[1:] # Remove header
        results = []
        for line in lines:
            command, mean, stddev, median, user, system, min, max = line.split(",")
            results.append({
                "command": command,
                "mean": round(float(mean), 4),
                "stddev": round(float(stddev), 4),
                "median": float(median),
                "user": float(user),
                "system": float(system),
                "min": float(min),
                "max": float(max),
            })
        return results




labels = []
cli_results_means = []
cli_results_stddevs = []
python_results_means = []
python_results_stddevs = []

def append_benchmark_result(filename):
    results = load_benchmark_results(filename)
    command  = " ".join(results[0]["command"].split(" ")[1:])
    labels.append(command)
    for result in results:
        if "python" in result["command"]:
            python_results_means.append(result["mean"])
            python_results_stddevs.append(result["stddev"])
        else:
            cli_results_means.append(result["mean"])
            cli_results_stddevs.append(result["stddev"])

for i in range(1, 8):
    append_benchmark_result(f"bench{i}.csv")

# Print a markdown table of the results
print("| Command | CLI | Python |")
print("| --- | --- | --- |")
for i in range(len(labels)):
    print(f"| `{labels[i]}` | {cli_results_means[i]} s ± {cli_results_stddevs[i]} s | {python_results_means[i]} s ± {python_results_stddevs[i]} s |")

ind = np.arange(len(cli_results_means))  # the x locations for the groups
width = 0.35  # the width of the bars


fig, ax = plt.subplots(figsize=(10,6))
rects1 = ax.bar(ind - width/2, cli_results_means, width, yerr=cli_results_stddevs,
                label='CLI')
rects2 = ax.bar(ind + width/2, python_results_means, width, yerr=python_results_stddevs,
                label='Python')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Seconds')
ax.set_title('Native CLI vs Python CLI driver')
ax.set_xticks(ind)
ax.set_xticklabels(labels)
ax.legend()

def autolabel(rects, xpos='center'):
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0, 'right': 1, 'left': -1}

    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(offset[xpos]*3, 3),  # use 3 points offset
                    textcoords="offset points",  # in both directions
                    ha=ha[xpos], va='bottom')


autolabel(rects1, "left")
autolabel(rects2, "right")
fig.subplots_adjust(bottom=0.2)
fig.set_dpi(100)

plt.xticks(rotation = 15)
# plt.show()
plt.savefig('plot.png')