# examples/basic_exp.py

from yanex import core as experiment
import time
import matplotlib.pyplot as plt

params = experiment.get_params()

with experiment.run():
    time.sleep(1.5)  # Simulate workload

    result = {
        "runtime_sec": 1.5,
        "docs_scanned": params.get("n_docs", 1000),
        "throughput": params.get("n_docs", 1000) / 1.5,
    }

    experiment.log_results(result)

    # Generate and save a figure
    fig = plt.figure()
    plt.plot([0, 1, 2], [0, result["docs_scanned"], 0])
    plot_path = "scan_plot.png"
    fig.savefig(plot_path)

    experiment.log_artifact("scan_plot.png", plot_path)
