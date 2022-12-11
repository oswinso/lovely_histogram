import ipdb
import matplotlib.pyplot as plt
import numpy as np

from lovely_histogram import plot_histogram


def main():
    rng = np.random.default_rng(seed=178423)
    size = (8, 12, 7)
    data1 = 1 * rng.standard_normal(size) + 0.8
    data2 = 0.4 * rng.standard_normal(size) - 0.5
    data = np.stack([data1, data2], axis=-1)

    fig, ax = plt.subplots(figsize=(12, 2), dpi=200, constrained_layout=True)
    plot_histogram(data, ax=ax)
    fig.savefig("histogram")


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
