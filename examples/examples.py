import matplotlib.pyplot as plt
import ipdb
import numpy as np

from lovely_histogram import plot_histogram


def main():
    rng = np.random.default_rng(seed=178423)
    data = 2 * rng.standard_normal(1024) + 0.3

    ax = plot_histogram(data)
    plt.show()


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
