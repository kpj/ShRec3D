"""
Visualizer for 3D point clouds
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


COLOR_LIST = ['blue', 'red', 'green']

def visualize(coord_pairs):
    """ Plot 3D coordinates
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    for i, pack in enumerate(coord_pairs):
        coords, label = pack
        ax.scatter(*zip(*coords), color=COLOR_LIST[i], label=label)

    ax.legend()
    plt.show()


if __name__ == '__main__':
    import sys
    import numpy as np

    if len(sys.argv) != 2:
        print('Usage: %s <coordinate file>' % sys.argv[0])
        sys.exit(1)

    fname = sys.argv[1]
    coords = np.load(fname)
    visualize([(coords, fname)])
