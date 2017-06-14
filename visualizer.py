"""
Visualizer for 3D point clouds
"""

from __future__ import division, print_function

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


COLOR_LIST = ['blue', 'red', 'green']

def gen_sphere(x_pos, y_pos, z_pos, rad):
    """ Generate sphere coordinates around given center
    """
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]

    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)

    x = rad * x + x_pos
    y = rad * y + y_pos
    z = rad * z + z_pos

    return (x, y, z)

def visualize(coord_pairs, show_spheres=True):
    """ Plot 3D coordinates
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    for i, pack in enumerate(coord_pairs):
        coords, label = pack
        ax.scatter(*zip(*coords), color=COLOR_LIST[i], label=label)

        if show_spheres:
            for xc, yc, zc in coords:
                ax.plot_wireframe(*gen_sphere(xc, yc, zc, 0.21), color=COLOR_LIST[i])

    ax.legend()
    plt.savefig('foo.svg', bbox_inches='tight')
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
