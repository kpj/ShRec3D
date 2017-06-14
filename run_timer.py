"""
Investigate complexity of algorithm
"""

from __future__ import division, print_function

import json, time, os
import traceback, sys

import numpy as np
import matplotlib.pyplot as plt

from main import apply_shrec3d


def generate_data(out_fname, data_directory):
    """ Run ShRec3D on all data in data directory
    """
    def store_result(duration, loci_number):
        """ Store result of current timing run
        """
        print('  %ds for %d loci' % (duration, loci_number))

        if os.path.isfile(out_fname):
            with open(out_fname, 'r') as fd:
                cur = json.load(fd)
        else:
            cur = []

        with open(out_fname, 'w') as fd:
            cur.append((loci_number, duration))
            json.dump(cur, fd)

    for fn in os.listdir(data_directory):
        fname = os.path.join(data_directory, fn)

        print('Loading "%s"...' % fname, end=' ', flush=True)
        contacts = np.loadtxt(fname)
        print('Done')

        start = time.time()
        try:
            apply_shrec3d(contacts)
        except:
            print('>>> Some error occured')
            traceback.print_exc()
        end = time.time()

        store_result(end-start, contacts.shape[0])


def plot_data(fname):
    """ Plot time points given in data file and compare to x**3
    """
    if not os.path.isfile(fname):
        print('No data has been generated yet, aborting...')
        sys.exit(1)

    with open(fname, 'r') as fd:
        data = json.load(fd)

    x = np.arange(0, max(data, key=lambda e: e[0])[0], 1)

    const = .55e-8
    func = lambda x: const * x**3

    plt.plot(
        *zip(*data),
        label=r'ShRec3D data points',
        linestyle='None', marker='h'
    )
    plt.plot(x, func(x), label=r'$ %.0e \cdot x^3$' % const)

    plt.title(r'Complexity ($\in \Theta\left(x^3\right)$) visualization of ShRec3D')
    plt.xlabel('loci number')
    plt.ylabel('execution time (seconds)')
    plt.legend(loc='best')

    plt.savefig('time_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """ Generate new data if directory is given, otherwise only try to plot existing data
    """
    data_file = 'shrec_timer.json'

    if len(sys.argv) == 2:
        generate_data(data_file, sys.argv[1])

    plot_data(data_file)

if __name__ == '__main__':
    main()
