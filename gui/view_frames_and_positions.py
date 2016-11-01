import sys, os
import numpy as np
import h5py
import scipy.constants as sc

import pyqtgraph as pg
import PyQt4.QtGui
import PyQt4.QtCore
import signal


root = os.path.split(os.path.abspath(__file__))[0]
root = os.path.split(root)[0]

sys.path.append(os.path.join(root, 'utils'))
sys.path.append(os.path.join(root, 'process'))

from widgets import Show_frames_widget

def parse_cmdline_args():
    import argparse
    import os
    parser = argparse.ArgumentParser(description='show the individual frames as well as a scatter plot of the sample coordinates')
    parser.add_argument('filename', type=str, \
                        help="file name of the h5 file")
    
    args = parser.parse_args()

    # check that cxi file exists
    if not os.path.exists(args.filename):
        raise NameError('cxi file does not exist: ' + args.filename)
    
    return args


def show_frames(f):
    signal.signal(signal.SIGINT, signal.SIG_DFL) # allow Control-C
    app = PyQt4.QtGui.QApplication([])
    
    # Qt main window
    Mwin = PyQt4.QtGui.QMainWindow()
    Mwin.setWindowTitle('View frames and sample positions')
    
    cw = Show_frames_widget(f)
    cw.initUI()
    
    # add the central widget to the main window
    Mwin.setCentralWidget(cw)
    
    Mwin.show()
    app.exec_()

if __name__ == '__main__':
    args = parse_cmdline_args()
    f = h5py.File(args.filename, 'r')
    show_frames(args.filename)
