"""
I need a write config file widget.
"""

import sys, os
import numpy as np
import h5py
import scipy.constants as sc

import pyqtgraph as pg
import PyQt4.QtGui
import PyQt4.QtCore
import signal
import copy 

sys.path.append(os.path.abspath('../utils/'))
sys.path.append(os.path.abspath('../process/'))

#from widgets import Write_config_file_widget
from basic_stitch  import parse_cmdline_args
from widgets import Show_stitch_widget

def write_config_file_gui(config_dict, output_dir):
    signal.signal(signal.SIGINT, signal.SIG_DFL) # allow Control-C
    app = PyQt4.QtGui.QApplication([])
    
    # Qt main window
    Mwin = PyQt4.QtGui.QMainWindow()
    Mwin.setWindowTitle('Write configuration files')
    
    cw = Write_config_file_widget(config_dict, output_dir)
    
    # add the central widget to the main window
    Mwin.setCentralWidget(cw)
    
    Mwin.show()
    app.exec_()


def stitch(f, filename, config_dict):
    signal.signal(signal.SIGINT, signal.SIG_DFL) # allow Control-C
    app = PyQt4.QtGui.QApplication([])
    
    # Qt main window
    Mwin = PyQt4.QtGui.QMainWindow()
    Mwin.setWindowTitle('Stitches Bitches')
    
    cw = Show_stitch_widget(f, filename, config_dict)
    
    # add the central widget to the main window
    Mwin.setCentralWidget(cw)
    
    Mwin.show()
    app.exec_()

    

if __name__ == '__main__':
    args, params = parse_cmdline_args()
    
    f = h5py.File(args.filename)
    stitch(f, args.filename, params)
    
    # set the outputdirectory to the same as the input file
    #output_file = os.path.split(args.filename)[0]  
    #output_file = os.path.join(output_file, 'basic_stitch.ini')
    
    #output_file = 'test.ini'
    #write_config_file_gui(params, output_file)
