"""

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
import ConfigParser

root = os.path.split(os.path.abspath(__file__))[0]
root = os.path.split(root)[0]

sys.path.append(os.path.join(root, 'utils'))
sys.path.append(os.path.join(root, 'process'))

from Ptychography import utils

#from widgets import Write_config_file_widget
from widgets import Show_stitch_widget
from widgets import Show_frames_widget
from widgets import Select_frames_widget
from widgets import Show_frames_selection_widget
from widgets import Mask_maker_widget
from widgets import Show_probe_widget

def load_config(filename, name = 'basic_stitch.ini'):
    # if config is non then read the default from the *.pty dir
    config = os.path.join(os.path.split(filename)[0], name)
    if not os.path.exists(config):
        config = os.path.join(root, 'process')
        config = os.path.join(config, name)
    
    # check that args.config exists
    if not os.path.exists(config):
        raise NameError('config file does not exist: ' + config)
    
    # process config file
    conf = ConfigParser.ConfigParser()
    conf.read(config)
    
    params = utils.parse_parameters(conf)
    return params
        

class Gui(PyQt4.QtGui.QTabWidget):
    def __init__(self):
        super(Gui, self).__init__()

    def initUI(self, filename):
        self.tabs = []
            
        self.setMovable(True)
        #self.setTabsClosable(True)

        # Show frames tab
        #################
        self.tabs.append( Show_frames_selection_widget(filename) )
        self.addTab(self.tabs[-1], "show frames")

        # Show stitch tab
        #################
        # load the default config file
        params = load_config(filename, name='basic_stitch.ini')
        self.tabs.append( Show_stitch_widget(filename, params) )
        self.addTab(self.tabs[-1], "show stitch")
        
        # mask Maker tab
        ################
        self.tabs.append( Mask_maker_widget(filename, 'mask', filename, 'mask') )
        self.addTab(self.tabs[-1], "mask maker")
        
        # probe maker tab
        #################
        # show real-space / detector-space probe
        # change defocus
        params = load_config(filename, name='make_probe.ini')
        self.tabs.append( Show_probe_widget(filename, params) )
        self.addTab(self.tabs[-1], "show probe")

        # sample maker tab
        ##################
        # back-propagation
        # stitch
        # random
        # sample-pos overlay
        
        # phase tab
        ##################

def gui(filename):
    signal.signal(signal.SIGINT, signal.SIG_DFL) # allow Control-C
    app = PyQt4.QtGui.QApplication([])
    
    # Qt main window
    Mwin = PyQt4.QtGui.QMainWindow()
    Mwin.setWindowTitle(filename)
    
    cw = Gui()
    cw.initUI(filename)
    
    # add the central widget to the main window
    Mwin.setCentralWidget(cw)
    
    Mwin.show()
    app.exec_()

def parse_cmdline_args():
    import argparse
    import os
    parser = argparse.ArgumentParser(description='calculate a basic stitch of the projection images')
    parser.add_argument('filename', type=str, \
                        help="file name of the *.pty file")
    
    args = parser.parse_args()
    
    # check that cxi file exists
    if not os.path.exists(args.filename):
        raise NameError('cxi file does not exist: ' + args.filename)
    
    return args

if __name__ == '__main__':
    args = parse_cmdline_args()
    
    gui(args.filename)
