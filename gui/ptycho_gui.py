"""

"""
import sys, os
import numpy as np
import h5py
import scipy.constants as sc

import pyqtgraph as pg
#import PyQt4.QtGui
try :
    from PyQt5 import QtGui
except :
    from PyQt4 import QtGui
import signal
import copy 
import ConfigParser

root = os.path.split(os.path.abspath(__file__))[0]
root = os.path.split(root)[0]

sys.path.insert(1, os.path.join(root, 'utils'))
sys.path.insert(1, os.path.join(root, 'process'))

from Ptychography import utils

#from widgets import Write_config_file_widget
from widgets import Show_stitch_widget
from widgets import Show_cpu_stitch_widget
from widgets import Show_EMC_widget
from widgets import Show_frames_widget
from widgets import Select_frames_widget
from widgets import Show_frames_selection_widget
from widgets import Mask_maker_widget
from widgets import Show_probe_widget
from widgets import Phase_widget
from widgets import config_default
from widgets import Show_h5_list_widget
from widgets import View_h5_data_widget
from widgets import Test_run_command_widget
from widgets import Zernike_widget
from widgets import Defocus_widget
from widgets import Show_make_pixel_shifts_widget

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
        
def init_file(filename):
    f = h5py.File(filename)
    print(config_default['output'], f.keys(), config_default['output'] not in f.keys())
    if config_default['output'] not in f.keys():
        # add the output group
        g = f.create_group(config_default['output'])

        # add the good_frames (assume they are all good)
        g['good_frames'] = np.arange(len(f[config_default['input']['data']]))

    # done 
    f.close()

class Gui(QtGui.QTabWidget):
    def __init__(self):
        super(Gui, self).__init__()

    def initUI(self, filename):
        # initialise the scratch space if not done already
        init_file(filename)
        
        self.tabs = []

        self.setMovable(True)
        #self.setTabsClosable(True)

        # Show frames tab
        #################
        self.tabs.append( Show_frames_selection_widget(filename) )
        self.addTab(self.tabs[-1], "show frames")

        # Show h5 list tab
        #################
        self.tabs.append( View_h5_data_widget(filename) )
        self.addTab(self.tabs[-1], "show h5 dataset")

        """
        # Show test
        #################
        self.tabs.append( Test_run_command_widget(filename) ) 
        self.addTab(self.tabs[-1], "testing")
        #self.tabs[-1].initUI()
        """
        
        # Show stitch tab
        #################
        # load the default config file
        params = load_config(filename, name='basic_stitch.ini')
        self.tabs.append( Show_stitch_widget(filename, params) )
        self.addTab(self.tabs[-1], "show stitch")

        # Show gpu stitch tab
        #####################
        # load the default config file
        params = load_config(filename, name='cpu_stitch.ini')
        self.tabs.append( Show_cpu_stitch_widget(filename, params) )
        self.addTab(self.tabs[-1], "stitch with pixel shifts")
        
        # Show EMC tab
        #####################
        # load the default config file
        params = load_config(filename, name='make_pixel_shifts.ini')
        self.tabs.append( Show_make_pixel_shifts_widget(filename, params) )
        self.addTab(self.tabs[-1], "make pixel shifts")

        # Show EMC tab
        #####################
        # load the default config file
        params = load_config(filename, name='EMC.ini')
        self.tabs.append( Show_EMC_widget(filename, params) )
        self.addTab(self.tabs[-1], "EMC stitcher")

        # Show gpu stitch tab
        #####################
        # load the default config file
        params = load_config(filename, name='Zernike.ini')
        self.tabs.append( Zernike_widget(filename, params) )
        self.addTab(self.tabs[-1], "Zernike decomposition widget")

        # mask Maker tab
        ################
        self.tabs.append( Mask_maker_widget(filename, config_default['output'] + '/mask', filename, config_default['output'] + '/mask') )
        self.addTab(self.tabs[-1], "mask maker")
        
        # defocus tab
        #################
        # load the default config file
        params = load_config(filename, name='defocus.ini')
        self.tabs.append( Defocus_widget(filename, params) )
        self.addTab(self.tabs[-1], "defocus")
        """
        # probe maker tab
        #################
        # show real-space / detector-space probe
        # change defocus
        params_probe = load_config(filename, name='make_probe.ini')
        self.tabs.append( Show_probe_widget(filename, params_probe) )
        self.addTab(self.tabs[-1], "show probe")

        # sample maker tab
        ##################
        # back-propagation
        # stitch
        # random
        # sample-pos overlay
        
        # phase tab
        ##################
        params = load_config(filename, name='phase.ini')
        self.tabs.append( Phase_widget(filename, params) )
        self.addTab(self.tabs[-1], "phase")
        """
        

def gui(filename):
    signal.signal(signal.SIGINT, signal.SIG_DFL) # allow Control-C
    app = QtGui.QApplication([])
    
    # Qt main window
    Mwin = QtGui.QMainWindow()
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
