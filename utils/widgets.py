#!/usr/bin/env python

"""
All of the widgets here read/write to a h5 file. 
It needs to have the following structure:
    data                : /entry_1/data_1/data
    positions [x, y, z] : /entry_1/sample_N/geometry/translation 
    detector pixel directions : /entry_1/instrument_1/detector_1/basis_vectors

Ideally these should be defined in a config file, but not now.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys, os
import numpy as np
import h5py
import scipy.constants as sc

import pyqtgraph as pg
#import PyQt4.QtGui
#import PyQt4.QtCore
try :
    from PyQt5 import QtGui, QtCore
except :
    from PyQt4 import QtGui, QtCore
import signal
import copy 

import ConfigParser

root = os.path.split(os.path.abspath(__file__))[0]
root = os.path.split(root)[0]

from Ptychography import utils

# set the config defaults 
config_default = {}
config_default['input'] = {} 
config_default['input']['data'] = '/entry_1/data_1/data'
config_default['input']['R']    = '/entry_1/sample_3/geometry/translation'
config_default['input']['whitefield'] = '/process_2/powder'

config_default['output'] = 'process_3'

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

class Show_h5_list_widget(QtGui.QWidget):
    def __init__(self, filename, names = None):
        super(Show_h5_list_widget, self).__init__()

        self.filename = filename
        self.names    = names
        
        # add the names to Qlist thing
        self.listWidget = QtGui.QListWidget(self)
        #self.listWidget.setMinimumWidth(self.listWidget.sizeHintForColumn(0))
        #self.listWidget.setMinimumHeight(500)
        
        # update list button
        ####################
        self.update_button = QtGui.QPushButton('update', self)
        self.update_button.clicked.connect(self.update)

        # get the list of groups and items
        self.dataset_names = [] 
        self.dataset_items = [] 
        
        f = h5py.File(filename, 'r')
        f.visititems(self.add_dataset_name)
        f.close()

        self.initUI()
    
    def initUI(self):
        # set the layout
        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.listWidget)
        layout.addWidget(self.update_button)
        
        # add the layout to the central widget
        self.setLayout(layout)

    def add_dataset_name(self, name, obj):
        names = self.names
        if isinstance(obj, h5py.Dataset):
            if ((names is None) or (names is not None and name in names)) \
                    and name not in self.dataset_names:
                self.dataset_names.append(name)
                self.dataset_items.append(QtGui.QListWidgetItem(self.listWidget))
                self.dataset_items[-1].setText(name)
    
    def update(self):
        f = h5py.File(self.filename, 'r')
        f.visititems(self.add_dataset_name)
        f.close()


class Show_nd_data_widget(QtGui.QWidget):
    def __init__(self):
        super(Show_nd_data_widget, self).__init__()

        self.plotW  = None
        self.plotW2 = None
        self.layout = None
        self.name   = None
        self.initUI()
    
    def initUI(self):
        # set the layout
        self.layout = QtGui.QVBoxLayout()
        
        # add the layout to the central widget
        self.setLayout(self.layout)
    
    def show(self, filename, name, refresh=False):
        """
        plots:
            (N,)      float, int          --> line plot
            (N, M<4)  float, int          --> line plots
            (N, M>4)  float, complex, int --> 2d image
            (N, M>4)  complex             --> 2d images (abs, angle, real, imag)
            (N, M, L) float, complex, int --> 2d images (real) with slider
        """
        # make plot
        f = h5py.File(filename, 'r')
        shape = f[name].shape

        if len(shape) == 1 :
            if refresh :
                self.plotW.setData(f[name][()])
            else :
                self.plotW = pg.PlotWidget(title = name)
                self.plotW.plot(f[name][()], pen=(255, 150, 150))
        
        elif len(shape) == 2 and shape[1] < 4 :
            pens = [(255, 150, 150), (150, 255, 150), (150, 150, 255)]
            if refresh :
                self.plotW.clear()
                for i in range(shape[1]):
                    self.plotW.setData(f[name][:, i], pen=pens[i])
            else :
                self.plotW = pg.PlotWidget(title = name + ' [0, 1, 2] are [r, g, b]')
                for i in range(shape[1]):
                    self.plotW.plot(f[name][:, i], pen=pens[i])

        elif len(shape) == 2 :
            if refresh :
                self.plotW.setImage(f[name][()].real.T, autoRange = False, autoLevels = False, autoHistogramRange = False)
            else :
                if 'complex' in f[name].dtype.name :
                    title = name + ' (abs, angle, real, imag)'
                else :
                    title = name
                
                frame_plt = pg.PlotItem(title = title)
                self.plotW = pg.ImageView(view = frame_plt)
                self.plotW.ui.menuBtn.hide()
                self.plotW.ui.roiBtn.hide()
                if 'complex' in f[name].dtype.name :
                    im = f[name][()].T
                    self.plotW.setImage(np.array([np.abs(im), np.angle(im), im.real, im.imag]))
                else :
                    self.plotW.setImage(f[name][()].T)

        elif len(shape) == 3 :
            if refresh :
                replot_frame()
            else :
                # show the first frame
                frame_plt = pg.PlotItem(title = name)
                self.plotW = pg.ImageView(view = frame_plt)
                self.plotW.ui.menuBtn.hide()
                self.plotW.ui.roiBtn.hide()
                self.plotW.setImage(f[name][0].real.T)
                
                # add a little 1d plot with a vline
                self.plotW2 = pg.PlotWidget(title = 'index')
                self.plotW2.plot(np.arange(f[name].shape[0]), pen=(255, 150, 150))
                vline = self.plotW2.addLine(x = 0, movable=True, bounds = [0, f[name].shape[0]-1])
                self.plotW2.setMaximumSize(10000000, 100)
                
                def replot_frame():
                    i = int(vline.value())
                    f = h5py.File(filename, 'r')
                    self.plotW.setImage( f[name][i].real.T, autoRange = False, autoLevels = False, autoHistogramRange = False)
                    f.close()
                    
                vline.sigPositionChanged.connect(replot_frame)

        f.close()
         
        # add to layout
        self.layout.addWidget(self.plotW, stretch = 1)
        
        if self.plotW2 is not None :
            self.layout.addWidget(self.plotW2, stretch = 0)
        
        # remember the last file and dataset for updating
        self.name     = name
        self.filename = filename
    
    def close(self):
        # remove from layout
        if self.layout is not None :
            if self.plotW is not None :
                self.layout.removeWidget(self.plotW)
            
            if self.plotW2 is not None :
                self.layout.removeWidget(self.plotW2)
        
        # close plot widget
        if self.plotW is not None :
            self.plotW.close()
            self.plotW = None
        
        if self.plotW2 is not None :
            self.plotW2.close()
            self.plotW2 = None
    
    def update(self):
        # update the current plot
        self.show(self.filename, self.name, True)


class View_h5_data_widget(QtGui.QWidget):
    def __init__(self, filename, names = None):
        super(View_h5_data_widget, self).__init__()
        
        self.filename = filename
        self.names = names
            
        self.show_list_widget = Show_h5_list_widget(filename, names = names)
        self.plot1dWidget = Show_nd_data_widget()
        
        # send a signal when an item is clicked
        self.show_list_widget.listWidget.itemClicked.connect(self.dataset_clicked)

        self.initUI()

    def initUI(self):
        layout = QtGui.QHBoxLayout()
        
        # add the layout to the central widget
        self.setLayout(layout)

        # add the h5 datasets list
        layout.addWidget(self.show_list_widget)
        
        # add the 1d viewer 
        layout.addWidget(self.plot1dWidget, stretch=1)
        

    def dataset_clicked(self, item):
        name = str(item.text())
        
        # close the last image
        self.plot1dWidget.close()
        
        # load the new one
        self.plot1dWidget.show(self.filename, name)
        
    def update(self):
        self.show_list_widget.update()
        self.plot1dWidget.update()

class Test_run_command_widget(QtGui.QWidget):
    def __init__(self, h5_filename):
        super(Test_run_command_widget, self).__init__()

        self.h5_filename = h5_filename
        
        # set the python filename
        pyname = 'template_command.py'
        
        self.py = os.path.join(root, 'process/' + pyname)
        
        # read the config file
        self.config_dict  = load_config(h5_filename, name = pyname[:-2] + 'ini')
        self.output_dir   = os.path.split(h5_filename)[0]
        self.config_out   = os.path.join(self.output_dir, pyname[:-2] + 'ini')
        self.config_group = 'template_command'
        
        self.cmd = 'python ' + self.py + ' ' + h5_filename + ' -c ' + self.config_out
        
        self.run_com_Widget = Run_command_template_widget(\
                             h5_filename, self.config_dict, self.config_out, \
                             self.config_group, self.cmd, h5_datas = ['image', 'error'])
        self.initUI()
    
    def initUI(self):
        # Make a grid layout
        layout = QtGui.QVBoxLayout()
        
        # add the layout to the central widget
        self.setLayout(layout)

        layout.addWidget(self.run_com_Widget)


        
class Run_command_template_widget(QtGui.QWidget):
    """
    I take a h5 filename and a config dictionary. You should
    subclass me. 
    
    GUI layout:

    Widget box:         output tabs
    Run command button  frame / error plots ...
    ...

    Config editor:
    output_group
    number of frames
    ...
    
    Status:
    Command:
    """
    def __init__(self, h5_filename, config_dict, config_out, config_group \
                 ,cmd, h5_datas = ['image', 'error']):
        super(Run_command_template_widget, self).__init__()
        
        self.h5_filename = h5_filename
        self.config_dict = config_dict
        self.config_out  = config_out
        self.config_group = config_group
        self.h5_datas    = h5_datas
        self.cmd         = cmd
        
        self.h5_out       = self.config_dict[config_group]['output_file']
        self.h5_out_group = self.config_dict[config_group]['output_group']
        if self.h5_out is None :
            self.h5_out = h5_filename
        
        self.initUI()

    def initUI(self):
        """
        """
        # Make a grid layout
        layout = QtGui.QVBoxLayout()
        
        # add the layout to the central widget
        self.setLayout(layout)

        # view data widget
        ##################
        # make a list of the output to look out for 
        out_names = [self.h5_out_group + '/' + d for d in self.h5_datas]
        self.view_output_widget = View_h5_data_widget(self.h5_out, names = out_names)
        
        # config widget
        ###############
        self.config_widget = Write_config_file_widget(self.config_dict, self.config_out)

        # run command widget
        ####################
        self.run_command_widget = Run_and_log_command()
        self.run_command_widget.finished_signal.connect(self.finished)
        
        # run command button
        ####################
        self.run_button = QtGui.QPushButton('Calculate', self)
        self.run_button.clicked.connect(self.run_button_clicked)
        
        # A do something button
        ##################
        self.do_button = QtGui.QPushButton('do something', self)
        self.do_button.clicked.connect(self.do_button_clicked)
        
        # add a spacer for the labels and such
        verticalSpacer = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        
        # set the layout
        ################
        vlay = QtGui.QVBoxLayout() 
        vlay.addWidget(self.run_button)
        vlay.addWidget(self.do_button)
        vlay.addWidget(self.config_widget)
        vlay.addStretch(1)
        
        hlay = QtGui.QHBoxLayout() 
        hlay.addLayout(vlay, stretch = 0)
        hlay.addWidget(self.view_output_widget, stretch = 1)

        layout.addLayout(hlay)
        layout.addWidget(self.run_command_widget)

    def run_button_clicked(self):
        # write the config file 
        #######################
        self.config_widget.write_file()
          
        # Run the command 
        #################
        self.run_command_widget.run_cmd(self.cmd)
    
    def finished(self):
        self.view_output_widget.update()
    
    def do_button_clicked(self):
        print('you pressed the do button: I do nothing')

class Phase_widget(QtGui.QWidget):
    def __init__(self, filename, config_dict):
        super(Phase_widget, self).__init__()
        
        # make a timer for updating O and P
        ###################################
        self.timer = QtCore.QTimer()
        self.timer.setInterval(2000) # milli-seconds
        self.timer.timeout.connect(self.show_OP)
        
        self.initUI(filename, config_dict)

    def initUI(self, filename, config_dict):
        # get the output directory
        self.output_dir = os.path.split(filename)[0]
        self.config_filename = os.path.join(self.output_dir, 'phase.ini')
        self.filename = filename
        
        # Make a grid layout
        layout = QtGui.QGridLayout()
        
        # add the layout to the central widget
        self.setLayout(layout)
        
        # config widget
        ###############
        self.config_widget = Write_config_file_widget(config_dict, self.config_filename)
        
        # sample amplitude plot
        #############################
        frame_plt = pg.PlotItem(title = 'sample amplitude') 
        self.o_amp_imageView = pg.ImageView(view = frame_plt)
        self.o_amp_imageView.ui.menuBtn.hide()
        self.o_amp_imageView.ui.roiBtn.hide()

        # sample phase plot
        #############################
        frame_plt = pg.PlotItem(title = 'sample phase') 
        self.o_phase_imageView = pg.ImageView(view = frame_plt)
        self.o_phase_imageView.ui.menuBtn.hide()
        self.o_phase_imageView.ui.roiBtn.hide()
        
        # probe amplitude plot
        #############################
        frame_plt = pg.PlotItem(title = 'probe amplitude') 
        self.p_amp_imageView = pg.ImageView(view = frame_plt)
        self.p_amp_imageView.ui.menuBtn.hide()
        self.p_amp_imageView.ui.roiBtn.hide()

        # probe phase plot
        #############################
        frame_plt = pg.PlotItem(title = 'probe phase') 
        self.p_phase_imageView = pg.ImageView(view = frame_plt)
        self.p_phase_imageView.ui.menuBtn.hide()
        self.p_phase_imageView.ui.roiBtn.hide()
        
        # Modulus error plot
        ####################
        self.eMod_plotW = pg.PlotWidget(bottom='iteration', left='error', title = 'Modulus error')

        self.show_OP()
        
        # run command widget
        ####################
        self.run_command_widget = Run_and_log_command()
        self.run_command_widget.finished_signal.connect(self.show_OP)
        
        # stop the timer when the command has finished
        self.run_command_widget.finished_signal.connect(self.timer.stop)
        
        # run command button
        ####################
        self.run_button = QtGui.QPushButton('Calculate', self)
        self.run_button.clicked.connect(self.run_button_clicked)
        
        # set sample and probe
        ######################
        self.set_button = QtGui.QPushButton('set: O, P', self)
        self.set_button.clicked.connect(self.set_button_clicked)
        
        # add a spacer for the labels and such
        verticalSpacer = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        
        # set the layouts
        #################
        
        # another grid layout for the images
        ####################################
        layout_ims = QtGui.QGridLayout()
        layout_ims.setRowStretch(0, 1)
        layout_ims.setRowStretch(1, 1)
        layout_ims.setRowStretch(2, 0)
        layout_ims.addWidget(self.o_amp_imageView,    0, 0, 1, 1)
        layout_ims.addWidget(self.o_phase_imageView,  0, 1, 1, 1)
        layout_ims.addWidget(self.p_amp_imageView,    1, 0, 1, 1)
        layout_ims.addWidget(self.p_phase_imageView,  1, 1, 1, 1)
        
        # make a widget and the grid layout to it
        W = QtGui.QWidget()
        W.setLayout(layout_ims)
        
        # then make a splitter between the emod plot and the 
        # images
        splitter = QtGui.QSplitter(QtCore.Qt.Vertical)
        splitter.addWidget(W)
        splitter.addWidget(self.eMod_plotW)
        layout.addWidget(splitter, 0, 1, 1, 1)

        # another grid layout for the buttons
        #####################################
        layout_but = QtGui.QGridLayout()
        layout.addLayout(layout_but, 0, 0, 1, 1)
        layout_but.addWidget(self.config_widget,       0, 0, 1, 1)
        layout_but.addWidget(self.run_button,          1, 0, 1, 1)
        layout_but.addWidget(self.set_button,          2, 0, 1, 1)
        layout_but.addItem(verticalSpacer,             3, 0, 1, 1)
        
        layout.addWidget(self.run_command_widget,  1, 0, 1, 2)
        
        layout.setColumnStretch(1, 1)
        #layout.setColumnMinimumWidth(1, 550)
        #layout.setColumnMinimumWidth(2, 150)
        self.layout = layout

    def show_OP(self):
        fopen = True
        self.f = h5py.File(self.filename, 'r')
        path = self.config_widget.output_config['phase']['h5_group']
        if path in self.f :
            p = self.f[path+'/P'][()]
            p_amp   = np.abs(p.T)
            p_phase = np.angle(p.T)
            
            o = self.f[path+'/O'][()]
            o_amp   = np.abs(o.T)
            o_phase = np.angle(o.T)

            eMod = self.f[path+'/eMod'][()]
            self.f.close()
            fopen = False
            
            self.o_amp_imageView.setImage(  o_amp,   autoRange = False, autoLevels = False, autoHistogramRange = False)
            self.o_phase_imageView.setImage(o_phase, autoRange = False, autoLevels = False, autoHistogramRange = False)
            self.p_amp_imageView.setImage(  p_amp,   autoRange = False, autoLevels = False, autoHistogramRange = False)
            self.p_phase_imageView.setImage(p_phase, autoRange = False, autoLevels = False, autoHistogramRange = False)
            self.eMod_plotW.clear()
            self.eMod_plotW.setXRange(0, len(eMod)+1)
            self.eMod_plotW.plot(eMod)
        if fopen :
            self.f.close()

    def run_button_clicked(self):
        # write the config file 
        #######################
        self.config_widget.write_file()
         
        # Run the command 
        #################
        py = os.path.join(root, 'process/phase.py')
        cmd = 'mpirun -n 16 python ' + py + ' ' + self.filename + ' ' + self.config_filename
        self.run_command_widget.run_cmd(cmd)

        # start a timer for O and P updates
        ###################################
        self.timer.start()
    
    def set_button_clicked(self):
        path = self.config_widget.output_config['phase']['h5_group']
        f = h5py.File(self.filename)
        P = f[path+'/P'][()]
        O = f[path+'/O'][()]
        print('writing O and P to file')
        if 'P' in f :
            del f['P']
        f['P'] = P
        if 'O' in f :
            del f['O']
        f['O'] = O
        print('sum O : ', np.sum(O))
        print('Done')
        f.close()

class Show_probe_widget(QtGui.QWidget):
    def __init__(self, filename, config_dict):
        super(Show_probe_widget, self).__init__()
        
        self.initUI(filename, config_dict)

    def initUI(self, filename, config_dict):
        """
        First show the stitch if there is one...
        
        We need: 
        an imageview on the left
        a config writer on the right
        below that a run button 
        below that a run and log command widget
        
        then wait for command to finish and show the image
        
        would be cool if we could show a scatter plot of the 
        positions on top of the stitch image.
        """
        
        # get the output directory
        self.output_dir = os.path.split(filename)[0]
        self.config_filename = os.path.join(self.output_dir, 'make_probe.ini')
        self.filename = filename
        self.f = h5py.File(filename, 'r')
        
        # Make a grid layout
        layout = QtGui.QGridLayout()
        
        # add the layout to the central widget
        self.setLayout(layout)
        
        # config widget
        ###############
        self.config_widget = Write_config_file_widget(config_dict, self.config_filename)

        # sample plane amplitude plot
        #############################
        frame_plt = pg.PlotItem(title = 'sample plane amplitude') 
        self.p_amp_imageView = pg.ImageView(view = frame_plt)
        self.p_amp_imageView.ui.menuBtn.hide()
        self.p_amp_imageView.ui.roiBtn.hide()

        # sample plane phase plot
        #############################
        #frame_plt = pg.PlotItem(title = '')
        #self.p_phase_imageView = pg.ImageView(view = frame_plt)
        #self.p_phase_imageView.ui.menuBtn.hide()
        #self.p_phase_imageView.ui.roiBtn.hide()

        # detector plane intensity plot
        #############################
        frame_plt = pg.PlotItem(title = 'detector plane intensity')
        self.P_int_imageView = pg.ImageView(view = frame_plt)
        self.P_int_imageView.ui.menuBtn.hide()
        self.P_int_imageView.ui.roiBtn.hide()
        
        # detector plane phase plot
        #############################
        frame_plt = pg.PlotItem(title = 'detector plane phase')
        self.P_phase_imageView = pg.ImageView(view = frame_plt)
        self.P_phase_imageView.ui.menuBtn.hide()
        self.P_phase_imageView.ui.roiBtn.hide()

        self.f.close()
        
        self.im_init = False
        self.show_probe()

        # run command widget
        ####################
        self.run_command_widget = Run_and_log_command()
        self.run_command_widget.finished_signal.connect(self.show_probe)

        # run command button
        ####################
        self.run_button = QtGui.QPushButton('Calculate', self)
        self.run_button.clicked.connect(self.run_button_clicked)
        
        # set sample and R
        ##################
        self.set_button = QtGui.QPushButton('set: P', self)
        self.set_button.clicked.connect(self.set_button_clicked)
        
        # add a spacer for the labels and such
        verticalSpacer = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        
        # set the layout
        ################
        layout.addWidget(self.p_amp_imageView,           0, 1, 2, 1)
        layout.addWidget(self.P_int_imageView,           2, 1, 1, 1)
        layout.addWidget(self.P_phase_imageView,         3, 1, 1, 1)
        
        layout.addWidget(self.config_widget,       0, 0, 1, 1)
        layout.addWidget(self.run_button,          1, 0, 1, 1)
        layout.addWidget(self.set_button,          2, 0, 1, 1)
        layout.addItem(verticalSpacer,             3, 0, 1, 1)
        layout.addWidget(self.run_command_widget,  4, 0, 1, 2)
        layout.setColumnStretch(1, 1)
        layout.setColumnMinimumWidth(0, 250)
        self.layout = layout

    def show_probe(self):
        self.f = h5py.File(self.filename, 'r')
        
        self.path = self.config_widget.output_config['make_probe']['h5_group']
        if self.path in self.f :
            p = self.f[self.path+'/P'][()]
            p_amp   = np.abs(p.T)
            #p_phase = np.angle(p.T)
            
            P = self.f[self.path+'/pupil'][()]
            P_int   = np.abs(P.T)**2
            P_phase = np.angle(P.T)
        
            if self.im_init :
                self.p_amp_imageView.setImage(  p_amp,   autoRange = False, autoLevels = False, autoHistogramRange = False)
                #self.p_phase_imageView.setImage(p_phase, autoRange = False, autoLevels = False, autoHistogramRange = False)
                self.P_int_imageView.setImage(  P_int,   autoRange = False, autoLevels = False, autoHistogramRange = False)
                self.P_phase_imageView.setImage(P_phase, autoRange = False, autoLevels = False, autoHistogramRange = False)
            else :
                self.p_amp_imageView.setImage(p_amp)
                #self.p_phase_imageView.setImage(p_phase)
                self.P_int_imageView.setImage(P_int)
                self.P_phase_imageView.setImage(P_phase)
                self.im_init = True
        self.f.close()
    
    def run_button_clicked(self):
        # write the config file 
        #######################
        self.config_widget.write_file()
         
        # Run the command 
        #################
        py = os.path.join(root, 'process/make_probe.py')
        cmd = 'python ' + py + ' ' + self.filename + ' -c ' + self.config_filename
        self.run_command_widget.run_cmd(cmd)
    
    def set_button_clicked(self):
        f = h5py.File(self.filename)
        P = f[self.path+'/P'][()]
        print('writing P to file')
        if 'P' in f :
            del f['P']
        f['P'] = P
        print('Done')
        f.close()

class Write_config_file_widget(QtGui.QWidget):
    def __init__(self, config_dict, output_filename):
        super(Write_config_file_widget, self).__init__()
        
        self.output_filename = output_filename
        self.initUI(config_dict)
    
    def initUI(self, config_dict):
        # Make a grid layout
        layout = QtGui.QGridLayout()
        
        # add the layout to the central widget
        self.setLayout(layout)
        
        self.output_config = copy.deepcopy(config_dict)
        
        i = 0
        # add the output config filename 
        ################################    
        fnam_label = QtGui.QLabel(self)
        fnam_label.setText(self.output_filename)
        
        # add the label to the layout
        layout.addWidget(fnam_label, i, 0, 1, 2)
        i += 1
        
        # we have 
        self.labels_lineedits = {}
        group_labels = []
        for group in config_dict.keys():
            # add a label for the group
            group_labels.append( QtGui.QLabel(self) )
            group_labels[-1].setText(group)
            # add the label to the layout
            layout.addWidget(group_labels[-1], i, 0, 1, 2)
            i += 1
            
            self.labels_lineedits[group] = {}
            # add the labels and line edits
            for key in config_dict[group].keys():
                self.labels_lineedits[group][key] = {}
                self.labels_lineedits[group][key]['label'] = QtGui.QLabel(self)
                self.labels_lineedits[group][key]['label'].setText(key)
                layout.addWidget(self.labels_lineedits[group][key]['label'], i, 0, 1, 1)
                
                self.labels_lineedits[group][key]['lineedit'] = QtGui.QLineEdit(self)
                # special case when config_dict[group][key] is a list
                if type(config_dict[group][key]) is list or type(config_dict[group][key]) is np.ndarray :
                    setT = ''
                    for ii in range(len(config_dict[group][key])-1):
                        setT += str(config_dict[group][key][ii]) + ','
                    setT += str(config_dict[group][key][-1])
                else :
                    setT = str(config_dict[group][key])
                self.labels_lineedits[group][key]['lineedit'].setText(setT)
                self.labels_lineedits[group][key]['lineedit'].editingFinished.connect(self.write_file)
                layout.addWidget(self.labels_lineedits[group][key]['lineedit'], i, 1, 1, 1)
                i += 1

    def write_file(self):
        with open(self.output_filename, 'w') as f:
            for group in self.labels_lineedits.keys():
                f.write('['+group+']' + '\n')
                
                for key in self.labels_lineedits[group].keys():
                    out_str = key
                    out_str = out_str + ' = '
                    out_str = out_str + str(self.labels_lineedits[group][key]['lineedit'].text())
                    f.write( out_str + '\n')

class Zernike_widget(QtGui.QWidget):
    def __init__(self, filename, config_dict):
        super(Zernike_widget, self).__init__()
        
        self.initUI(filename, config_dict)

    def initUI(self, filename, config_dict):
        # get the output directory
        self.output_dir = os.path.split(filename)[0]
        self.config_filename = os.path.join(self.output_dir, 'Zernike.ini')
        self.filename = filename
        self.f = h5py.File(filename, 'r')
        
        # Make a grid layout
        layout = QtGui.QGridLayout()
        
        # add the layout to the central widget
        self.setLayout(layout)
        
        # Zernike plot
        #############
        self.z_path = config_dict['Zernike']['h5_group']+'/Zernike_coefficients'
        title    = 'Zernike polynomial coefficients'
        self.z_plotsW = pg.PlotWidget(bottom='Noll index', left='coefficient', title = title)
         
        if self.z_path in self.f:
            z = self.f[self.z_path][()]
            self.z_plotsW.plot(z, pen=(255, 150, 150))
        self.f.close()
        
        # config widget
        ###############
        self.config_widget = Write_config_file_widget(config_dict, self.config_filename)

        # run command widget
        ####################
        self.run_command_widget = Run_and_log_command()
        self.run_command_widget.finished_signal.connect(self.finished)
        
        # run command button
        ####################
        self.run_button = QtGui.QPushButton('Calculate stitch', self)
        self.run_button.clicked.connect(self.run_button_clicked)
        
        # add a spacer for the labels and such
        verticalSpacer = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        
        # set the layout
        ################
        layout.addWidget(self.z_plotsW,            0, 1, 5, 1)
        layout.addWidget(self.config_widget,       0, 0, 1, 1)
        layout.addWidget(self.run_button,          1, 0, 1, 1)
        #layout.addWidget(self.ref_button,          2, 0, 1, 1)
        #layout.addWidget(self.set_button,          3, 0, 1, 1)
        layout.addItem(verticalSpacer,             4, 0, 1, 1)
        layout.addWidget(self.run_command_widget,  5, 0, 1, 2)
        #layout.addWidget(self.run_ref_widget,      6, 0, 1, 2)
        layout.setColumnStretch(1, 1)
        layout.setColumnMinimumWidth(0, 250)
        self.layout = layout

    def run_button_clicked(self):
        # write the config file 
        #######################
        self.config_widget.write_file()
    
        # Run the command 
        #################
        py = os.path.join(root, 'process/Zernike.py')
        cmd = 'python ' + py + ' ' + self.filename + ' -c ' + self.config_filename
        self.run_command_widget.run_cmd(cmd)
    
    def finished(self):
        self.f = h5py.File(self.filename, 'r')
        print(self.z_path)
        plotitem = self.z_plotsW.getPlotItem()
        plotitem.clear()
        plotitem.plot(self.f[self.z_path][()])
        self.f.close()

class Show_cpu_stitch_widget(QtGui.QWidget):
    def __init__(self, filename, config_dict):
        super(Show_cpu_stitch_widget, self).__init__()
        
        self.initUI(filename, config_dict)

    def initUI(self, filename, config_dict):
        """
        First show the stitch if there is one...
        
        We need: 
        an imageview on the left
        a config writer on the right
        below that a run button 
        below that a run and log command widget
        
        then wait for command to finish and show the image
        
        would be cool if we could show a scatter plot of the 
        positions on top of the stitch image.
        """
        # get the output directory
        self.output_dir = os.path.split(filename)[0]
        self.config_filename = os.path.join(self.output_dir, 'cpu_stitch.ini')
        self.filename = filename
        self.f = h5py.File(filename, 'r')
        
        # Make a grid layout
        layout = QtGui.QGridLayout()
        
        # add the layout to the central widget
        self.setLayout(layout)
        
        # stitch plot
        #############
        frame_plt = pg.PlotItem(title = 'Stitch with pixel shifts')
        self.imageView = pg.ImageView(view = frame_plt)
        self.imageView.ui.menuBtn.hide()
        self.imageView.ui.roiBtn.hide()
        self.stitch_path = config_dict['cpu_stitch']['h5_group']+'/O'
        #self.R_path = config_dict['stitch']['h5_group']+'/R'
        self.im_init = False
        if self.stitch_path in self.f :
            print(self.f[self.stitch_path].shape)
            t = self.f[self.stitch_path].value.T.real
            self.imageView.setImage(t)
            self.im_init = True
        self.f.close()
        
        # config widget
        ###############
        self.config_widget = Write_config_file_widget(config_dict, self.config_filename)

        # run command widget
        ####################
        self.run_command_widget = Run_and_log_command()
        self.run_command_widget.finished_signal.connect(self.finished)
        
        # run command button
        ####################
        self.run_button = QtGui.QPushButton('Calculate stitch', self)
        self.run_button.clicked.connect(self.run_button_clicked)
        
        # add a spacer for the labels and such
        verticalSpacer = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        
        # set the layout
        ################
        layout.addWidget(self.imageView,           0, 1, 5, 1)
        layout.addWidget(self.config_widget,       0, 0, 1, 1)
        layout.addWidget(self.run_button,          1, 0, 1, 1)
        #layout.addWidget(self.ref_button,          2, 0, 1, 1)
        #layout.addWidget(self.set_button,          3, 0, 1, 1)
        layout.addItem(verticalSpacer,             4, 0, 1, 1)
        layout.addWidget(self.run_command_widget,  5, 0, 1, 2)
        #layout.addWidget(self.run_ref_widget,      6, 0, 1, 2)
        layout.setColumnStretch(1, 1)
        layout.setColumnMinimumWidth(0, 250)
        self.layout = layout

    def run_button_clicked(self):
        # write the config file 
        #######################
        self.config_widget.write_file()
    
        # Run the command 
        #################
        py = os.path.join(root, 'process/cpu_stitch.py')
        cmd = 'python ' + py + ' ' + self.filename + ' -c ' + self.config_filename
        self.run_command_widget.run_cmd(cmd)
    
    def finished(self):
        self.f = h5py.File(self.filename, 'r')
        print(self.stitch_path)
        t = self.f[self.stitch_path].value.T.real
        if self.im_init :
            self.imageView.setImage(t, autoRange = False, autoLevels = False, autoHistogramRange = False)
            self.im_init = True
        else :
            self.imageView.setImage(t)
            self.im_init = True
        self.f.close()

class Show_EMC_widget(QtGui.QWidget):
    def __init__(self, filename, config_dict):
        super(Show_EMC_widget, self).__init__()
        
        self.initUI(filename, config_dict)

    def initUI(self, filename, config_dict):
        """
        First show the stitch if there is one...
        
        We need: 
        an imageview on the left
        a config writer on the right
        below that a run button 
        below that a run and log command widget
        
        then wait for command to finish and show the image
        
        would be cool if we could show a scatter plot of the 
        positions on top of the stitch image.
        """
        # get the output directory
        self.output_dir = os.path.split(filename)[0]
        self.config_filename = os.path.join(self.output_dir, 'EMC.ini')
        self.filename = filename
        self.f = h5py.File(filename, 'r')
        
        # Make a grid layout
        layout = QtGui.QGridLayout()
        
        # add the layout to the central widget
        self.setLayout(layout)
        
        # stitch plot
        #############
        frame_plt = pg.PlotItem(title = 'EMC merge')
        self.imageView = pg.ImageView(view = frame_plt)
        self.imageView.ui.menuBtn.hide()
        self.imageView.ui.roiBtn.hide()
        self.stitch_path = config_dict['EMC']['h5_group']+'/O'
        #self.R_path = config_dict['stitch']['h5_group']+'/R'
        self.im_init = False
        if self.stitch_path in self.f :
            print(self.f[self.stitch_path].shape)
            t = self.f[self.stitch_path].value.T.real
            self.imageView.setImage(t)
            self.im_init = True
        self.f.close()
        
        # config widget
        ###############
        self.config_widget = Write_config_file_widget(config_dict, self.config_filename)

        # run command widget
        ####################
        self.run_command_widget = Run_and_log_command()
        self.run_command_widget.finished_signal.connect(self.finished)
        
        # run command button
        ####################
        self.run_button = QtGui.QPushButton('Calculate stitch', self)
        self.run_button.clicked.connect(self.run_button_clicked)
        
        # add a spacer for the labels and such
        verticalSpacer = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        
        # set the layout
        ################
        layout.addWidget(self.imageView,           0, 1, 5, 1)
        layout.addWidget(self.config_widget,       0, 0, 1, 1)
        layout.addWidget(self.run_button,          1, 0, 1, 1)
        #layout.addWidget(self.ref_button,          2, 0, 1, 1)
        #layout.addWidget(self.set_button,          3, 0, 1, 1)
        layout.addItem(verticalSpacer,             4, 0, 1, 1)
        layout.addWidget(self.run_command_widget,  5, 0, 1, 2)
        #layout.addWidget(self.run_ref_widget,      6, 0, 1, 2)
        layout.setColumnStretch(1, 1)
        layout.setColumnMinimumWidth(0, 250)
        self.layout = layout

    def run_button_clicked(self):
        # write the config file 
        #######################
        self.config_widget.write_file()
    
        # Run the command 
        #################
        py = os.path.join(root, 'process/EMC.py')
        cmd = 'mpirun -np 16 python ' + py + ' ' + self.filename + ' -c ' + self.config_filename
        self.run_command_widget.run_cmd(cmd)
    
    def finished(self):
        self.f = h5py.File(self.filename, 'r')
        print(self.stitch_path)
        t = self.f[self.stitch_path].value.T.real
        if self.im_init :
            self.imageView.setImage(t, autoRange = False, autoLevels = False, autoHistogramRange = False)
            self.im_init = True
        else :
            self.imageView.setImage(t)
            self.im_init = True
        self.f.close()

class Show_make_pixel_shifts_widget(QtGui.QWidget):
    def __init__(self, filename, config_dict):
        super(Show_make_pixel_shifts_widget, self).__init__()
        
        self.initUI(filename, config_dict)

    def initUI(self, filename, config_dict):
        """
        First show the stitch if there is one...
        
        We need: 
        an imageview on the left
        a config writer on the right
        below that a run button 
        below that a run and log command widget
        
        then wait for command to finish and show the image
        
        would be cool if we could show a scatter plot of the 
        positions on top of the stitch image.
        """
        # get the output directory
        self.output_dir = os.path.split(filename)[0]
        self.config_filename = os.path.join(self.output_dir, 'make_pixel_shifts.ini')
        self.filename = filename
        self.f = h5py.File(filename, 'r')
        
        # Make a grid layout
        layout = QtGui.QGridLayout()
        
        # add the layout to the central widget
        self.setLayout(layout)
        
        # stitch plot
        #############
        self.plotW = pg.PlotWidget(title = 'pixel shifts ss / fs --> green / red')
        
        self.h5_path  = config_dict['make_pixel_shifts']['h5_group'] + '/1d_profile'
        self.h5_path2 = config_dict['make_pixel_shifts']['h5_group'] + '/1d_profile_plot'
        self.im_init = False
        if self.h5_path+'_fs' in self.f :
            fs = self.f[self.h5_path+'_fs'][()]
            ss = self.f[self.h5_path+'_ss'][()]
            self.plotW.plot(fs, pen=(255, 150, 150))
            self.plotW.plot(ss, pen=(150, 255, 150))
            
            if self.h5_path2+'_fs' in self.f :
                fs2 = self.f[self.h5_path2+'_fs'][()]
                ss2 = self.f[self.h5_path2+'_ss'][()]
                self.plotW.plot(fs2, pen=(255, 0, 0))
                self.plotW.plot(ss2, pen=(0, 255, 0))
            self.im_init = True
        self.f.close()
        
        # config widget
        ###############
        self.config_widget = Write_config_file_widget(config_dict, self.config_filename)

        # run command widget
        ####################
        self.run_command_widget = Run_and_log_command()
        self.run_command_widget.finished_signal.connect(self.finished)
        
        # run command button
        ####################
        self.run_button = QtGui.QPushButton('Calculate stitch', self)
        self.run_button.clicked.connect(self.run_button_clicked)
        
        # add a spacer for the labels and such
        verticalSpacer = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        
        # set the layout
        ################
        layout.addWidget(self.plotW,               0, 1, 5, 1)
        layout.addWidget(self.config_widget,       0, 0, 1, 1)
        layout.addWidget(self.run_button,          1, 0, 1, 1)
        #layout.addWidget(self.ref_button,          2, 0, 1, 1)
        #layout.addWidget(self.set_button,          3, 0, 1, 1)
        layout.addItem(verticalSpacer,             4, 0, 1, 1)
        layout.addWidget(self.run_command_widget,  5, 0, 1, 2)
        #layout.addWidget(self.run_ref_widget,      6, 0, 1, 2)
        layout.setColumnStretch(1, 1)
        layout.setColumnMinimumWidth(0, 250)
        self.layout = layout

    def run_button_clicked(self):
        # write the config file 
        #######################
        self.config_widget.write_file()
    
        # Run the command 
        #################
        py = os.path.join(root, 'process/make_pixel_shifts.py')
        cmd = 'python ' + py + ' ' + self.filename + ' -c ' + self.config_filename
        self.run_command_widget.run_cmd(cmd)
    
    def finished(self):
        self.f = h5py.File(self.filename, 'r')
        if self.im_init :
            self.plotW.clear()
        fs = self.f[self.h5_path+'_fs'][()]
        ss = self.f[self.h5_path+'_ss'][()]
        self.plotW.plot(fs, pen=(255, 150, 150))
        self.plotW.plot(ss, pen=(150, 255, 150))

        if self.h5_path2+'_fs' in self.f :
            fs2 = self.f[self.h5_path2+'_fs'][()]
            ss2 = self.f[self.h5_path2+'_ss'][()]
            self.plotW.plot(fs2, pen=(255, 0, 0))
            self.plotW.plot(ss2, pen=(0, 255, 0))
        self.f.close()

class Show_gpu_stitch_widget(QtGui.QWidget):
    def __init__(self, filename, config_dict):
        super(Show_gpu_stitch_widget, self).__init__()
        
        self.initUI(filename, config_dict)

    def initUI(self, filename, config_dict):
        """
        First show the stitch if there is one...
        
        We need: 
        an imageview on the left
        a config writer on the right
        below that a run button 
        below that a run and log command widget
        
        then wait for command to finish and show the image
        
        would be cool if we could show a scatter plot of the 
        positions on top of the stitch image.
        """
        # get the output directory
        self.output_dir = os.path.split(filename)[0]
        self.config_filename = os.path.join(self.output_dir, 'gpu_stitch.ini')
        self.filename = filename
        self.f = h5py.File(filename, 'r')
        
        # Make a grid layout
        layout = QtGui.QGridLayout()
        
        # add the layout to the central widget
        self.setLayout(layout)
        
        # stitch plot
        #############
        frame_plt = pg.PlotItem(title = 'Stitch with pixel shifts')
        self.imageView = pg.ImageView(view = frame_plt)
        self.imageView.ui.menuBtn.hide()
        self.imageView.ui.roiBtn.hide()
        self.stitch_path = config_dict['gpu_stitch']['h5_group']+'/O_gpu_stitch'
        #self.R_path = config_dict['stitch']['h5_group']+'/R'
        self.im_init = False
        if self.stitch_path in self.f :
            print(self.f[self.stitch_path].shape)
            t = self.f[self.stitch_path].value.T.real
            self.imageView.setImage(t)
            self.im_init = True
        self.f.close()
        
        # config widget
        ###############
        self.config_widget = Write_config_file_widget(config_dict, self.config_filename)

        # run command widget
        ####################
        self.run_command_widget = Run_and_log_command()
        self.run_command_widget.finished_signal.connect(self.finished)
        
        # run command button
        ####################
        self.run_button = QtGui.QPushButton('Calculate stitch', self)
        self.run_button.clicked.connect(self.run_button_clicked)
        
        # add a spacer for the labels and such
        verticalSpacer = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        
        # set the layout
        ################
        layout.addWidget(self.imageView,           0, 1, 5, 1)
        layout.addWidget(self.config_widget,       0, 0, 1, 1)
        layout.addWidget(self.run_button,          1, 0, 1, 1)
        #layout.addWidget(self.ref_button,          2, 0, 1, 1)
        #layout.addWidget(self.set_button,          3, 0, 1, 1)
        layout.addItem(verticalSpacer,             4, 0, 1, 1)
        layout.addWidget(self.run_command_widget,  5, 0, 1, 2)
        #layout.addWidget(self.run_ref_widget,      6, 0, 1, 2)
        layout.setColumnStretch(1, 1)
        layout.setColumnMinimumWidth(0, 250)
        self.layout = layout

    def run_button_clicked(self):
        # write the config file 
        #######################
        self.config_widget.write_file()
    
        # Run the command 
        #################
        py = os.path.join(root, 'process/gpu_stitch.py')
        cmd = 'python ' + py + ' ' + self.filename + ' -c ' + self.config_filename
        self.run_command_widget.run_cmd(cmd)
    
    def finished(self):
        self.f = h5py.File(self.filename, 'r')
        print(self.stitch_path)
        t = self.f[self.stitch_path].value.T.real
        if self.im_init :
            self.imageView.setImage(t, autoRange = False, autoLevels = False, autoHistogramRange = False)
            self.im_init = True
        else :
            self.imageView.setImage(t)
            self.im_init = True
        self.f.close()
    
    def set_button_clicked(self):
        pass
        """
        f = h5py.File(self.filename)
        O = f[self.stitch_path].value
        #R = f[self.R_path].value
        print('writing O to file')
        if 'O' in f :
            del f['O']
        f['O'] = O
        #if 'R' in f :
        #    del f['R']
        #f['R'] = R
        
        if 'metadata/defocus' in f :
            del f['metadata/defocus']
        f['metadata/defocus'] = self.config_widget.output_config['stitch']['defocus']
        print('Done')
        f.close()
        """

class Show_stitch_widget(QtGui.QWidget):
    def __init__(self, filename, config_dict):
        super(Show_stitch_widget, self).__init__()
        
        self.initUI(filename, config_dict)

    def initUI(self, filename, config_dict):
        """
        First show the stitch if there is one...
        
        We need: 
        an imageview on the left
        a config writer on the right
        below that a run button 
        below that a run and log command widget
        
        then wait for command to finish and show the image
        
        would be cool if we could show a scatter plot of the 
        positions on top of the stitch image.
        """
        # get the output directory
        self.output_dir = os.path.split(filename)[0]
        self.config_filename = os.path.join(self.output_dir, 'basic_stitch.ini')
        self.filename = filename
        self.f = h5py.File(filename, 'r')
        
        # Make a grid layout
        layout = QtGui.QGridLayout()
        
        # add the layout to the central widget
        self.setLayout(layout)
        
        # stitch plot
        #############
        frame_plt = pg.PlotItem(title = 'Stitch')
        self.imageView = pg.ImageView(view = frame_plt)
        self.imageView.ui.menuBtn.hide()
        self.imageView.ui.roiBtn.hide()
        self.stitch_path = config_dict['stitch']['h5_group']+'/O'
        #self.R_path = config_dict['stitch']['h5_group']+'/R'
        self.im_init = False
        if self.stitch_path in self.f :
            print(self.f[self.stitch_path].shape)
            t = self.f[self.stitch_path].value.T.real
            self.imageView.setImage(t)
            self.im_init = True
        self.f.close()
        
        # config widget
        ###############
        self.config_widget = Write_config_file_widget(config_dict, self.config_filename)

        # run command widget
        ####################
        self.run_command_widget = Run_and_log_command()
        self.run_command_widget.finished_signal.connect(self.finished)
        
        # run command button
        ####################
        self.run_button = QtGui.QPushButton('Calculate stitch', self)
        self.run_button.clicked.connect(self.run_button_clicked)
        
        # set sample and R
        ##################
        #self.set_button = QtGui.QPushButton('set: O, R and defocus', self)
        #self.set_button.clicked.connect(self.set_button_clicked)
        
        # refine positions command widget
        #################################
        #self.run_ref_widget = Run_and_log_command()
        #self.run_ref_widget.finished_signal.connect(self.ref_positions_done)
        
        # refine positions button
        #################################
        #self.ref_button = QtGui.QPushButton('refine positions', self)
        #self.ref_button.clicked.connect(self.ref_button_clicked)
        
        # add a spacer for the labels and such
        verticalSpacer = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        
        # set the layout
        ################
        layout.addWidget(self.imageView,           0, 1, 5, 1)
        layout.addWidget(self.config_widget,       0, 0, 1, 1)
        layout.addWidget(self.run_button,          1, 0, 1, 1)
        #layout.addWidget(self.ref_button,          2, 0, 1, 1)
        #layout.addWidget(self.set_button,          3, 0, 1, 1)
        layout.addItem(verticalSpacer,             4, 0, 1, 1)
        layout.addWidget(self.run_command_widget,  5, 0, 1, 2)
        #layout.addWidget(self.run_ref_widget,      6, 0, 1, 2)
        layout.setColumnStretch(1, 1)
        layout.setColumnMinimumWidth(0, 250)
        self.layout = layout

    def ref_button_clicked(self):
        py = os.path.join(root, 'process/position_refinement.py')
        cmd = 'python ' + py + ' ' + self.filename 
        self.run_ref_widget.run_cmd(cmd)

    def ref_positions_done(self):
        print('Done!')

    def run_button_clicked(self):
        # write the config file 
        #######################
        self.config_widget.write_file()
    
        # Run the command 
        #################
        py = os.path.join(root, 'process/basic_stitch.py')
        cmd = 'python ' + py + ' ' + self.filename + ' -c ' + self.config_filename
        self.run_command_widget.run_cmd(cmd)
    
    def finished(self):
        self.f = h5py.File(self.filename, 'r')
        print(self.stitch_path)
        t = self.f[self.stitch_path].value.T.real
        if self.im_init :
            self.imageView.setImage(t, autoRange = False, autoLevels = False, autoHistogramRange = False)
            self.im_init = True
        else :
            self.imageView.setImage(t)
            self.im_init = True
        self.f.close()
    
    def set_button_clicked(self):
        pass
        """
        f = h5py.File(self.filename)
        O = f[self.stitch_path].value
        #R = f[self.R_path].value
        print('writing O to file')
        if 'O' in f :
            del f['O']
        f['O'] = O
        #if 'R' in f :
        #    del f['R']
        #f['R'] = R
        
        if 'metadata/defocus' in f :
            del f['metadata/defocus']
        f['metadata/defocus'] = self.config_widget.output_config['stitch']['defocus']
        print('Done')
        f.close()
        """

class Defocus_widget(QtGui.QWidget):
    def __init__(self, filename, config_dict):
        super(Defocus_widget, self).__init__()
        
        self.initUI(filename, config_dict)

    def initUI(self, filename, config_dict):
        # get the output directory
        self.output_dir = os.path.split(filename)[0]
        self.config_filename = os.path.join(self.output_dir, 'defocus.ini')
        self.filename = filename
        self.f = h5py.File(filename, 'r')
        
        # Make a grid layout
        layout = QtGui.QGridLayout()
        
        # add the layout to the central widget
        self.setLayout(layout)
        
        # stitch plot
        #############
        frame_plt = pg.PlotItem(title = 'defocused probe')
        self.imageView = pg.ImageView(view = frame_plt)
        self.imageView.ui.menuBtn.hide()
        self.imageView.ui.roiBtn.hide()
        self.stitch_path = config_dict['defocus']['h5_group']+'/probe_df'
        #self.R_path = config_dict['stitch']['h5_group']+'/R'
        self.im_init = False
        if self.stitch_path in self.f :
            print(self.f[self.stitch_path].shape)
            t = np.abs(self.f[self.stitch_path].value.T)**2
            self.imageView.setImage(t)
            self.im_init = True
        self.f.close()
        
        # config widget
        ###############
        self.config_widget = Write_config_file_widget(config_dict, self.config_filename)

        # run command widget
        ####################
        self.run_command_widget = Run_and_log_command()
        self.run_command_widget.finished_signal.connect(self.finished)
        
        # run command button
        ####################
        self.run_button = QtGui.QPushButton('Calculate defocused probe', self)
        self.run_button.clicked.connect(self.run_button_clicked)
        
        # add a spacer for the labels and such
        verticalSpacer = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        
        # set the layout
        ################
        layout.addWidget(self.imageView,           0, 1, 5, 1)
        layout.addWidget(self.config_widget,       0, 0, 1, 1)
        layout.addWidget(self.run_button,          1, 0, 1, 1)
        #layout.addWidget(self.ref_button,          2, 0, 1, 1)
        #layout.addWidget(self.set_button,          3, 0, 1, 1)
        layout.addItem(verticalSpacer,             4, 0, 1, 1)
        layout.addWidget(self.run_command_widget,  5, 0, 1, 2)
        #layout.addWidget(self.run_ref_widget,      6, 0, 1, 2)
        layout.setColumnStretch(1, 1)
        layout.setColumnMinimumWidth(0, 250)
        self.layout = layout

    def run_button_clicked(self):
        # write the config file 
        #######################
        self.config_widget.write_file()
    
        # Run the command 
        #################
        py = os.path.join(root, 'process/defocus.py')
        cmd = 'python ' + py + ' ' + self.filename + ' -c ' + self.config_filename
        self.run_command_widget.run_cmd(cmd)
    
    def finished(self):
        self.f = h5py.File(self.filename, 'r')
        print(self.stitch_path)
        t = np.abs(self.f[self.stitch_path].value.T)**2
        if self.im_init :
            self.imageView.setImage(t, autoRange = False, autoLevels = False, autoHistogramRange = False)
            self.im_init = True
        else :
            self.imageView.setImage(t)
            self.im_init = True
        self.f.close()

class Show_frames_widget(QtGui.QWidget):
    def __init__(self, filename, config = config_default):
        super(Show_frames_widget, self).__init__()
        
        self.filename = filename
        self.config   = config
        self.initUI()

    def initUI(self):
        """
        the frame view has fs --> left and ss --> down
        so we should scatter plot the coordinates accourdingly
        """
        # Make a grid layout
        layout = QtGui.QGridLayout()
        
        # add the layout to the central widget
        self.setLayout(layout)
        
        # Now we can add widgets to the layout
        #win = pg.GraphicsWindow(title="results")
        #layout.addWidget(win, 0, 0, 1, 1)
        f = h5py.File(self.filename, 'r')
        
        # X and Y plot 
        ##############
        R = f[self.config['input']['R']]
        X = R[:, 0]
        Y = R[:, 0]
        title = 'realspace x (red) and y (green) sample positions in pixel units'
        position_plotsW = pg.PlotWidget(bottom='frame number', left='position', title = title)
        position_plotsW.plot(X, pen=(255, 150, 150))
        position_plotsW.plot(Y, pen=(150, 255, 150))
        
        # vline
        vline = position_plotsW.addLine(x = 0, movable=True, bounds = [0, len(X)-1])

        # frame plot
        ############
        frame_plt = pg.PlotItem(title = 'Frame View')
        imageView = pg.ImageView(view = frame_plt)
        imageView.ui.menuBtn.hide()
        imageView.ui.roiBtn.hide()
        imageView.setImage(f[self.config['input']['data']][0].T.astype(np.float))
        #imageView.show()

        # scatter plot
        ##############
        ## 1: X/YPZT
        colour = 'r'
        brush  = pg.mkBrush(colour)
        s1     = pg.ScatterPlotItem(size = 2, pxMode=True)
        spot_sizes = np.ones((len(X),), dtype=float)*2
        spots1 = []
        n      = len(X)
        #scale  = data['quality']
        for i in range(n):
            spots1.append({'pos':(X[i], Y[i]),
                           'brush':brush})
        s1.addPoints(spots1)

        s2     = pg.ScatterPlotItem(size = 10, pxMode=True, brush = pg.mkBrush('b'))

        scatter_plot = pg.PlotWidget(title='x,y scatter plot', left='y position', bottom='x position')
        scatter_plot.addItem(s1)
        scatter_plot.addItem(s2)
        
        layout.addWidget(imageView      , 0, 0, 1, 1)
        layout.addWidget(scatter_plot   , 0, 1, 1, 1)
        layout.addWidget(position_plotsW, 1, 0, 1, 2)
        layout.setColumnMinimumWidth(0, 800)
        layout.setRowMinimumHeight(0, 500)

        j = 0
        def replot_frame():
            i = int(vline.value())
            s2.setData([X[i]], [Y[i]])

            f = h5py.File(self.filename, 'r')
            imageView.setImage( f[self.config['input']['data']][i].T.astype(np.float), autoRange = False, autoLevels = False, autoHistogramRange = False)
            f.close()
            
        vline.sigPositionChanged.connect(replot_frame)
        f.close()

class Show_frames_selection_widget(QtGui.QWidget):
    def __init__(self, filename):
        super(Show_frames_selection_widget, self).__init__()
        
        self.filename = filename
        self.initUI()

    def initUI(self):
        # Make a grid layout
        layout = QtGui.QGridLayout()
        
        # add the layout to the central widget
        self.setLayout(layout)
        
        # Now we can add widgets to the layout
        #win = pg.GraphicsWindow(title="results")
        #layout.addWidget(win, 0, 0, 1, 1)
        f = h5py.File(self.filename, 'r')
        
        # X and Y plot 
        ##############
        R = f[config_default['input']['R']]
        X = R[:, 0]
        Y = R[:, 1]
        title = 'realspace x (red) and y (green) sample positions in pixel units'
        position_plotsW = pg.PlotWidget(bottom='frame number', left='position', title = title)
        position_plotsW.plot(X, pen=(255, 150, 150))
        position_plotsW.plot(Y, pen=(150, 255, 150))
        
        # vline
        vline = position_plotsW.addLine(x = 0, movable=True, bounds = [0, len(X)-1])

        # frame plot
        ############
        self.whitefield  = f[config_default['input']['whitefield']][()]
        self.whitefield[self.whitefield==0] = 1.
        self.whitefield  = self.whitefield.astype(np.float) / float(f[config_default['input']['data']].shape[0])

        self.mkframe = lambda i, f : f[config_default['input']['data']][i] / self.whitefield
        
        frame_plt = pg.PlotItem(title = 'Frame View')
        imageView = pg.ImageView(view = frame_plt)
        imageView.ui.menuBtn.hide()
        imageView.ui.roiBtn.hide()
        imageView.setImage(self.mkframe(0, f).T.astype(np.float))
        #imageView.show()

        # scatter plot
        ##############
        ## 1: X/YPZT
        self.scatter_plot = Select_frames_widget(self.filename)
        
        layout.addWidget(imageView      , 0, 0, 1, 1)
        layout.addWidget(self.scatter_plot   , 0, 1, 1, 1)
        layout.addWidget(position_plotsW, 1, 0, 1, 2)
        layout.setColumnMinimumWidth(0, 800)
        layout.setRowMinimumHeight(0, 500)

        j = 0
        def replot_frame():
            i = int(vline.value())
            self.scatter_plot.replot(i)

            f = h5py.File(self.filename, 'r')
            imageView.setImage( self.mkframe(i, f).T.astype(np.float), autoRange = False, autoLevels = False, autoHistogramRange = False)
            f.close()
            
        vline.sigPositionChanged.connect(replot_frame)
        f.close()

class Select_frames_widget(QtGui.QWidget):
    """
    Draw a scatter plot of the X-Y coordinates in f[R]
    """
        
    def __init__(self, filename):
        super(Select_frames_widget, self).__init__()
        
        self.filename = filename
        self.frames = []
        self.initUI()

    def initUI(self):
        # Make a grid layout
        layout = QtGui.QGridLayout()
        
        # add the layout to the central widget
        self.setLayout(layout)
        
        # Now we can add widgets to the layout
        f = h5py.File(self.filename, 'r')

        # Get the X and Y coords
        ########################
        R = f[config_default['input']['R']]
        X = R[:, 0]
        Y = R[:, 1]
        self.X = X
        self.Y = Y
        
        self.frames = np.zeros((len(X),), dtype=bool)
        self.frames[f[config_default['output']]['good_frames'][()]] = True
        
        # scatter plot
        ##############
        self.good_frame_pen     = pg.mkPen((255, 150, 150))
        self.bad_frame_pen      = pg.mkPen(None)
        #self.selected_frame_pen = pg.mkPen((150, 150, 255))
        
        self.s1    = pg.ScatterPlotItem(size=5, pen=self.good_frame_pen, brush=pg.mkBrush(255, 255, 255, 120))
        spots = [{'pos': [X[i], Y[i]], 'data': i} for i in range(len(R))] 
        self.s1.addPoints(spots)

        # Temp
        ######
        if 'process_3/cpu_stitch/R' in f :
            R = f['process_3/cpu_stitch/R']
            X2 = R[:, 0]
            Y2 = R[:, 1]
            self.X2 = X2
            self.Y2 = Y2
            
            # scatter plot
            ##############
            self.good_frame_pen2     = pg.mkPen((150, 255, 150))
            
            self.s12    = pg.ScatterPlotItem(size=5, pen=self.good_frame_pen2, brush=pg.mkBrush(255, 255, 255, 120))
            spots2 = [{'pos': [X2[i], Y2[i]], 'data': i} for i in range(len(R))] 
            self.s12.addPoints(spots2)
        else :
            self.s12 = None

        ## Make all plots clickable
        def clicked(plot, points):
            for p in points:
                self.frames[p.data()] = ~self.frames[p.data()]
                if self.frames[p.data()] :
                    p.setPen(self.good_frame_pen)
                else :
                    p.setPen(self.bad_frame_pen)

        self.s1.sigClicked.connect(clicked)

        self.update_selected_points()

        ## Show the selected frame
        ##########################
        self.s2     = pg.ScatterPlotItem(size = 10, pxMode=True, brush = pg.mkBrush('b'))

        ## rectangular ROI selection
        ############################
        # put it out of the way
        span    = [0.1 * (X.max()-X.min()), 0.1 * (Y.max()-Y.min())]
        courner = [X.min()-1.5*span[0], Y.min()-1.5*span[1]]
        self.roi = pg.RectROI(courner, span)
        self.roi.setZValue(10)                       # make sure ROI is drawn above image
        ROI_button_good   = QtGui.QPushButton('good frames')
        ROI_button_bad    = QtGui.QPushButton('bad frames')
        ROI_button_toggle = QtGui.QPushButton('toggle frames')
        write_button      = QtGui.QPushButton('write to file')
        ROI_button_good.clicked.connect(   lambda : self.mask_ROI(self.roi, 0))
        ROI_button_bad.clicked.connect(    lambda : self.mask_ROI(self.roi, 1))
        ROI_button_toggle.clicked.connect( lambda : self.mask_ROI(self.roi, 2))
        write_button.clicked.connect(               self.write_good_frames)
        
        scatter_plot = pg.PlotWidget(title='x,y scatter plot', left='y position', bottom='x position')
        scatter_plot.addItem(self.roi)
        scatter_plot.addItem(self.s1)
        scatter_plot.addItem(self.s2)

        if self.s12 is not None :
            scatter_plot.addItem(self.s12)
        
        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(ROI_button_good)
        hbox.addWidget(ROI_button_bad)
        hbox.addWidget(ROI_button_toggle)
        hbox.addWidget(write_button)
        hbox.addStretch(1)
        
        layout.addWidget(scatter_plot   , 0, 0, 1, 1)
        layout.addLayout(hbox           , 1, 0, 1, 1)
        
        f.close()

    def write_good_frames(self):
        f = h5py.File(self.filename, 'a')
        key = config_default['output'] + '/good_frames'
        if key in f :
            del f[key]
        f[key] = np.where(self.frames)[0]
        f.close()

    def update_selected_points(self):
        pens = [self.good_frame_pen if f else self.bad_frame_pen for f in self.frames]
        self.s1.setPen(pens)

    def replot(self, frame):
        self.s2.setData([self.X[frame]], [self.Y[frame]])

    def mask_ROI(self, roi, good_bad_toggle = 0):
        sides   = [roi.size()[0], roi.size()[1]]
        courner = [roi.pos()[0], roi.pos()[1]]
        
        top_right   = [courner[0] + sides[0], courner[1] + sides[1]]
        bottom_left = courner
        
        y_in_rect = (self.Y <= top_right[1])   & (self.Y >= bottom_left[1])
        x_in_rect = (self.X >= bottom_left[0]) & (self.X <= top_right[0])
        
        if good_bad_toggle == 0 :
            self.frames[ y_in_rect * x_in_rect ] = True
        elif good_bad_toggle == 1 :
            self.frames[ y_in_rect * x_in_rect ] = False
        elif good_bad_toggle == 2 :
            self.frames[ y_in_rect * x_in_rect ] = ~self.frames[ y_in_rect * x_in_rect ]
    
        self.update_selected_points()

class Run_and_log_command(QtGui.QWidget):
    """
    run a command and send a signal when it complete, or it has failed.

    use a Qt timer to check the process
    
    realtime streaming of the terminal output has so proved to be fruitless
    """
    finished_signal = QtCore.pyqtSignal(bool)
    
    def __init__(self):
        super(Run_and_log_command, self).__init__()
        
        self.polling_interval = 0.1
        self.initUI()
        
    def initUI(self):
        """
        Just setup a qlabel showing the shell command
        and another showing the status of the process
        """
        # Make a grid layout
        #layout = QtGui.QGridLayout()
        hbox = QtGui.QHBoxLayout()
        
        # add the layout to the central widget
        self.setLayout(hbox)
        
        # show the command being executed
        self.command_label0 = QtGui.QLabel(self)
        self.command_label0.setText('<b>Command:</b>')
        self.command_label  = QtGui.QLabel(self)
        #self.command_label.setMaximumSize(50, 250)
         
        # show the status of the command
        self.status_label0  = QtGui.QLabel(self)
        self.status_label0.setText('<b>Status:</b>')
        self.status_label   = QtGui.QLabel(self)
        
        # add to layout
        hbox.addWidget(self.status_label0)
        hbox.addWidget(self.status_label)
        hbox.addWidget(self.command_label0)
        hbox.addWidget(self.command_label)
        hbox.addStretch(1)

        #layout.addWidget(self.status_label0,  0, 0, 1, 1)
        #layout.addWidget(self.status_label,   0, 1, 1, 1)
        #layout.addWidget(self.command_label0, 1, 0, 1, 1)
        #layout.addWidget(self.command_label,  1, 1, 1, 1)
         
    def run_cmd(self, cmd):
        from subprocess import PIPE, Popen
        import shlex
        self.command_label.setText(cmd)
        self.status_label.setText('running the command')
        self.p = Popen(shlex.split(cmd), stdout = PIPE, stderr = PIPE)
        
        # start a Qt timer to update the status
        QtCore.QTimer.singleShot(self.polling_interval, self.update_status)
    
    def update_status(self):
        status = self.p.poll()
        if status is None :
            self.status_label.setText('Running')
             
            # start a Qt timer to update the status
            QtCore.QTimer.singleShot(self.polling_interval, self.update_status)
        
        elif status is 0 :
            self.status_label.setText('Finished')
            
            # get the output and error msg
            self.output, self.err_msg = self.p.communicate()
            
            # emmit a signal when complete
            self.finished_signal.emit(True)
            print('Output   :', self.output)
            
        else :
            self.status_label.setText(str(status))
            
            # get the output and error msg
            self.output, self.err_msg = self.p.communicate()
            print('Output   :', self.output)
            print('Error msg:', self.err_msg)
            
            # emmit a signal when complete
            self.finished_signal.emit(False)

class Mask_maker_widget(QtGui.QWidget):
    """
    """
    cspad_psana_shape = (4, 8, 185, 388)
    cspad_geom_shape  = (1480, 1552)

    def __init__(self, fnam, mask = None, output_file=None, output_path=None):
        super(Mask_maker_widget, self).__init__()

        f = h5py.File(fnam, 'r')
        
        #cspad = f['whitefield'][()]
        cspad = np.sum(f['/entry_1/data_1/data'][:10], axis=0)

        if mask not in f :
            mask = np.ones_like(cspad).astype(np.bool)
        else :
            mask = f[mask][()]
        f.close()
        
        # this is not in fact a cspad image
        self.cspad_shape_flag = 'other'
        self.cspad = cspad

        if output_file is not None :
            self.output_file = output_file
        else :
            self.output_file = 'mask.h5'

        if output_path is not None :
            self.output_path = output_path
        else :
            self.output_path = 'data/data'

        self.mask  = np.ones_like(self.cspad, dtype=np.bool)
        self.geom_fnam = None
        
        i, j = np.meshgrid(range(self.cspad.shape[0]), range(self.cspad.shape[1]), indexing='ij')
        self.y_map, self.x_map = (i-self.cspad.shape[0]//2, j-self.cspad.shape[1]//2)
        self.cspad_shape = self.cspad.shape
        
        self.mask_edges    = False
        self.mask_unbonded = False

        self.unbonded_pixels = self.make_unbonded_pixels()
        self.asic_edges      = self.make_asic_edges()
        if mask is not None :
            self.mask_clicked  = mask
        else :
            self.mask_clicked  = np.ones_like(self.mask)
        
        self.initUI()
        
    def updateDisplayRGB(self, auto = False):
        """
        Make an RGB image (N, M, 3) (pyqt will interprate this as RGB automatically)
        with masked pixels shown in blue at the maximum value of the cspad. 
        This ensures that the masked pixels are shown at full brightness.
        """
        trans      = np.fliplr(self.cspad.T)
        trans_mask = np.fliplr(self.mask.T)
        self.cspad_max  = self.cspad.max()

        # convert to RGB
        # Set masked pixels to B
        display_data = np.zeros((trans.shape[0], trans.shape[1], 3), dtype = self.cspad.dtype)
        display_data[:, :, 0] = trans * trans_mask
        display_data[:, :, 1] = trans * trans_mask
        display_data[:, :, 2] = trans + (self.cspad_max - trans) * ~trans_mask
        
        self.display_RGB = display_data
        if auto :
            self.plot.setImage(self.display_RGB)
        else :
            self.plot.setImage(self.display_RGB, autoRange = False, autoLevels = False, autoHistogramRange = False)

    def generate_mask(self):
        self.mask.fill(1)

        if self.mask_unbonded :
            self.mask *= self.unbonded_pixels

        if self.mask_edges :
            self.mask *= self.asic_edges

        self.mask *= self.mask_clicked

    def update_mask_unbonded(self, state):
        if state > 0 :
            print('adding unbonded pixels to the mask')
            self.mask_unbonded = True
        else :
            print('removing unbonded pixels from the mask')
            self.mask_unbonded = False
        
        self.generate_mask()
        self.updateDisplayRGB()

    def update_mask_edges(self, state):
        if state > 0 :
            print('adding asic edges to the mask')
            self.mask_edges = True
        else :
            print('removing asic edges from the mask')
            self.mask_edges = False
        
        self.generate_mask()
        self.updateDisplayRGB()

    def save_mask(self):
        print('updating mask...')
        self.generate_mask()

        mask = self.mask
        
        print('outputing mask as np.int16 (h5py does not support boolean arrays yet)...')
        f = h5py.File(self.output_file)
        if self.output_path in f :
            del f[self.output_path]
        f[self.output_path] = mask
        f.close()
        print('Done!')
        
    def mask_ROI(self, roi):
        sides   = [roi.size()[1], roi.size()[0]]
        courner = [self.cspad_shape[0]/2. - roi.pos()[1], \
                   roi.pos()[0] - self.cspad_shape[1]/2.]

        top_left     = [np.rint(courner[0]) - 1, np.rint(courner[1])]
        bottom_right = [np.rint(courner[0] - sides[0]), np.rint(courner[1] + sides[1]) - 1]

        y_in_rect = (self.y_map <= top_left[0]) & (self.y_map >= bottom_right[0])
        x_in_rect = (self.x_map >= top_left[1]) & (self.x_map <= bottom_right[1])
        i2, j2 = np.where( y_in_rect & x_in_rect )
        self.apply_ROI(i2, j2)

    def mask_ROI_circle(self, roi):
        # get the xy coords of the centre and the radius
        rad    = roi.size()[0]/2. + 0.5
        centre = [self.cspad_shape[0]/2. - roi.pos()[1] - rad, \
                  roi.pos()[0] + rad - self.cspad_shape[1]/2.]
        
        r_map = np.sqrt((self.y_map-centre[0])**2 + (self.x_map-centre[1])**2)
        i2, j2 = np.where( r_map <= rad )
        self.apply_ROI(i2, j2)

    def apply_ROI(self, i2, j2):
        if self.toggle_checkbox.isChecked():
            self.mask_clicked[i2, j2] = ~self.mask_clicked[i2, j2]
        elif self.mask_checkbox.isChecked():
            self.mask_clicked[i2, j2] = False
        elif self.unmask_checkbox.isChecked():
            self.mask_clicked[i2, j2] = True
        
        self.generate_mask()
        self.updateDisplayRGB()
    
    def mask_hist(self):
        min_max = self.plot.getHistogramWidget().item.getLevels()
        
        if self.toggle_checkbox.isChecked():
            self.mask_clicked[np.where(self.cspad < min_max[0])] = ~self.mask_clicked[np.where(self.cspad < min_max[0])]
            self.mask_clicked[np.where(self.cspad > min_max[1])] = ~self.mask_clicked[np.where(self.cspad > min_max[1])]
        elif self.mask_checkbox.isChecked():
            self.mask_clicked[np.where(self.cspad < min_max[0])] = False
            self.mask_clicked[np.where(self.cspad > min_max[1])] = False
        elif self.unmask_checkbox.isChecked():
            self.mask_clicked[np.where(self.cspad < min_max[0])] = True
            self.mask_clicked[np.where(self.cspad > min_max[1])] = True
        
        self.generate_mask()
        self.updateDisplayRGB()

    def initUI(self):
        ## 2D plot for the cspad and mask
        #################################
        self.plot = pg.ImageView()

        ## save mask button
        #################################
        save_button = QtGui.QPushButton('save mask')
        save_button.clicked.connect(self.save_mask)

        # rectangular ROI selection
        #################################
        self.roi = pg.RectROI([-200,-200], [100, 100])
        self.plot.addItem(self.roi)
        self.roi.setZValue(10)                       # make sure ROI is drawn above image
        ROI_button = QtGui.QPushButton('mask rectangular ROI')
        ROI_button.clicked.connect(lambda : self.mask_ROI(self.roi))

        # circular ROI selection
        #################################
        self.roi_circle = pg.CircleROI([-200,200], [101, 101])
        self.plot.addItem(self.roi_circle)
        self.roi.setZValue(10)                       # make sure ROI is drawn above image
        ROI_circle_button = QtGui.QPushButton('mask circular ROI')
        ROI_circle_button.clicked.connect(lambda : self.mask_ROI_circle(self.roi_circle))

        # histogram mask button
        #################################
        hist_button = QtGui.QPushButton('mask outside histogram')
        hist_button.clicked.connect(self.mask_hist)

        # toggle / mask / unmask checkboxes
        #################################
        self.toggle_checkbox   = QtGui.QCheckBox('toggle')
        self.mask_checkbox     = QtGui.QCheckBox('mask')
        self.unmask_checkbox   = QtGui.QCheckBox('unmask')
        self.toggle_checkbox.setChecked(True)   
        
        self.toggle_group      = QtGui.QButtonGroup()#"masking behaviour")
        self.toggle_group.addButton(self.toggle_checkbox)   
        self.toggle_group.addButton(self.mask_checkbox)   
        self.toggle_group.addButton(self.unmask_checkbox)   
        self.toggle_group.setExclusive(True)
        
        # toggle / mask / unmask checkboxes
        #################################
        self.toggle_checkbox   = QtGui.QCheckBox('toggle')
        self.mask_checkbox     = QtGui.QCheckBox('mask')
        self.unmask_checkbox   = QtGui.QCheckBox('unmask')
        self.toggle_checkbox.setChecked(True)   
        
        self.toggle_group      = QtGui.QButtonGroup()#"masking behaviour")
        self.toggle_group.addButton(self.toggle_checkbox)   
        self.toggle_group.addButton(self.mask_checkbox)   
        self.toggle_group.addButton(self.unmask_checkbox)   
        self.toggle_group.setExclusive(True)

        # mouse hover ij value label
        #################################
        ij_label = QtGui.QLabel()
        disp = 'ss fs {0:5} {1:5}   value {2:2}'.format('-', '-', '-')
        ij_label.setText(disp)
        self.plot.scene.sigMouseMoved.connect( lambda pos: self.mouseMoved(ij_label, pos) )
        
        # unbonded pixels checkbox
        #################################
        unbonded_checkbox = QtGui.QCheckBox('unbonded pixels')
        unbonded_checkbox.stateChanged.connect( self.update_mask_unbonded )
        if self.cspad_shape_flag == 'other' :
            unbonded_checkbox.setEnabled(False)
        
        # asic edges checkbox
        #################################
        edges_checkbox = QtGui.QCheckBox('asic edges')
        edges_checkbox.stateChanged.connect( self.update_mask_edges )
        if self.cspad_shape_flag == 'other' :
            edges_checkbox.setEnabled(False)
        
        # mouse click mask 
        #################################
        self.plot.scene.sigMouseClicked.connect( lambda click: self.mouseClicked(self.plot, click) )

        # Create a grid layout to manage the widgets size and position
        #################################
        layout = QtGui.QGridLayout()
        self.setLayout(layout)

        ## Add widgets to the layout in their proper positions
        layout.addWidget(save_button, 0, 0)             # upper-left
        layout.addWidget(ROI_button, 1, 0)              # upper-left
        layout.addWidget(ROI_circle_button, 2, 0)       # upper-left
        layout.addWidget(hist_button, 3, 0)             # upper-left
        layout.addWidget(self.toggle_checkbox, 4, 0)    # upper-left
        layout.addWidget(self.mask_checkbox, 5, 0)      # upper-left
        layout.addWidget(self.unmask_checkbox, 6, 0)    # upper-left
        layout.addWidget(ij_label, 7, 0)                # upper-left
        layout.addWidget(unbonded_checkbox, 8, 0)       # middle-left
        layout.addWidget(edges_checkbox, 9, 0)          # bottom-left
        layout.addWidget(self.plot, 0, 1, 9, 1)         # plot goes on right side, spanning 3 rows
        layout.setColumnStretch(1, 1)
        layout.setColumnMinimumWidth(0, 250)
        
        # display the image
        self.generate_mask()
        self.updateDisplayRGB(auto = True)

    def mouseMoved(self, ij_label, pos):
        img = self.plot.getImageItem()
        if self.geom_fnam is not None :
            ij = [self.cspad_shape[0] - 1 - int(img.mapFromScene(pos).y()), int(img.mapFromScene(pos).x())] # ss, fs
            if (0 <= ij[0] < self.cspad_shape[0]) and (0 <= ij[1] < self.cspad_shape[1]):
                ij_label.setText('ss fs value: %d %d %.2e' % (self.ss_geom[ij[0], ij[1]], self.fs_geom[ij[0], ij[1]], self.cspad_geom[ij[0], ij[1]]) )
        else :
            ij = [self.cspad.shape[0] - 1 - int(img.mapFromScene(pos).y()), int(img.mapFromScene(pos).x())] # ss, fs
            if (0 <= ij[0] < self.cspad.shape[0]) and (0 <= ij[1] < self.cspad.shape[1]):
                ij_label.setText('ss fs value: %d %d %.2e' % (ij[0], ij[1], self.cspad[ij[0], ij[1]]) )

    def mouseClicked(self, plot, click):
        if click.button() == 1:
            img = plot.getImageItem()
            i0 = int(img.mapFromScene(click.pos()).y())
            j0 = int(img.mapFromScene(click.pos()).x())
            i1 = self.cspad.shape[0] - 1 - i0 # array ss (with the fliplr and .T)
            j1 = j0                           # array fs (with the fliplr and .T)
            if (0 <= i1 < self.cspad.shape[0]) and (0 <= j1 < self.cspad.shape[1]):
                if self.toggle_checkbox.isChecked():
                    self.mask_clicked[i1, j1] = ~self.mask_clicked[i1, j1]
                    self.mask[i1, j1]         = ~self.mask[i1, j1]
                elif self.mask_checkbox.isChecked():
                    self.mask_clicked[i1, j1] = False
                    self.mask[i1, j1]         = False
                elif self.unmask_checkbox.isChecked():
                    self.mask_clicked[i1, j1] = True
                    self.mask[i1, j1]         = True
                if self.mask[i1, j1] :
                    self.display_RGB[j0, i0, :] = np.array([1,1,1]) * self.cspad[i1, j1]
                else :
                    self.display_RGB[j0, i0, :] = np.array([0,0,1]) * self.cspad_max
            
            self.plot.setImage(self.display_RGB, autoRange = False, autoLevels = False, autoHistogramRange = False)
    
    def make_unbonded_pixels(self):
        cspad_psana_shape = self.cspad_psana_shape
        cspad_geom_shape  = self.cspad_geom_shape

        def ijkl_to_ss_fs(cspad_ijkl):
            """ 
            0: 388        388: 2 * 388  2*388: 3*388  3*388: 4*388
            (0, 0, :, :)  (1, 0, :, :)  (2, 0, :, :)  (3, 0, :, :)
            (0, 1, :, :)  (1, 1, :, :)  (2, 1, :, :)  (3, 1, :, :)
            (0, 2, :, :)  (1, 2, :, :)  (2, 2, :, :)  (3, 2, :, :)
            ...           ...           ...           ...
            (0, 7, :, :)  (1, 7, :, :)  (2, 7, :, :)  (3, 7, :, :)
            """
            if cspad_ijkl.shape != cspad_psana_shape :
                raise ValueError('cspad input is not the required shape:' + str(cspad_psana_shape) )

            cspad_ij = np.zeros(cspad_geom_shape, dtype=cspad_ijkl.dtype)
            for i in range(4):
                cspad_ij[:, i * cspad_psana_shape[3]: (i+1) * cspad_psana_shape[3]] = cspad_ijkl[i].reshape((cspad_psana_shape[1] * cspad_psana_shape[2], cspad_psana_shape[3]))

            return cspad_ij

        mask = np.ones(cspad_psana_shape)

        for q in range(cspad_psana_shape[0]):
            for p in range(cspad_psana_shape[1]):
                for a in range(2):
                    for i in range(19):
                        mask[q, p, i * 10, i * 10] = 0
                        mask[q, p, i * 10, i * 10 + cspad_psana_shape[-1]//2] = 0

        mask_slab = ijkl_to_ss_fs(mask)

        import scipy.signal
        kernal = np.array([ [0,1,0], [1,1,1], [0,1,0] ], dtype=np.float)
        mask_pad = scipy.signal.convolve(1 - mask_slab.astype(np.float), kernal, mode = 'same') < 1
        return mask_pad

    def make_asic_edges(self, arrayin = None, pad = 0):
        mask_edges = np.ones(self.cspad_geom_shape, dtype=np.bool)
        mask_edges[:: 185, :] = 0
        mask_edges[184 :: 185, :] = 0
        mask_edges[:, :: 194] = 0
        mask_edges[:, 193 :: 194] = 0

        if pad != 0 :
            mask_edges = scipy.signal.convolve(1 - mask_edges.astype(np.float), np.ones((pad, pad), dtype=np.float), mode = 'same') < 1
        return mask_edges

    def edges(self, shape, pad = 0):
        mask_edges = np.ones(shape)
        mask_edges[0, :]  = 0
        mask_edges[-1, :] = 0
        mask_edges[:, 0]  = 0
        mask_edges[:, -1] = 0

        if pad != 0 :
            mask_edges = scipy.signal.convolve(1 - mask_edges.astype(np.float), np.ones((pad, pad), dtype=np.float), mode = 'same') < 1
        return mask_edges

def run_and_log_command():
    signal.signal(signal.SIGINT, signal.SIG_DFL) # allow Control-C
    app = QtGui.QApplication([])
    
    # Qt main window
    Mwin = QtGui.QMainWindow()
    Mwin.setWindowTitle('run and log command')
    
    cw = Run_and_log_command()
    
    # add the central widget to the main window
    Mwin.setCentralWidget(cw)
    
    print('running command')
    import time
    time.sleep(1)
    cw.run_cmd('mpirun -n 4 python test.py')

    def print_finished(x):
        if x :
            print('Finished!')
        else :
            print('Something went wrong...')

    cw.finished_signal.connect(print_finished)

    print('app exec')
    Mwin.show()
    app.exec_()


if __name__ == '__main__':
    run_and_log_command()
