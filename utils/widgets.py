#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys, os
import numpy as np
import h5py
import scipy.constants as sc

import pyqtgraph as pg
import PyQt4.QtGui
import PyQt4.QtCore
import signal
import copy 

root = os.path.split(os.path.abspath(__file__))[0]
root = os.path.split(root)[0]

class Write_config_file_widget(PyQt4.QtGui.QWidget):
    def __init__(self, config_dict, output_filename):
        super(Write_config_file_widget, self).__init__()
        
        self.output_filename = output_filename
        self.initUI(config_dict)
    
    def initUI(self, config_dict):
        # Make a grid layout
        layout = PyQt4.QtGui.QGridLayout()
        
        # add the layout to the central widget
        self.setLayout(layout)
        
        self.output_config = copy.deepcopy(config_dict)
        
        i = 0
        # add the output config filename 
        ################################    
        fnam_label = PyQt4.QtGui.QLabel(self)
        fnam_label.setText(self.output_filename)
        
        # add the label to the layout
        layout.addWidget(fnam_label, i, 0, 1, 2)
        i += 1
        
        # we have 
        self.labels_lineedits = {}
        group_labels = []
        for group in config_dict.keys():
            # add a label for the group
            group_labels.append( PyQt4.QtGui.QLabel(self) )
            group_labels[-1].setText(group)
            # add the label to the layout
            layout.addWidget(group_labels[-1], i, 0, 1, 2)
            i += 1
            
            self.labels_lineedits[group] = {}
            # add the labels and line edits
            for key in config_dict[group].keys():
                self.labels_lineedits[group][key] = {}
                self.labels_lineedits[group][key]['label'] = PyQt4.QtGui.QLabel(self)
                self.labels_lineedits[group][key]['label'].setText(key)
                layout.addWidget(self.labels_lineedits[group][key]['label'], i, 0, 1, 1)
                
                self.labels_lineedits[group][key]['lineedit'] = PyQt4.QtGui.QLineEdit(self)
                self.labels_lineedits[group][key]['lineedit'].setText(str(config_dict[group][key]))
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

class Show_stitch_widget(PyQt4.QtGui.QWidget):
    def __init__(self, filename, config_dict):
        super(Show_stitch_widget, self).__init__()
        
        self.stitch_array  = np.zeros((2048, 2048), dtype=np.float)
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
        layout = PyQt4.QtGui.QGridLayout()
        
        # add the layout to the central widget
        self.setLayout(layout)
        
        # stitch plot
        #############
        frame_plt = pg.PlotItem(title = 'Stitch')
        self.imageView = pg.ImageView(view = frame_plt)
        self.imageView.ui.menuBtn.hide()
        self.imageView.ui.roiBtn.hide()
        if 'stitch' in self.f :
            print(self.f['stitch/stitch'].shape)
            t = self.f['stitch/stitch'].value.T.astype(np.float)
            self.imageView.setImage(t)
            self.im_init = True
        else :
            self.im_init = False
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
        self.run_button = PyQt4.QtGui.QPushButton('Calculate', self)
        self.run_button.clicked.connect(self.run_button_clicked)
        
        # add a spacer for the labels and such
        verticalSpacer = PyQt4.QtGui.QSpacerItem(20, 40, PyQt4.QtGui.QSizePolicy.Minimum, PyQt4.QtGui.QSizePolicy.Expanding)
        
        # set the layout
        ################
        layout.addWidget(self.imageView,           0, 1, 3, 1)
        layout.addWidget(self.config_widget,       0, 0, 1, 1)
        layout.addWidget(self.run_button,          1, 0, 1, 1)
        layout.addItem(verticalSpacer,             2, 0, 1, 1)
        layout.addWidget(self.run_command_widget,  3, 0, 1, 2)
        layout.setColumnStretch(1, 1)
        layout.setColumnMinimumWidth(0, 250)
        self.layout = layout

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
        t = self.f['stitch/stitch'].value.T.astype(np.float)
        if self.im_init :
            self.imageView.setImage(t, autoRange = False, autoLevels = False, autoHistogramRange = False)
        else :
            self.imageView.setImage(t)
        self.f.close()

class Show_frames_widget(PyQt4.QtGui.QWidget):
    def __init__(self, filename):
        super(Show_frames_widget, self).__init__()
        
        self.filename = filename
        self.initUI()

    def initUI(self):
        # Make a grid layout
        layout = PyQt4.QtGui.QGridLayout()
        
        # add the layout to the central widget
        self.setLayout(layout)
        
        # Now we can add widgets to the layout
        #win = pg.GraphicsWindow(title="results")
        #layout.addWidget(win, 0, 0, 1, 1)
        f = h5py.File(self.filename, 'r')
        
        # X and Y plot 
        ##############
        R = f['R']
        X = R[:, 1]
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
        imageView.setImage(f['data'][0].T.astype(np.float))
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
            imageView.setImage( f['data'][i].T.astype(np.float), autoRange = False, autoLevels = False, autoHistogramRange = False)
            f.close()
            
        vline.sigPositionChanged.connect(replot_frame)
        f.close()

class Show_frames_selection_widget(PyQt4.QtGui.QWidget):
    def __init__(self, filename):
        super(Show_frames_selection_widget, self).__init__()
        
        self.filename = filename
        self.initUI()

    def initUI(self):
        # Make a grid layout
        layout = PyQt4.QtGui.QGridLayout()
        
        # add the layout to the central widget
        self.setLayout(layout)
        
        # Now we can add widgets to the layout
        #win = pg.GraphicsWindow(title="results")
        #layout.addWidget(win, 0, 0, 1, 1)
        f = h5py.File(self.filename, 'r')
        
        # X and Y plot 
        ##############
        R = f['R']
        X = R[:, 1]
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
        imageView.setImage(f['data'][0].T.astype(np.float))
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
            imageView.setImage( f['data'][i].T.astype(np.float), autoRange = False, autoLevels = False, autoHistogramRange = False)
            f.close()
            
        vline.sigPositionChanged.connect(replot_frame)
        f.close()

class Select_frames_widget(PyQt4.QtGui.QWidget):
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
        layout = PyQt4.QtGui.QGridLayout()
        
        # add the layout to the central widget
        self.setLayout(layout)
        
        # Now we can add widgets to the layout
        f = h5py.File(self.filename, 'r')
        
        # Get the X and Y coords
        ########################
        R = f['R']
        X = R[:, 1]
        Y = R[:, 0]
        self.X = X
        self.Y = Y
        
        # select all frames initially
        self.frames = np.ones((len(X),), dtype=bool)
        
        # scatter plot
        ##############
        self.good_frame_pen     = pg.mkPen((255, 150, 150))
        self.bad_frame_pen      = pg.mkPen(None)
        #self.selected_frame_pen = pg.mkPen((150, 150, 255))
        
        self.s1    = pg.ScatterPlotItem(size=5, pen=self.good_frame_pen, brush=pg.mkBrush(255, 255, 255, 120))
        spots = [{'pos': [X[i], Y[i]], 'data': i} for i in range(len(R))] 
        self.s1.addPoints(spots)

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
        ROI_button_good   = PyQt4.QtGui.QPushButton('good frames')
        ROI_button_bad    = PyQt4.QtGui.QPushButton('bad frames')
        ROI_button_toggle = PyQt4.QtGui.QPushButton('toggle frames')
        ROI_button_good.clicked.connect(   lambda : self.mask_ROI(self.roi, 0))
        ROI_button_bad.clicked.connect(    lambda : self.mask_ROI(self.roi, 1))
        ROI_button_toggle.clicked.connect( lambda : self.mask_ROI(self.roi, 2))
        
        scatter_plot = pg.PlotWidget(title='x,y scatter plot', left='y position', bottom='x position')
        scatter_plot.addItem(self.roi)
        scatter_plot.addItem(self.s1)
        scatter_plot.addItem(self.s2)

        hbox = PyQt4.QtGui.QHBoxLayout()
        hbox.addWidget(ROI_button_good)
        hbox.addWidget(ROI_button_bad)
        hbox.addWidget(ROI_button_toggle)
        hbox.addStretch(1)

        layout.addWidget(scatter_plot   , 0, 0, 1, 1)
        layout.addLayout(hbox           , 1, 0, 1, 1)
        
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

class Run_and_log_command(PyQt4.QtGui.QWidget):
    """
    run a command and send a signal when it complete, or it has failed.

    use a Qt timer to check the process
    
    realtime streaming of the terminal output has so proved to be fruitless
    """
    finished_signal = PyQt4.QtCore.pyqtSignal(bool)
    
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
        #layout = PyQt4.QtGui.QGridLayout()
        hbox = PyQt4.QtGui.QHBoxLayout()
        
        # add the layout to the central widget
        self.setLayout(hbox)
        
        # show the command being executed
        self.command_label0 = PyQt4.QtGui.QLabel(self)
        self.command_label0.setText('<b>Command:</b>')
        self.command_label  = PyQt4.QtGui.QLabel(self)
        #self.command_label.setMaximumSize(50, 250)
         
        # show the status of the command
        self.status_label0  = PyQt4.QtGui.QLabel(self)
        self.status_label0.setText('<b>Status:</b>')
        self.status_label   = PyQt4.QtGui.QLabel(self)
        
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
        PyQt4.QtCore.QTimer.singleShot(self.polling_interval, self.update_status)
    
    def update_status(self):
        status = self.p.poll()
        if status is None :
            self.status_label.setText('Running')
             
            # start a Qt timer to update the status
            PyQt4.QtCore.QTimer.singleShot(self.polling_interval, self.update_status)
        
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

class Mask_maker_widget(PyQt4.QtGui.QWidget):
    """
    """
    def __init__(self, cspad, mask = None, output_file=None, output_path=None):
        super(Mask_maker_widget, self).__init__()
        
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

        self.unbonded_pixels = unbonded_pixels()
        self.asic_edges      = asic_edges()
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
        
        toggle_group           = QtGui.QButtonGroup()#"masking behaviour")
        toggle_group.addButton(self.toggle_checkbox)   
        toggle_group.addButton(self.mask_checkbox)   
        toggle_group.addButton(self.unmask_checkbox)   
        toggle_group.setExclusive(True)
        
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

def run_and_log_command():
    signal.signal(signal.SIGINT, signal.SIG_DFL) # allow Control-C
    app = PyQt4.QtGui.QApplication([])
    
    # Qt main window
    Mwin = PyQt4.QtGui.QMainWindow()
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
