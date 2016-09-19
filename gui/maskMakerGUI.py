#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import h5py
from PyQt4 import QtGui
import pyqtgraph as pg
import numpy as np
import scipy
#import geometry_funcs as gf

cspad_psana_shape = (4, 8, 185, 388)
cspad_geom_shape  = (1480, 1552)

def unbonded_pixels():
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

def asic_edges(arrayin = None, pad = 0):
    mask_edges = np.ones(cspad_geom_shape, dtype=np.bool)
    mask_edges[:: 185, :] = 0
    mask_edges[184 :: 185, :] = 0
    mask_edges[:, :: 194] = 0
    mask_edges[:, 193 :: 194] = 0

    if pad != 0 :
        mask_edges = scipy.signal.convolve(1 - mask_edges.astype(np.float), np.ones((pad, pad), dtype=np.float), mode = 'same') < 1
    return mask_edges

def edges(shape, pad = 0):
    mask_edges = np.ones(shape)
    mask_edges[0, :]  = 0
    mask_edges[-1, :] = 0
    mask_edges[:, 0]  = 0
    mask_edges[:, -1] = 0

    if pad != 0 :
        mask_edges = scipy.signal.convolve(1 - mask_edges.astype(np.float), np.ones((pad, pad), dtype=np.float), mode = 'same') < 1
    return mask_edges

class Application:
    def __init__(self, cspad, geom_fnam = None, mask = None, output_file=None, output_path=None):
        # check if the cspad is psana shaped
        if cspad.shape == (4, 8, 185, 388) :
            self.cspad = gf.ijkl_to_ss_fs(cspad)
            self.cspad_shape_flag = 'psana'
        elif cspad.shape == (4 * 8, 185, 388) :
            self.cspad = gf.ijkl_to_ss_fs(cspad.reshape((4,8,185, 388)))
            self.cspad_shape_flag = 'psana2'
        elif cspad.shape == (1480, 1552):
            self.cspad_shape_flag = 'slab'
            self.cspad = cspad
        else :
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
        self.geom_fnam = geom_fnam

        if self.geom_fnam is not None :
            self.pixel_maps, self.cspad_shape = gf.get_ij_slab_shaped(self.geom_fnam)
            i, j = np.meshgrid(range(self.cspad.shape[0]), range(self.cspad.shape[1]), indexing='ij')
            self.ss_geom = gf.apply_geom(self.geom_fnam, i)
            self.fs_geom = gf.apply_geom(self.geom_fnam, j)
            self.cspad_geom = np.zeros(self.cspad_shape, dtype=self.cspad.dtype)
            self.mask_geom  = np.zeros(self.cspad_shape, dtype=np.bool)
            #
            self.background = np.where(np.fliplr(gf.apply_geom(self.geom_fnam, np.ones_like(self.mask)).astype(np.bool).T) == False)
            # 
            # get the xy coords as a slab
            self.y_map, self.x_map = gf.make_yx_from_1480_1552(geom_fnam)
        else :
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
        if self.geom_fnam is not None :
            self.cspad_geom[self.pixel_maps[0], self.pixel_maps[1]] = self.cspad.ravel()
            self.mask_geom[self.pixel_maps[0], self.pixel_maps[1]]  = self.mask.ravel()
            trans      = np.fliplr(self.cspad_geom.T)
            trans_mask = np.fliplr(self.mask_geom.T) 
            #
            # I need to make the mask True between the asics...
            trans_mask[self.background] = True
        else :
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

        if self.cspad_shape_flag == 'psana' :
            print('shifting back to original cspad shape:', self.cspad_shape_flag)
            mask = gf.ss_fs_to_ijkl(self.mask)
        elif self.cspad_shape_flag == 'psana2' : 
            print('shifting back to original cspad shape:', self.cspad_shape_flag)
            mask = gf.ss_fs_to_ijkl(self.mask)
            mask = mask.reshape((32, 185, 388))
        elif self.cspad_shape_flag == 'slab' :
            mask = self.mask
        elif self.cspad_shape_flag == 'other' :
            mask = self.mask
        
        print('outputing mask as np.int16 (h5py does not support boolean arrays yet)...')
        f = h5py.File(self.output_file)
        dset = f.require_dataset(self.output_path, mask.shape, mask.dtype)
        dset[:] = mask
        #f.create_dataset(self.output_path, data = mask.astype(np.int16))
        f.close()
        print('Done!')
        
    def mask_ROI(self, roi):
        sides   = [roi.size()[1], roi.size()[0]]
        courner = [self.cspad_shape[0]/2. - roi.pos()[1], \
                   roi.pos()[0] - self.cspad_shape[1]/2.]

        top_left     = [np.rint(courner[0]) - 1, np.rint(courner[1])]
        bottom_right = [np.rint(courner[0] - sides[0]), np.rint(courner[1] + sides[1]) - 1]

        if self.geom_fnam is not None :
            # why?
            top_left[0]     += 2
            bottom_right[1] += 1
            bottom_right[0] += 1
        
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
        # Always start by initializing Qt (only once per application)
        app = QtGui.QApplication([])

        # Define a top-level widget to hold everything
        w = QtGui.QWidget()

        # 2D plot for the cspad and mask
        self.plot = pg.ImageView()

        # save mask button
        save_button = QtGui.QPushButton('save mask')
        save_button.clicked.connect(self.save_mask)

        # rectangular ROI selection
        self.roi = pg.RectROI([-200,-200], [100, 100])
        self.plot.addItem(self.roi)
        self.roi.setZValue(10)                       # make sure ROI is drawn above image
        ROI_button = QtGui.QPushButton('mask rectangular ROI')
        ROI_button.clicked.connect(lambda : self.mask_ROI(self.roi))

        # circular ROI selection
        self.roi_circle = pg.CircleROI([-200,200], [101, 101])
        self.plot.addItem(self.roi_circle)
        self.roi.setZValue(10)                       # make sure ROI is drawn above image
        ROI_circle_button = QtGui.QPushButton('mask circular ROI')
        ROI_circle_button.clicked.connect(lambda : self.mask_ROI_circle(self.roi_circle))

        # histogram mask button
        hist_button = QtGui.QPushButton('mask outside histogram')
        hist_button.clicked.connect(self.mask_hist)

        # toggle / mask / unmask checkboxes
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
        ij_label = QtGui.QLabel()
        disp = 'ss fs {0:5} {1:5}   value {2:2}'.format('-', '-', '-')
        ij_label.setText(disp)
        self.plot.scene.sigMouseMoved.connect( lambda pos: self.mouseMoved(ij_label, pos) )
        
        # unbonded pixels checkbox
        unbonded_checkbox = QtGui.QCheckBox('unbonded pixels')
        unbonded_checkbox.stateChanged.connect( self.update_mask_unbonded )
        if self.cspad_shape_flag == 'other' :
            unbonded_checkbox.setEnabled(False)
        
        # asic edges checkbox
        edges_checkbox = QtGui.QCheckBox('asic edges')
        edges_checkbox.stateChanged.connect( self.update_mask_edges )
        if self.cspad_shape_flag == 'other' :
            edges_checkbox.setEnabled(False)
        
        # mouse click mask 
        self.plot.scene.sigMouseClicked.connect( lambda click: self.mouseClicked(self.plot, click) )

        # Create a grid layout to manage the widgets size and position
        layout = QtGui.QGridLayout()
        w.setLayout(layout)

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

        # centre the circle initially 
        if self.geom_fnam is not None :
            self.roi_circle.setPos([self.cspad_shape[0]//2 - 1 - 50, self.cspad_shape[1]//2 - 1 - 50])

        ## Display the widget as a new window
        w.show()

        ## Start the Qt event loop
        app.exec_()
    
    def mouseMoved(self, ij_label, pos):
        img = self.plot.getImageItem()
        if self.geom_fnam is not None :
            ij = [self.cspad_shape[0] - 1 - int(img.mapFromScene(pos).y()), int(img.mapFromScene(pos).x())] # ss, fs
            if (0 <= ij[0] < self.cspad_shape[0]) and (0 <= ij[1] < self.cspad_shape[1]):
#                ij_label.setText('ss fs value: ' + str(self.ss_geom[ij[0], ij[1]]).rjust(5) + str(self.fs_geom[ij[0], ij[1]]).rjust(5) + str(self.cspad_geom[ij[0], ij[1]]).rjust(8) )
                ij_label.setText('ss fs value: %d %d %.2e' % (self.ss_geom[ij[0], ij[1]], self.fs_geom[ij[0], ij[1]], self.cspad_geom[ij[0], ij[1]]) )
        else :
            ij = [self.cspad.shape[0] - 1 - int(img.mapFromScene(pos).y()), int(img.mapFromScene(pos).x())] # ss, fs
            if (0 <= ij[0] < self.cspad.shape[0]) and (0 <= ij[1] < self.cspad.shape[1]):
                ij_label.setText('ss fs value: %d %d %.2e' % (ij[0], ij[1], self.cspad[ij[0], ij[1]]) )
#                ij_label.setText('ss fs value: ' + str(ij[0]).rjust(5) + str(ij[1]).rjust(5) + str(self.cspad[ij[0], ij[1]]).rjust(8) )

    def mouseClicked(self, plot, click):
        if click.button() == 1:
            img = plot.getImageItem()
            if self.geom_fnam is not None :
                i0 = int(img.mapFromScene(click.pos()).y())
                j0 = int(img.mapFromScene(click.pos()).x())
                i1 = self.cspad_shape[0] - 1 - i0 # array ss (with the fliplr and .T)
                j1 = j0                           # array fs (with the fliplr and .T)
                if (0 <= i1 < self.cspad_shape[0]) and (0 <= j1 < self.cspad_shape[1]):
                    i = self.ss_geom[i1, j1]  # un-geometry corrected ss
                    j = self.fs_geom[i1, j1]  # un-geometry corrected fs
                    if i == 0 and j == 0 and i1 != 0 and j1 != 0 :
                        return 
                    else :
                        if self.toggle_checkbox.isChecked():
                            self.mask_clicked[i, j] = ~self.mask_clicked[i, j]
                            self.mask[i, j]         = ~self.mask[i, j]
                        elif self.mask_checkbox.isChecked():
                            self.mask_clicked[i, j] = False
                            self.mask[i, j]         = False
                        elif self.unmask_checkbox.isChecked():
                            self.mask_clicked[i, j] = True
                            self.mask[i, j]         = True
                        
                        if self.mask[i, j] :
                            self.display_RGB[j0, i0, :] = np.array([1,1,1]) * self.cspad[i, j]
                        else :
                            self.display_RGB[j0, i0, :] = np.array([0,0,1]) * self.cspad_max
            else :
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

def parse_cmdline_args():
    parser = argparse.ArgumentParser(description='CsPadMaskMaker - mask making, but with a mouse!')
    parser.add_argument('cspad_fnam', type=str, help="filename for the hdf5 cspad image file")
    parser.add_argument('h5path', type=str, help="hdf5 path for the 2D cspad data")
    parser.add_argument('-g', '--geometry', type=str, help="path to the CrystFEL geometry file for the image")
    parser.add_argument('-m', '--mask', type=str, help="path to the h5file of the starting mask")
    parser.add_argument('-mp', '--mask_h5path', type=str, help="path inside the h5file of the starting mask")
    return parser.parse_args()
    
if __name__ == '__main__':
    args = parse_cmdline_args()

    # load the image
    f = h5py.File(args.cspad_fnam, 'r')
    cspad = f[args.h5path].value
    f.close()

    # load the predefined mask
    if args.mask is not None :
        f = h5py.File(args.mask, 'r')
        mask = f[args.mask_h5path].value.astype(np.bool)
        f.close()
    else :
        mask = None

    # start the gui
    Application(cspad, geom_fnam = args.geometry, mask = mask, output_file=args.mask, output_path=args.mask_h5path)
    """
    ap = Application(cspad, geom_fnam = args.geometry, mask = mask)
    """
    


