import scipy.constants as sc
import h5py
import numpy as np

import sys, os

import time
import ConfigParser

root = os.path.split(os.path.abspath(__file__))[0]
root = os.path.split(root)[0]
sys.path.append(root)
sys.path.append(os.path.join(root, 'utils'))
sys.path.append(os.path.join(root, 'process'))

from Ptychography import utils as Putils
import utils 
import optics 
from numpy.polynomial import polynomial as P
from numpy.polynomial import legendre as L

#import pyximport; pyximport.install()
#from feature_matching import feature_map_cython

class Cpu_stitcher():

    def __init__(self, data, mask, W, R, O, X_ij): 
	dtype     = np.float64
	self.IW   = mask * data * W
        self.Od   = np.zeros_like(data, dtype=dtype)
	self.mask = mask
	self.WW   = mask * W**2
	self.W    = mask * W
	self.R    = np.rint(R).astype(np.int)
	
	# add room for half a data frame
        self.R[:, 0] -= data.shape[1]//2
        self.R[:, 1] -= data.shape[2]//2
	
	if O is None :
	    self.O  = O
	else :
	    self.O  = O
	
        # the regular pixel values
        self.i, self.j = np.indices(self.IW.shape[1 :])

	if X_ij is not None :
	    self.X_ij = np.rint(X_ij).astype(np.int)
	else :
	    self.X_ij = np.zeros_like([self.i, self.j])
         
        # make the object grid
        Oshape = (int(round(self.i.max() + np.max(np.abs(self.R[:, 0])) + data.shape[1]//2)), \
                  int(round(self.j.max() + np.max(np.abs(self.R[:, 1])) + data.shape[2]//2)))
        
        if O is None :
            self.O = np.zeros(Oshape, dtype=dtype)
        else :
	    self.O = O

	self.WWmap = np.zeros_like(self.O)
    
    def forward_map(self, X_ij):
	for k in range(self.IW.shape[0]):
	    self.Od[k] =  self.O[self.i + X_ij[0] - self.R[k][0], self.j + X_ij[1] - self.R[k][1]] 
	return self.Od
    
    def inverse_map(self, X_ij):
	self.O.fill(0)
	self.WWmap.fill(0)
	for k in range(self.IW.shape[0]):
	    self.O[    self.i + X_ij[0] - self.R[k][0], self.j + X_ij[1] - self.R[k][1]] += self.IW[k]
	    self.WWmap[self.i + X_ij[0] - self.R[k][0], self.j + X_ij[1] - self.R[k][1]] += self.WW
	self.O /= (self.WWmap + 1.0e-5)
        self.O[self.O==0] = 1.
	return self.O
    
    def calc_error(self, X_ij):
        self.O       = self.inverse_map(X_ij)
        self.Od      = self.forward_map(X_ij)
        
        # sum |sqrt(I) - sqrt(I_forward)|^2
        self.error_map  = (np.sqrt(self.IW) - np.sqrt(self.Od * self.WW))**2 / \
                          (1.0e-5+self.W)
        error = np.sum( self.error_map )
        return error
    
    def cut_object(self, O, pad = 2):
        shape = (self.IW.shape[0], pad * self.IW.shape[1], pad * self.IW.shape[2])
        Os    = np.zeros( shape, dtype = O.dtype)
        i, j  = np.indices(shape[1:])
        i    -= self.IW.shape[1] * (pad - 1)//2 + 1
        j    -= self.IW.shape[2] * (pad - 1)//2 + 1
	for k in range(self.IW.shape[0]):
	    Os[k] =  self.O[i - self.R[k][0], j - self.R[k][1]] 

        # return the slice objects such that Od[k] = Os[k][i, j]
        i, j = slice(data.shape[1]//2+1,3*data.shape[1]//2+1,1), slice(data.shape[2]//2+1,3*data.shape[2]//2+1,1)
        return Os, i, j

    def speckle_tracking_update(self, steps=8, window=16, search_window=50):
        from multiprocessing import Pool
        import itertools
          
        X_ij = self.X_ij
        mask = self.mask
        
        Os     = []
        deltas = []
        NCCs   = []
        errors = []
        
        errors.append(self.calc_error(X_ij))
        print 'Error:', errors[-1]

        Os.append(self.O.copy())
        deltas.append(X_ij.copy())
        NCCs.append(np.zeros_like(X_ij[0]))
        
        pool = Pool(processes=self.IW.shape[0])
        for ii in range(10):
            print '\n\nloop :', ii
            
            print 'setting up... '
            forwards, i, j = self.cut_object(self.O)
            frames         = self.IW / (1.0e-5+self.WW)
             
            #print '\npython stitch:'
            #x1, n1 =              feature_map(cpu_stitcher.IW[60] / (1.0e-5+cpu_stitcher.WW), Os[60], cpu_stitcher.X_ij, mask, i.start, j.start, window=4, search_window=8)
            #print '\ncython stitch:'
            #x2, n2 = utils.feature_map_cython(cpu_stitcher.IW[60] / (1.0e-5+cpu_stitcher.WW), Os[60], cpu_stitcher.X_ij, mask.astype(np.int), \
            #i.start, j.start, window=4, search_window=8)
             
            #speckle_track_np(forwards[0], frames[0], mask, 6)
            
            print 'sending to workers '
            print forwards.shape, frames.shape, mask.shape, window, search_window, steps
            args = itertools.izip( frames, forwards, itertools.repeat(X_ij), itertools.repeat(mask.astype(np.int)), \
                                   itertools.repeat(i.start), itertools.repeat(j.start), \
                                   itertools.repeat(window), itertools.repeat(search_window), itertools.repeat(steps) )
            #res  = pool.map(feature_map_cython_wrap, args)
            res  = [feature_map_cython_wrap(arg) for arg in args]
            
            print 'workers are done'
            nccs = np.array([i[1][::steps, ::steps] for i in res])
            di   = np.array([i[0][0][::steps, ::steps] for i in res])
            dj   = np.array([i[0][1][::steps, ::steps] for i in res])
            
            print nccs.shape, di.shape, dj.shape
            # do a weigted sum
            norm = np.sum(nccs, axis=0) + 1.0e-10
            X_ij[0] += np.sum(di*nccs, axis=0) / norm
            X_ij[1] += np.sum(dj*nccs, axis=0) / norm
            
            errors.append(self.calc_error(X_ij))
            print 'Error:', errors[-1]
            
            if errors[-1] > errors[-2] and ii > 0 :
                break
            
            Os.append(self.O.copy())
            deltas.append(X_ij.copy())
        
        return X_ij, Os, errors

def feature_map_cython_wrap(x):
    return utils.feature_map_cython(*x)

def feature_map_wrap(x):
    return feature_map(*x)

def feature_map(Od, O, X_ij, mask, i, j, window=10, search_window=20, steps=1, offset_i=0, offset_j=0):
    """
    Od = data / W # data view of the object
    O  = forward guess for the object in the region
    i, j = location of the top right courner of Od in O
    """
    X_ij_new = np.zeros_like(X_ij)
    ncc_w    = np.zeros( (search_window, search_window), dtype=np.float)
    ncc      = np.zeros( Od.shape, dtype=np.float)
    ii, jj   = np.indices(Od.shape)
    imap = ii + X_ij[0] + i
    jmap = jj + X_ij[1] + j

    for ii in range(offset_i, Od.shape[0], steps):
        for jj in range(offset_j, Od.shape[1], steps):
            # get the data segment
            i_d = slice(max(ii-window//2, 0), min(ii+window//2, Od.shape[0]), 1) 
            j_d = slice(max(jj-window//2, 0), min(jj+window//2, Od.shape[1]), 1)
            m     = mask[i_d, j_d]
            data  = m * Od[i_d, j_d]

            ncc_w.fill(0)
            # search the local area of O
            # compare 'data' with the search area in 'O'
            indices = range(-search_window//2, search_window//2, 1)
            for k in range(search_window):
                for l in range(search_window):
                    data_O = O[imap[i_d, j_d]+indices[k], jmap[i_d, j_d]+indices[l]] * m
                    
                    X, XX = np.mean(data),   np.mean(data**2)
                    Y, YY = np.mean(data_O), np.mean(data_O**2)
                    XY    = np.mean(data*data_O)
                    
                    # calculate the pearson coefficient
                    den = (np.sqrt(XX - X**2) * np.sqrt(YY - Y**2))
                    
                    # check if we are out of range
                    if den < 1.0e-5 :
                        ncc_w[k, l]   = -1.
                    else :
                        ncc_w[k, l]   = (XY - X*Y) / den
                    
            k, l  = np.unravel_index(np.argmax(ncc_w), ncc_w.shape)
            ncc_w = (ncc_w + 1.)/2.
            ncc_w = ncc_w / np.sum(ncc_w)
            ncc[ii, jj]         = ncc_w[k, l]
            X_ij_new[0][ii, jj] = X_ij[0][ii, jj] + indices[k]
            X_ij_new[1][ii, jj] = X_ij[1][ii, jj] + indices[l]

            # -------------------------------------
            # Plot the match
            # -------------------------------------
            i_im = X_ij[0][ii, jj] + indices[k] + imap[ii, jj]
            j_im = X_ij[1][ii, jj] + indices[l] + jmap[ii, jj]
            print 'i, j', ii, jj, 'found match at: image (full) reference frame:', (i_im, j_im),'change in X_ij:', indices[k], indices[l], 'confidence:', ncc_w[k, l]

            import matplotlib.pyplot as plt
            image = O
            coin  = Od
            
            fig = plt.figure(figsize=(16, 6))
            ax1 = plt.subplot(1, 3, 1, adjustable='box-forced')
            #ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1, adjustable='box-forced')
            ax2 = plt.subplot(1, 3, 2, adjustable='box-forced')
            ax3 = plt.subplot(1, 3, 3, adjustable='box-forced')

            vmin = [0.8, 1.2]
            ax1.imshow(coin, interpolation='nearest', cmap='Greys_r', vmin=vmin)
            ax1.set_axis_off()
            ax1.set_title('template')
            # highlight template region
            hcoin, wcoin = i_d.stop - i_d.start, j_d.stop - j_d.start
            rect2 = plt.Rectangle((j_d.start, i_d.start), wcoin, hcoin, edgecolor='r', facecolor='none')
            ax1.add_patch(rect2)
            
            ax2.imshow(image, interpolation='nearest', cmap='Greys_r', vmin=vmin)
            ax2.set_axis_off()
            ax2.set_title('image')
            # highlight matched region
            hm, wm = imap[i_d, j_d].max() - imap[i_d, j_d].min(), jmap[i_d, j_d].max() - jmap[i_d, j_d].min() 
            rect   = plt.Rectangle((jmap[i_d, j_d].min()+indices[l], imap[i_d, j_d].min()+indices[k]), wcoin, hcoin, edgecolor='r', facecolor='none')
            ax2.add_patch(rect)
            # highlight searched region
            hsearch, wsearch = imap[i_d, j_d].max() - imap[i_d, j_d].min() + search_window, jmap[i_d, j_d].max() - jmap[i_d, j_d].min() + search_window
            rect3 = plt.Rectangle((jmap[i_d, j_d].min()-search_window//2, imap[i_d, j_d].min()-search_window//2), wsearch, hsearch, edgecolor='g', facecolor='none')
            ax2.add_patch(rect3)
            
            ax3.imshow(ncc_w, interpolation='nearest', cmap='Greys_r')
            ax3.set_axis_off()
            ax3.set_title('`match_template`\nresult')
            # highlight matched region
            ax3.autoscale(False)
            ax3.plot(l, k, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)

            plt.show()
    return X_ij_new, ncc

def parse_cmdline_args():
    import argparse
    import os
    parser = argparse.ArgumentParser(description='calculate a stitch of the projection images with phase gradients')
    parser.add_argument('filename', type=str, \
                        help="file name of the *.pty file")
    parser.add_argument('-c', '--config', type=str, \
                        help="file name of the configuration file")
    
    args = parser.parse_args()
    
    # check that cxi file exists
    if not os.path.exists(args.filename):
        raise NameError('cxi file does not exist: ' + args.filename)
    
    # if config is non then read the default from the *.pty dir
    if args.config is None :
        args.config = os.path.join(os.path.split(args.filename)[0], 'gpu_stitch.ini')
        if not os.path.exists(args.config):
            args.config = '../process/gpu_stitch.ini'
    
    # check that args.config exists
    if not os.path.exists(args.config):
        raise NameError('config file does not exist: ' + args.config)
    
    # process config file
    config = ConfigParser.ConfigParser()
    config.read(args.config)
    
    params = Putils.parse_parameters(config)
    
    return args, params

if __name__ == '__main__':
    args, params = parse_cmdline_args()
    f = h5py.File(args.filename)
    
    ################################
    # Get the inputs
    # frames, df, R, O, W, ROI, mask
    # Zernike polynomials
    ################################
    group = params['gpu_stitch']['h5_group']
    
    # ROI
    # ------------------
    if params['gpu_stitch']['roi'] is not None :
        ROI = params['gpu_stitch']['roi']
    else :
        ROI = [0, f['entry_1/data_1/data'].shape[0], 0, f['entry_1/data_1/data'].shape[1]]
    
    # frames
    # ------------------
    # get the frames to process
    if 'good_frames' in f[group] :
        good_frames = list(f[group]['good_frames'][()])
    else :
        good_frames = range(f['entry_1/data_1/data'].shape[0])
    
    data = np.array([f['/entry_1/data_1/data'][fi][ROI[0]:ROI[1], ROI[2]:ROI[3]] for fi in good_frames])
    
    # df
    # ------------------
    # get the sample to detector distance
    if params['gpu_stitch']['defocus'] is not None :
        df = params['gpu_stitch']['defocus']
    else :
        df = f['/entry_1/sample_3/geometry/translation'][0, 2]
    
    # R
    # ------------------
    # get the pixel shift coordinates along ss and fs
    R, du = utils.get_Fresnel_pixel_shifts_cxi(f, good_frames, params['gpu_stitch']['defocus'], offset_to_zero=True)
    
    # allow for astigmatism
    if params['gpu_stitch']['defocus_fs'] is not None :
        R[:, 1] *= df / params['gpu_stitch']['defocus_fs']
    
    R = np.rint(R).astype(np.int)
    
    # W
    # ------------------
    # get the whitefield
    if params['gpu_stitch']['whitefield'] is not None :
        W = f[params['gpu_stitch']['whitefield']][()][ROI[0]:ROI[1], ROI[2]:ROI[3]].astype(np.float)
        if params['gpu_stitch']['whitefield'] == 'process_2/powder' :
            W /= float(f['/entry_1/data_1/data'].shape[0])
    else :
        W = np.mean(data, axis=0)

    # mask
    # ------------------
    # mask hot / dead pixels
    if params['gpu_stitch']['mask'] is None :
        bit_mask = f['entry_1/instrument_1/detector_1/mask'].value
        # hot (4) and dead (8) pixels
        mask     = ~np.bitwise_and(bit_mask, 4 + 8).astype(np.bool) 
    else :
        mask = f[params['gpu_stitch']['mask']].value
    mask     = mask[ROI[0]:ROI[1], ROI[2]:ROI[3]]

    # delta_ij
    # -------------------
    if params['gpu_stitch']['delta_ij'] is not None and 'pixel_shifts_ss_gpu_stitch' in f[params['gpu_stitch']['h5_group']].keys():
        delta_ij    = np.zeros((2,) + f['/entry_1/data_1/data'].shape[1:], dtype=np.float)
        delta_ij[0] = f[params['gpu_stitch']['h5_group']]['pixel_shifts_ss_gpu_stitch']
        delta_ij[1] = f[params['gpu_stitch']['h5_group']]['pixel_shifts_fs_gpu_stitch']
        delta_ij    = delta_ij[:, ROI[0]:ROI[1], ROI[2]:ROI[3]]
    else :
        delta_ij    = None


    if delta_ij is not None :
        delta_ij.fill(0)
    cpu_stitcher = Cpu_stitcher(data, mask, W, R, None, delta_ij)

    """
    if params['gpu_stitch']['fit_grads'] :
        delta_ij, Os, errors = cpu_stitcher.speckle_tracking_update(steps=params['gpu_stitch']['steps'], \
                                                                    window=params['gpu_stitch']['window'], \
                                                                    search_window=params['gpu_stitch']['search_window'])
    else :
        print 'stitching...'
        Os       = [cpu_stitcher.inverse_map(cpu_stitcher.X_ij)]
        delta_ij = cpu_stitcher.X_ij
        errors   = [0]

    print 'Object Field of view:', np.array(Os[-1].shape) * du
    print 'Object shape:        ', Os[-1].shape
    print 'Pixel size:          ', du
    """

    self = cpu_stitcher
    steps=params['gpu_stitch']['steps']
    window=params['gpu_stitch']['window']
    search_window=params['gpu_stitch']['search_window']


    from multiprocessing import Pool
    import itertools
      
    X_ij = self.X_ij.astype(np.float)
    mask = self.mask
    
    Os     = []
    deltas = []
    NCCs   = []
    errors = []
    
    errors.append(self.calc_error(np.rint(X_ij).astype(np.int)))
    print 'Error:', errors[-1]

    Os.append(self.O.copy())
    deltas.append(X_ij.copy())
    NCCs.append(np.zeros_like(X_ij[0]))
    
    pool = Pool(processes=self.IW.shape[0])
    for ii in range(4):
        print '\n\nloop :', ii
        
        print 'setting up... '
        forwards, i, j = self.cut_object(self.O)
        frames         = self.IW / (1.0e-5+self.WW)
         
        # add a random offset between 0 --> steps-1
        #offset_i = np.random.randint(0, steps, forwards.shape[0])
        #offset_j = np.random.randint(0, steps, forwards.shape[0])
        offset_i = np.arange(forwards.shape[0]) // steps
        offset_j = np.arange(forwards.shape[0]) % steps
        
        print 'sending to workers '
        print forwards.shape, frames.shape, mask.shape, window, search_window, steps
        args = itertools.izip( frames, forwards, itertools.repeat(np.rint(X_ij).astype(np.int)), itertools.repeat(mask.astype(np.int)), \
                               itertools.repeat(i.start), itertools.repeat(j.start), \
                               itertools.repeat(window), itertools.repeat(search_window), itertools.repeat(steps), \
                               offset_i, offset_j)
        
        #X_ij_new, NCC = feature_map_cython_wrap(args.next())
        res  = pool.map(feature_map_cython_wrap, args)
        
        X_ij_old = np.rint(X_ij).astype(np.int)
        Ods_old =  [self.O[self.i + X_ij_old[0] - self.R[k][0], self.j + X_ij_old[1] - self.R[k][1]] for k in range(frames.shape[0])]

        X_ij_new = np.rint(res[10][0]).astype(np.int)
        Ods_new =  [self.O[self.i + X_ij_new[0] - self.R[k][0], self.j + X_ij_new[1] - self.R[k][1]] for k in range(frames.shape[0])]


        break
        """
        if ii == -1 :
            res  = [feature_map_wrap(arg) for arg in args]
        else :
            #res  = [feature_map_cython_wrap(arg) for arg in args]
            res  = pool.map(feature_map_cython_wrap, args)

        print 'workers are done'
        nccs = np.array([i[1] for i in res])
        di   = np.array([i[0][0] for i in res])
        dj   = np.array([i[0][1] for i in res])
        
        print nccs.shape, di.shape, dj.shape
        # do a weigted sum
        norm = np.sum(nccs, axis=0) + 1.0e-10
        X_ij[0] += np.sum(di*nccs, axis=0) / norm
        X_ij[1] += np.sum(dj*nccs, axis=0) / norm
        
        errors.append(self.calc_error(np.rint(X_ij).astype(np.int)))
        print 'Error:', errors[-1]
        
        #if errors[-1] > errors[-2] and ii > 0 :
        #    break
        
        Os.append(self.O.copy())
        deltas.append(X_ij.copy())
        """
