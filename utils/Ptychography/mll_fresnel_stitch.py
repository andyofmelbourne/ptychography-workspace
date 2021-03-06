import numpy as np
import h5py
import scipy.constants as sc
import time
from scipy import ndimage
import os

import Ptychography.ptychography.era as era
from Ptychography import DM
from Ptychography import ERA
from Ptychography import utils
from Ptychography import write_cxi

from mll_cxi_wrapper import *

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def make_P_heatmap(P, R, shape):
    P_heatmap = np.zeros(shape, dtype = P.real.dtype)
    #P_temp    = np.zeros(shape, dtype = P.real.dtype)
    #P_temp[:P.shape[0], :P.shape[1]] = (P.conj() * P).real
    P_temp = (P.conj() * P).real
    for r in R : 
        #P_heatmap += multiroll(P_temp, [-r[0], -r[1]]) 
        P_heatmap[-r[0]:P.shape[0]-r[0], -r[1]:P.shape[1]-r[1]] += P_temp
    return P_heatmap

def make_O_heatmap(O, R, shape):
    O_heatmap = np.zeros(O.shape, dtype = O.real.dtype)
    O_temp    = (O * O.conj()).real
    for r in R : 
        O_heatmap += era.multiroll(O_temp, [r[0], r[1]]) 
    return O_heatmap[:shape[0], :shape[1]]

def psup_P(exits, O, R, O_heatmap = None, alpha = 1.0e-10, MPI_dtype = MPI.DOUBLE, MPI_c_dtype = MPI.DOUBLE):
    PT = np.zeros(exits[0].shape, exits.dtype)
    
    # Calculate denominator
    #----------------------
    # but only do this if it hasn't been done already
    # (we must set O_heatmap = None when the object/coords has changed)
    if O_heatmap is None : 
        O_heatmapT = np.ascontiguousarray(make_O_heatmap(O, R, PT.shape))
        #O_heatmapT = era.make_O_heatmap(O, R, PT.shape) produces a non-contig. array for some reason
        O_heatmap  = np.empty_like(O_heatmapT)
        comm.Allreduce([O_heatmapT, MPI_dtype], \
                       [O_heatmap,  MPI_dtype], \
                       op=MPI.SUM)

    # Calculate numerator
    #--------------------
    Oc = O.conj()
    for r, exit in zip(R, exits):
        PT += exit * Oc[-r[0]:PT.shape[0]-r[0], -r[1]:PT.shape[1]-r[1]] 
         
    # divide
    #-------
    P = np.empty_like(PT)
    comm.Allreduce([PT, MPI_c_dtype], \
                   [P, MPI_c_dtype],   \
                    op=MPI.SUM)
    P  = P / (O_heatmap + alpha)
    
    return P, O_heatmap

def psup_O(exits, P, R, O_shape, P_heatmap = None, alpha = 1.0e-10, MPI_dtype = MPI.DOUBLE, MPI_c_dtype = MPI.DOUBLE, verbose = False, sample_blur = None):
    OT = np.zeros(O_shape, P.dtype)
    
    # Calculate denominator
    #----------------------
    # but only do this if it hasn't been done already
    # (we must set P_heatmap = None when the probe/coords has changed)
    if P_heatmap is None : 
        P_heatmapT = make_P_heatmap(P, R, O_shape)
        P_heatmap  = np.empty_like(P_heatmapT)
        #comm.Allreduce([P_heatmapT, MPI.__TypeDict__[P_heatmapT.dtype.char]], \
        #               [P_heatmap,  MPI.__TypeDict__[P_heatmap.dtype.char]], \
        #               op=MPI.SUM)
        comm.Allreduce([P_heatmapT, MPI_dtype], \
                       [P_heatmap,  MPI_dtype], \
                       op=MPI.SUM)

    # Calculate numerator
    #--------------------
    for r, exit in zip(R, exits):
        OT[-r[0]:P.shape[0]-r[0], -r[1]:P.shape[1]-r[1]] += exit * P.conj()
    
    # divide
    # here we need to do an all reduce
    #---------------------------------
    O = np.empty_like(OT)
    #comm.Allreduce([OT, MPI.__TypeDict__[OT.dtype.char]], \
    #               [O, MPI.__TypeDict__[O.dtype.char]],   \
    #                op=MPI.SUM)
    comm.Allreduce([OT, MPI_c_dtype], \
                   [O, MPI_c_dtype],  \
                    op=MPI.SUM)
    comm.Barrier()
    O  = O / (P_heatmap + alpha)

    if sample_blur is not None :
        import scipy.ndimage
        O.real = scipy.ndimage.gaussian_filter(O.real, sample_blur, mode='wrap')
        O.imag = scipy.ndimage.gaussian_filter(O.imag, sample_blur, mode='wrap')
    
    # set a maximum value for the amplitude of the object
    #O = np.clip(np.abs(O), 0.0, 2.0) * np.exp(1.0J * np.angle(O))
    return O, P_heatmap

def OP_sup(I, R, whitefield, O, mask):
    if O is None :
        # find the smallest array that fits O
        # This is just U = M + R[:, 0].max() - R[:, 0].min()
        #              V = K + R[:, 1].max() - R[:, 1].min()
        shape = (I.shape[1] + R[:, 0].max() - R[:, 0].min(),\
                 I.shape[2] + R[:, 1].max() - R[:, 1].min())
        O = np.ones(shape, dtype = np.float64)

    # subtract an overall offset from R's
    R[:, 0] -= R[:, 0].max()
    R[:, 1] -= R[:, 1].max()
    
    P = whitefield**2
    
    for i in range(4):
        O0 = O.copy()
        O, P_heatmap = psup_O(I, P, R, O.shape, None)
        P, O_heatmap = psup_P(I, O, R)
        print i, np.sum( (O0 - O)**2 )
    return O

def save_plot(O, R, Rpix, scan, f):
    # scan 
    fnam = os.path.split(scan)[-1][:-4] 
    fnam = fnam + '/' + fnam + '_stitch.png'

    # get the sample positions
    fast_axis   = f['entry_1/instrument_1/motor_positions/scan_axes/Fast axis/name'].value
    slow_axis   = f['entry_1/instrument_1/motor_positions/scan_axes/Slow axis/name'].value
    mll1_name   = f['entry_1/sample_1/name'].value
    mll2_name   = f['entry_1/sample_2/name'].value
    sample_name = f['entry_1/sample_3/name'].value
    z           = f['entry_1/instrument_1/detector_1/distance'].value
    
    # get the beam energy and wavelength
    E    = f['entry_1/instrument_1/source_1/energy'].value 
    
    import matplotlib.pyplot as plt
    
    plt.ioff()

    # get the x-axis positions assuming 'x' is the fast scan
    dx = (R[:, 0].max() - R[:, 0].min()) / (Rpix[:, 1].max() - Rpix[:, 1].min())
    dy = (R[:, 1].max() - R[:, 1].min()) / (Rpix[:, 0].max() - Rpix[:, 0].min())
    fast_values = np.arange(O.shape[1]) * dx
    slow_values = np.arange(O.shape[0]) * dy

    print dx, dy
    
    # choose the units for fs and ss
    units = []
    for axis, value in zip([fast_axis, slow_axis], [fast_values, slow_values]):
        d = np.abs(value[-1] - value[0])
        if axis[0] in ['X', 'Y', 'Z']:
            if d < 1.0e-6 :
                unit = [1.0e9, 'nm']
            elif d < 1.0e-3 :
                unit = [1.0e6, 'um']
            elif d < 1.0e-2 :
                unit = [1.0e3, 'mm']
            elif d < 1.0e-1 :
                unit = [1.0e2, 'cm']
            else :
                unit = [1.0, 'm']
            units.append(unit)
        elif axis[0] in ['r', 'y', 'p']:
            if d < 1.0e-6 :
                unit = [1.0e9, 'nrad']
            elif d < 1.0e-3 :
                unit = [1.0e6, 'urad']
            elif d < 1.0e-2 :
                unit = [1.0e3, 'mrad']
            elif d < 1.0e-1 :
                unit = [1.0e2, 'crad']
            else :
                unit = [1.0, 'rad']
            units.append(unit)
        else :
            units.append([1.0, '?'])

    #fig = plt.figure(figsize = [3, 3], dpi=900)
    fig = plt.figure(dpi=900)

    extent = [fast_values[0] * units[0][0], fast_values[-1] * units[0][0], 
              slow_values[0] * units[1][0], slow_values[-1] * units[1][0]]

    # test
    #y, x = np.meshgrid(range(10), range(20), indexing = 'ij')
    #r = np.sqrt(x**2 + y**2)
    #extent = [x.minparams['input']['cxi_fnam'](), x.max(), y.min(), y.max()]
    #ax.imshow(r, extent = extent, aspect = 'auto')#, origin='lower left')

    #vmin = [np.percentile(O, 20.0), np.percentile(O, 80.0) * 1.1]
    vmin = [0.8, 1.2]

    aspect = (extent[1] - extent[0]) / (extent[3] - extent[2])
    
    ax = plt.subplot(111)
    ax.set_xlabel(fast_axis + ' ('+units[0][1]+')')
    ax.set_ylabel(slow_axis + ' ('+units[1][1]+')')
    ax.set_title('Scan:'+scan+' Sample:'+sample_name+'\n MLL1:'+mll1_name+' MLL2:'+mll2_name+'\n Energy (keV):'+'{0:.2f}'.format( 1.0e-3 * E / sc.e ) , fontsize=6)
    plt.subplots_adjust(bottom=0.14, right=0.99, left=0.18, top=0.88)
    ax.imshow(O, extent = extent, origin='lower left', interpolation='nearest', cmap='Greys_r', vmin = vmin)

    # We change the fontsize of minor ticks label 
    plt.tick_params(axis='both', which='major', labelsize=6)
    plt.tick_params(axis='both', which='minor', labelsize=5)

    print 'saving figure:', fnam
    plt.tight_layout()
    plt.savefig(fnam)
    plt.close(fig)
    #plt.show()

if __name__ == '__main__':
    params = utils.parse_cmdline_args()
    
    if rank == 0 :
        print '\n\nLoading', params['input']['cxi_fnam'] 
        f = h5py.File(params['input']['cxi_fnam'], 'r')
         
        print '\n\nMask'
        print '####'
        mask, I_crop_pad_downsample, crop_pad_downsample_nomask = get_mask(f, params)
                
        print '\n\nR'
        print '####'
        R, Rpix, Rindex, F = get_Rs(f, mask, params)
        
        print '\n\nI'
        print '####'
        I = get_Is(f, mask, Rindex, I_crop_pad_downsample, params)
        
        print '\n\nP0'
        print '####'
        P0, prop, iprop, whitefield, exps = make_P0(f, mask, Rindex, F, I_crop_pad_downsample, crop_pad_downsample_nomask, params)
        
        #O0 = make_O0(I, Rpix, whitefield + 0J, None, mask)
        O = OP_sup(I, Rpix, whitefield, None, mask)

        save_plot(O, R, Rpix, params['input']['cxi_fnam'], f)

        """
        Os = []
        for defocus in np.linspace(0.1e-4, 2.0e-4, 50):
            print '\n\n\nDefocus:', defocus
            
            params['input']['defocus'] = defocus
            #params['input']['defocus'] = 0.0005467500000000003

            R, Rpix, Rindex, F = get_Rs(f, mask, params)
            
            O0 = OP_sup(I, Rpix, whitefield, None, mask)
            
            Os.append(O0.copy())
        
        shape_new = Os[0].shape

        for i in range(len(Os)):
            Os[i] = zero_pad_to_nearest_pow2(Os[i], shape_new = shape_new)
        
        Os = np.array(Os)
        
        import pyqtgraph as pg
        """

     
