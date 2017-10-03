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
import scipy

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def expand(I, Ip, Xj, Rk, search_window, mask):
    """
    Wkj = I(xj - Rk - Xj) Ip(xj)
    """
    di, dj = np.indices(Ip.shape)
    
    Wkj = np.zeros( (search_window**2,) + Ip.shape, dtype=np.float)
    for k in range(len(Rk)):
        for ii, i in enumerate(range((-search_window)//2, search_window//2, 1)):
            for jj, j in enumerate(range((-search_window)//2, search_window//2, 1)):
                d = np.zeros_like(data[0])
                
                ss = di + Xj[0] + i - R[k][0]
                fs = dj + Xj[1] + j - R[k][1]
                sfmask = (ss > 0) * (ss < O.shape[0]) * (fs > 0) * (fs < O.shape[1]) * mask
                
                kk              = search_window*ii+jj

                # this is the warped view of frame k with shift j
                Wkj[kk][sfmask] = Ip[sfmask] * I[ss[sfmask], fs[sfmask]]
    

def maximise(data, mask, R, I, Ip, X_ij, is_edge, search_window, window=6):
    """
    k = frame index
    j = pixel index (merge)
    i = pixel index (data)
    Wkj = I(xj - Rk - Xj) Ip(xj)
    Rij = \prod_k Wkj^Kki e^-Wkj
    """
    P      = np.zeros( (search_window**2,) + data.shape[1:], dtype=np.float)
    m      = np.zeros( (search_window**2,) + data.shape[1:], dtype=np.uint16)
    di, dj = np.indices(Ip.shape)

    import scipy.special
    if rank == 0 :
        print('\n\ncalculating probabilities...')
    
    for ii, i in enumerate(range((-search_window)//2, search_window//2, 1)):
        for jj, j in enumerate(range((-search_window)//2, search_window//2, 1)):
            kk= search_window*ii+jj
            for k in range(data.shape[0]):
                if not is_edge[k] :
                    # generate the shifted view of the object for frame k
                    # with no warping
                    d         = np.zeros_like(data[0])
                    ss        = di + i - R[k][0] + X_ij[0]
                    fs        = dj + j - R[k][1] + X_ij[1]
                    sfmask    = (ss > 0) * (ss < I.shape[0]) * (fs > 0) * (fs < I.shape[1]) * mask
                    d[sfmask] = Ip[sfmask] * I[ss[sfmask], fs[sfmask]]

                    # exclude zeros
                    sfmask[d<1] = False
                    
                    #p      += likelihood_match(data[k], d, window, sfmask)
                    l       = data[k][sfmask] * np.log(d[sfmask]) - d[sfmask]
                    P[kk][sfmask]  += l
                    m[kk][sfmask]  += 1
            
            P[kk]   = no_wrap_smooth(P[kk], mask, window/2.)
                
    # all reduce P
    comm.Allreduce(P.copy(), P)
    comm.Allreduce(m.copy(), m)
    
    # P: log likelihood --> probability normalised over the shifts
    P   -= np.max(P, axis=0)
    P    = np.exp( P)
    
    #P    = np.array( [no_wrap_smooth(p, mask, 10) for p in P] )
    # normalise
    psum = np.sum(P, axis=0)
    #i    = (psum == 0)
    #psum[i] = 1.
    P  /= psum
    #P  *= ~i
    P  = P * (m>2)
    P0 = P.copy()
    #P0 = t
    
    X_ij_out = update_Xij(P, X_ij)
    
    # EMC style update
    Iout     = update_O_EMC(P, I.shape, R, X_ij, data, Ip, mask)
    
    # sharp update
    #Iout     = update_O(I, R, X_ij_out, data, Ip, mask)
    
    return Iout, P0, X_ij_out
    
def likelihood_match(data, lamb, window, mask):
    """
    return l(lamb; data) = sum_window (data[i] ln(lamb[i]) - lamb[i])
    """
    #import scipy.special
    i = (lamb>0)*mask
    l = np.zeros(data.shape, dtype=np.float)
    l[i] = data[i] * np.log(lamb[i]) - lamb[i] #- scipy.special.gammaln(data[i]+1)
    if (window is not None) or (window > 0) :
        l    = no_wrap_smooth(l, mask, window/2.)
    l[~i] = 0
    return l

def no_wrap_smooth(l, mask, sig):
    l = scipy.ndimage.gaussian_filter(l, sig, mode='constant')
    n = scipy.ndimage.gaussian_filter(mask.astype(np.float), sig, mode='constant')
    n[n==0] = 1.
    return l / n
    
def update_Xij(P, X_ij):
    search_window = int(np.sqrt(P.shape[0]))
    
    # now update X
    X_ij_out = X_ij.copy()
    dy, dx = np.unravel_index(np.argmax(P, axis=0), (search_window, search_window))
    dy = dy + (-search_window)//2
    dx = dx + (-search_window)//2
    
    #dy_mean = np.mean(dy)
    #dx_mean = np.mean(dx)
    #dy = np.rint(dy.astype(np.float)-dy_mean).astype(np.int)
    #dx = np.rint(dx.astype(np.float)-dx_mean).astype(np.int)
    
    X_ij_out[0] += dy
    X_ij_out[1] += dx
    return X_ij_out

def update_O_EMC(P, Oshape, R, X_ij, data, W, mask):
    # now update W 
    Iout = np.zeros(Oshape, dtype=np.float)
    N    = np.zeros(Oshape, dtype=np.float)
    
    search_window = int(np.sqrt(P.shape[0]))
    di, dj = np.indices(W.shape)
    if rank == 0 :
        print('merging...')
    for k in range(data.shape[0]):
        for ii, i in enumerate(range((-search_window)//2, search_window//2, 1)):
            for jj, j in enumerate(range((-search_window)//2, search_window//2, 1)):
                kk        = search_window*ii+jj
                
                # generate the shifted view of the object for frame k
                # with no warping
                ss        = di + i - R[k][0] + X_ij[0]
                fs        = dj + j - R[k][1] + X_ij[1]
                sfmask    = (ss > 0) * (ss < Oshape[0]) * (fs > 0) * (fs < Oshape[1]) * mask
                
                Iout[ss[sfmask], fs[sfmask]] += P[kk][sfmask] * data[k][sfmask] * W[sfmask]
                N[ss[sfmask], fs[sfmask]]    += P[kk][sfmask] * W[sfmask]**2
        
    comm.Allreduce(N.copy(), N)
    comm.Allreduce(Iout.copy(), Iout)
    
    Iout /= (N+1.0e-5)
    #Iout[Iout==0] = 1.
    return Iout

def update_O(O, R, X_ij, data, W, mask):
    O_out   = np.ascontiguousarray(np.zeros_like(O))
    W_map   = np.ascontiguousarray(np.zeros_like(O))
    
    di, dj = np.indices(data[0].shape)
    
    for k in range(data.shape[0]):
        d = np.zeros_like(data[0])
        
        ss = di + X_ij[0] - R[k][0]
        fs = dj + X_ij[1] - R[k][1]
        sfmask = (ss > 0) * (ss < O.shape[0]) * (fs > 0) * (fs < O.shape[1]) * mask
        
        # add to O_out
        O_out[ss[sfmask], fs[sfmask]] += W[sfmask] * data[k][sfmask]
        W_map[ss[sfmask], fs[sfmask]] += W[sfmask]**2
                
    # calculate the next object
    comm.Allreduce([W_map.copy(), MPI.DOUBLE], \
                   [W_map,        MPI.DOUBLE], \
                   op=MPI.SUM)
    
    comm.Allreduce([O_out.copy(), MPI.DOUBLE], \
                   [O_out,        MPI.DOUBLE], \
                   op=MPI.SUM)
    
    W_map[W_map==0] = 1.
    O_out          /= W_map
    #O_out[O_out==0] = 1.
        
    return O_out


def chunkIt(seq, num):
    splits = np.mgrid[0:len(seq):(num+1)*1J].astype(np.int)
    out    = []
    for i in range(splits.shape[0]-1):
        out.append(seq[splits[i]:splits[i+1]])
    return out

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
        args.config = os.path.join(os.path.split(args.filename)[0], 'tie_stitch.ini')
        if not os.path.exists(args.config):
            args.config = '../process/tie_stitch.ini'
    
    # check that args.config exists
    if not os.path.exists(args.config):
        raise NameError('config file does not exist: ' + args.config)
    
    # process config file
    config = ConfigParser.ConfigParser()
    config.read(args.config)
    
    params = Putils.parse_parameters(config)
    
    return args, params

def prepare_input(dshape, R, O, X_ij):
    RO = np.rint(R).astype(np.int)
    
    # the regular pixel values
    i, j = np.indices(dshape[1 :])
    
    if X_ij is not None :
        X_ijO = np.rint(X_ij).astype(np.int)
    else :
        X_ijO = np.zeros_like([i, j])
     
    # make the object grid
    # add room for half a data frame
    RO[:, 0] -= dshape[1]//2
    RO[:, 1] -= dshape[2]//2
    Oshape = (int(round(i.max() + np.max(np.abs(RO[:, 0])) + dshape[1]//2)), \
              int(round(j.max() + np.max(np.abs(RO[:, 1])) + dshape[2]//2)))
    #Oshape = (int(round(i.max() + np.max(np.abs(RO[:, 0])) )), \
    #          int(round(j.max() + np.max(np.abs(RO[:, 1])) )))
    
    if O is None :
        O = np.zeros(Oshape, dtype=np.float)
    return RO, O, X_ijO
    
def downsample_array(ar, factor=2, av_bin = 'bin'):
    """
    downsample ar by 'factor' accounting for the mask
    """
    shape   = ar.shape
    i, j    = np.arange(shape[0]), np.arange(shape[1])
    i2, j2  = i//factor, j//factor
    i3, j3  = np.unique(i2), np.unique(j2)
    
    i, j    = np.meshgrid(i2, j2, indexing='ij')
    i2, j2  = np.meshgrid(i3, j3, indexing='ij')

    ij  = (i*i2.shape[1]  + j ).ravel()
    ij2 = (i2*i2.shape[1] + j2).ravel()

    ar_hist = np.bincount(ij, ar.ravel())
    
    return ar_hist[ij2].reshape(i2.shape)

def upsample_array(ar, shape, factor=2, smooth=True):
    i, j    = np.arange(shape[0]), np.arange(shape[1])
    i2, j2  = i//factor, j//factor
    i3, j3  = np.unique(i2), np.unique(j2)
    
    i, j    = np.meshgrid(i2, j2, indexing='ij')
    i2, j2  = np.meshgrid(i3, j3, indexing='ij')
    
    ij  = (i*i2.shape[1]  + j ).ravel()
    ij2 = (i2*i2.shape[1] + j2).ravel()
    
    ar_hist = np.bincount(ij2, ar.ravel())
    ar_out  = ar_hist[ij].reshape(i.shape)
    
    ar_out = no_wrap_smooth(ar_out, np.ones_like(ar_out), factor/8.)
    return ar_out
    
    
def downsample(data, mask, R, O, W, X_ij, factor = 2):
    
    data_out = np.array([downsample_array(d*mask, factor) for d in data.astype(np.float)])
    mask_out = downsample_array(mask.astype(np.int), factor)>0
    O_out    = downsample_array(O, factor) 
    W_out    = downsample_array(W*mask, factor) 
    X_ij_out = np.array([downsample_array(x, factor) for x in X_ij], dtype=X_ij.dtype)
    R_out    = R // factor

    # normalise
    n        = downsample_array(mask.astype(np.float), factor)
    n[n==0]  = 1
    data_out *= factor**2 / n
    W_out    *= factor**2 / n
    
    no        = downsample_array(np.ones_like(O), factor)
    no[no==0] = 1
    O_out    /= factor**2 * no 
    
    # bit more complicated
    X_ij_out = np.rint(X_ij_out.astype(np.float) / float(factor**3)).astype(np.int)
    return data_out, mask_out, R_out, O_out, W_out, X_ij_out

def determine_edges(R, X_ij, search_window):
    """
    Find the R's for which any edge of any frame hits the edge
    (0, 0) is the top left courner of the first frame
    """
    di, dj = np.indices(X_ij[0].shape)
    RO     = np.rint(R).astype(np.int)
    Oshape = (int(round(di.max() + np.max(np.abs(RO[:, 0])) )), \
              int(round(dj.max() + np.max(np.abs(RO[:, 1])) )))
    is_edge = np.zeros((len(R),), dtype=np.bool)
    edginess = np.zeros((len(R),), dtype=np.int)
    for k in range(R.shape[0]):
        for i in [(-search_window)//2, search_window//2, 1]:
            for j in [(-search_window)//2, search_window//2, 1]:
                ss        = di + i - RO[k][0] + X_ij[0]
                fs        = dj + j - RO[k][1] + X_ij[1]
                sfmask    = (ss > 0) * (ss < Oshape[0]) * (fs > 0) * (fs < Oshape[0])
                edginess[k] += np.sum(~sfmask)
                if edginess[k]>0 :
                    is_edge[k] = True
    return is_edge, edginess
    
def unwrap_smooth(X_ij, smooth):
    import skimage.restoration
    
    dx, dy = X_ij
    dx = skimage.restoration.unwrap_phase(dx, wrap_around=(False, False))
    dy = skimage.restoration.unwrap_phase(dy, wrap_around=(False, False))
    
    dx = no_wrap_smooth(dx, np.ones_like(X_ij[0]), smooth)
    dy = no_wrap_smooth(dy, np.ones_like(X_ij[1]), smooth)
    
    X_ij_out = np.empty_like(X_ij)
    X_ij_out[0] = np.rint(dx).astype(np.int)
    X_ij_out[1] = np.rint(dy).astype(np.int)
    """
    dx = X_ij[0].astype(np.float)
    # rescale between -pi and pi
    dxmin, dxmax = dx.min(), dx.max()
    dx = (dx - dxmin) * 2 * np.pi / (dxmax-dxmin) - np.pi
    # unwrap
    dx = skimage.restoration.unwrap_phase(dx, wrap_around=(False, False))
    # rescale 
    dx = (dx + np.pi) * (dxmax-dxmin) / (2.*np.pi) + dxmin
    # smooth
    dx = no_wrap_smooth(dx, np.ones_like(X_ij[0]), smooth)
    X_ij[0] = np.rint(dx).astype(np.int)

    dy = X_ij[1].astype(np.float)
    # rescale between -pi and pi
    dymin, dymax = dy.min(), dy.max()
    dy = (dy - dymin) * 2 * np.pi / (dymax-dymin) - np.pi
    # unwrap
    dy = skimage.restoration.unwrap_phase(dy, wrap_around=(False, False))
    # rescale 
    dy = (dy + np.pi) * (dymax-dymin) / (2.*np.pi) + dymin
    # smooth
    dy = no_wrap_smooth(dy, np.ones_like(X_ij[0]), smooth)
    X_ij[1] = np.rint(dy).astype(np.int)
    """
    return X_ij_out

def read_from_file(fnam, params):
    f = h5py.File(fnam)
    
    ################################
    # Get the inputs
    # frames, df, R, O, W, ROI, mask
    ################################
    group = params['EMC']['h5_group']
    
    # ROI
    # ------------------
    if params['EMC']['roi'] is not None :
        ROI = params['EMC']['roi']
    else :
        ROI = [0, f['entry_1/data_1/data'].shape[0], 0, f['entry_1/data_1/data'].shape[1]]
    
    # frames
    # ------------------
    # get the frames to process
    if params['EMC']['good_frames'] is not None :
        good_frames = list(f[params['EMC']['good_frames']][()])
    else :
        good_frames = range(f['entry_1/data_1/data'].shape[0])
    
    # everyone has their own frames
    my_frames = chunkIt(good_frames, size)[rank]
    data = np.array([f['/entry_1/data_1/data'][fi][ROI[0]:ROI[1], ROI[2]:ROI[3]] for fi in my_frames])
    
    # df
    # ------------------
    # get the sample to detector distance
    if params['EMC']['defocus'] is not None :
        df = params['EMC']['defocus']
    else :
        df = f['/entry_1/sample_3/geometry/translation'][0, 2]
    
    # R
    # ------------------
    # get the pixel shift coordinates along ss and fs
    R, dx = utils.get_Fresnel_pixel_shifts_cxi(f, good_frames, params['EMC']['defocus'], offset_to_zero=True)
    
    # allow for astigmatism
    if params['EMC']['defocus_fs'] is not None :
        R[:, 1] *= df / params['EMC']['defocus_fs']
    
    # W
    # ------------------
    # get the whitefield
    if params['EMC']['whitefield'] is not None :
        W = f[params['EMC']['whitefield']][()][ROI[0]:ROI[1], ROI[2]:ROI[3]].astype(np.float)
        if params['EMC']['whitefield'] == 'process_2/powder' :
            W /= float(f['/entry_1/data_1/data'].shape[0])
    else :
        W = np.mean(data, axis=0)

    # mask
    # ------------------
    # mask hot / dead pixels
    if params['EMC']['mask'] is None :
        bit_mask = f['entry_1/instrument_1/detector_1/mask'].value
        # hot (4) and dead (8) pixels
        mask     = ~np.bitwise_and(bit_mask, 4 + 8).astype(np.bool) 
    else :
        mask = f[params['EMC']['mask']].value
    mask     = mask[ROI[0]:ROI[1], ROI[2]:ROI[3]]

    # delta_ij
    # -------------------
    if params['EMC']['pixel_shifts'] is not None :
        delta_ij    = np.zeros((2,) + f['/entry_1/data_1/data'].shape[1:], dtype=np.float)
        delta_ij    = f[params['EMC']['pixel_shifts']][()]
        delta_ij    = delta_ij[:, ROI[0]:ROI[1], ROI[2]:ROI[3]]
        delta_from_file = True

        # if delta_ij is zero within the ROI then extend the pixel shifts by padding
        from cpu_stitch import fill_pixel_shifts_from_edge
        delta_ij = fill_pixel_shifts_from_edge(delta_ij)
    else :
        delta_from_file = False
        delta_ij    = None
        delta_ij    = np.zeros((2,)+data.shape[1:], dtype=np.int)
    
    f.close()
    return ROI, good_frames, my_frames, data, df, R, dx, W, mask, delta_ij


def write_output(fnam, params, pupil_full, phase_full, 
                 aberrations_full, P_sample, P_focus, 
                 Os, delta_ij_full, P, W_full):
    f = h5py.File(fnam)
    
    if params['EMC']['pixel_shifts'] is not None :
        delta_from_file = True
    else :
        delta_from_file = False
    
    # write the result 
    ##################
    if params['EMC']['output_file'] is not None :
        g = h5py.File(params['EMC']['output_file'])
        outputdir = os.path.split(params['EMC']['output_file'])[0]
    else :
        g = f
        outputdir = os.path.split(fnam)[0]
    
    group = params['EMC']['h5_group']
    if group not in g:
        print g.keys()
        g.create_group(group)

    print '\nwriting to file:'
    
    # pupil
    key = params['EMC']['h5_group']+'/pupil'
    if key in g :
        del g[key]
    g[key] = pupil_full

    # phase
    key = params['EMC']['h5_group']+'/phase'
    if key in g :
        del g[key]
    g[key] = phase_full

    # aberration
    key = params['EMC']['h5_group']+'/abberation'
    if key in g :
        del g[key]
    g[key] = aberrations_full

    # sample plane probe propagation
    key = params['EMC']['h5_group']+'/probe_sample_fres'
    if key in g :
        del g[key]
    g[key] = P_sample

    # focal spot
    key = params['EMC']['h5_group']+'/probe_focus'
    if key in g :
        del g[key]
    g[key] = P_focus
    
    # object history
    if len(Os) > 1:
        key = params['EMC']['h5_group']+'/Os'
        if delta_from_file is True and key in g and Os.shape[1:] == g[key].shape[1:] :
            Os = [g[key][i] for i in range(g[key].shape[0])] + list(Os[1:])  
        if key in g :
            del g[key]
        g[key] = np.array(Os)
    
    # pixel shifts
    key = params['EMC']['h5_group']+'/pixel_shifts'
    if key in g :
        del g[key]
    g[key] = delta_ij_full
    
    # Pearson coefficients 
    print(len(P), P.shape, P.dtype)
    if P is not None :
        key = params['EMC']['h5_group']+'/probs'
        if key in g :
            del g[key]
        g[key] = P

    # object
    key = params['EMC']['h5_group']+'/O'
    if key in g :
        del g[key]
    g[key] = Os[-1] #np.sqrt(O).astype(np.complex128)
    
    # whitefield
    key = params['EMC']['h5_group']+'/whitefield'
    if key in g :
        del g[key]
    g[key] = W_full
    
    g.close()
    
    # copy the config file
    ######################
    try :
        import shutil
        shutil.copy(args.config, outputdir)
    except Exception as e :
        print e

if __name__ == '__main__':
    args, params = parse_cmdline_args()
    ROI, good_frames, my_frames, data, df, R, dx, W, mask, delta_ij = read_from_file(args.filename, params)
    
    search_window = params['EMC']['search_window']
    window        = params['EMC']['window']
    minframes     = params['EMC']['min_frames']

    # determine edges from grid
    is_edge, edginess = determine_edges(R, delta_ij, search_window)
    if np.sum(~is_edge) < minframes :
        for i in range(minframes):
            j = np.argmin(edginess)
            is_edge[j] = False
            edginess[j] = 99999999999
        
    if rank == 0 :
        print('found:', np.sum(is_edge),' edge frames')
        print('found:', np.sum(~is_edge),' non edge frames')
    
    
    R, O, X_ij = prepare_input(data.shape, R, None, delta_ij)
    W         *= mask

    shape = data.shape[1:]
    Oshape = O.shape
    factor = params['EMC']['downsample']
    
    W0 = W.copy()
    if factor is not None and factor > 1 :
        data, mask, R, O, W, X_ij = downsample(data, mask, R, O, W, X_ij, factor = factor)

    my_R       = np.array(chunkIt(R, size)[rank])
    my_edge    = np.array(chunkIt(is_edge, size)[rank])
    O          = update_O(O, my_R, X_ij, data.astype(np.float), W, mask)
    
    
    if rank==0 :
        Os    = []
        X_ijs = []
        if factor > 1 :
            Os.append(upsample_array(O, Oshape, factor))
            X_ijs.append(np.array([factor*upsample_array(x, shape, factor) for x in X_ij]))
        else :
            Os.append(O.copy())
            X_ijs.append(X_ij.copy())

    for i in range(params['EMC']['max_iters_outer']) :
        if rank==0 :
            print(i)
        X_ij_t = X_ij.copy()
        X_new  = X_ij.copy()
        
        # refine :
        for j in range(params['EMC']['max_iters']) :
            X_old = X_new.copy()
            O, P, X_new = maximise(data.astype(np.float), mask, my_R, O, W, X_ij_t, my_edge, search_window, window)
            
            if rank==0 :
                print(j)
                # store the updated object
                if factor > 1 :
                    Os.append(upsample_array(O, Oshape, factor))
                    X_ijs.append(np.array([factor*upsample_array(x, shape, factor) for x in X_new]))
                else :
                    Os.append(O.copy())
                    X_ijs.append(X_new.copy())
             
            # break if no change
            if np.allclose(X_new, X_old):
                if rank==0 :
                    print('converged inner loop', j)
                break
        
        if np.allclose(X_new, X_ij):
            if rank==0 :
                print('converged outer loop', i)
            break
        
        # unwrap X 
        if params['EMC']['unwrap'] :
            if rank == 0 :
                print('unwraping')
            X_ij = unwrap_smooth(X_new, params['EMC']['smooth']) 
        else :
            X_ij = X_new.copy()
        
        # reinitialise O
        #O = update_O(O, my_R, X_ij, data.astype(np.float), W, mask)
            
        #if rank==0 :
            # store the updated object
            #Os.append(upsample_array(O, shape, factor))
            #Os.append(upsample_array(O, shape, factor))
    
    
    if rank == 0 :
        f = h5py.File('temp.h5')
        key = 'X_ijs'
        if key in f:
            del f[key]
        f[key] = np.array(X_ijs)
        
        key = 'Os'
        if key in f:
            del f[key]
        f[key] = np.array(Os)
        
        key = 'P'
        if key in f:
            del f[key]
        f[key] = np.array(P)
        f.close()

    W = W0
    
    # Finish
    ########
    if rank == 0 :
        Os    = np.array(Os)
        X_ijs = np.array(X_ijs)
        
        O = Os[-1].copy()
        X_ij = X_ijs[-1].copy()
        
        import cpu_stitch as cs
        f = h5py.File(args.filename)

        # get the phase
        ###############
        import scipy.constants as sc
        du = np.array([f['/entry_1/instrument_1/detector_1/y_pixel_size'][()], f['/entry_1/instrument_1/detector_1/x_pixel_size'][()]])
        z  = f['/entry_1/instrument_1/detector_1/distance'][()]
        E  = f['/entry_1/instrument_1/source_1/energy'][()]
        wavelen = sc.h * sc.c / E
        
        if 'rot_degrees' in params['EMC'] and params['EMC']['rot_degrees'] is not None :
            aberrations, phase = cs.pixel_shifts_to_phase(X_ij2, z, 75.0e-6, wavelen, df)
        else :
            aberrations, phase = cs.pixel_shifts_to_phase(X_ij, z, 75.0e-6, wavelen, df)
        pupil              = np.sqrt(W) * np.exp(1J * phase)
        
        # put back into det frame
        #########################
        delta_ij_full = np.zeros((2,) + f['entry_1/data_1/data'].shape[1:], dtype=np.float)
        delta_ij_full[0][ROI[0]:ROI[1], ROI[2]:ROI[3]] = X_ij[0]
        delta_ij_full[1][ROI[0]:ROI[1], ROI[2]:ROI[3]] = X_ij[1]
        
        phase_full = np.zeros(f['entry_1/data_1/data'].shape[1:], dtype=np.float)
        phase_full[ROI[0]:ROI[1], ROI[2]:ROI[3]] = phase
        
        aberrations_full = np.zeros(f['entry_1/data_1/data'].shape[1:], dtype=np.float)
        aberrations_full[ROI[0]:ROI[1], ROI[2]:ROI[3]] = aberrations
        
        pupil_full = np.zeros(f['entry_1/data_1/data'].shape[1:], dtype=np.complex)
        pupil_full[ROI[0]:ROI[1], ROI[2]:ROI[3]] = pupil
        
        W_full = np.zeros(f['entry_1/data_1/data'].shape[1:], dtype=np.float)
        W_full[ROI[0]:ROI[1], ROI[2]:ROI[3]] = W
        
        f.close()
        
        # get the focus spot
        ####################
        P_focus = cs.get_focus_probe(pupil)
        
        # get the sample plane probe
        ############################
        P_sample = cs.get_sample_plane_probe(pupil, wavelen, z, du, df)
        
        write_output(args.filename, params, pupil_full, phase_full, 
                     aberrations_full, P_sample, P_focus, 
                     Os, delta_ij_full, P, W_full)
