"""
take a *.pty file then crop it accourding to the mask
then bin it accourding to the configuration file
The sample coordinates, whitefield, mask ... are all adjusted accordingly

Needs to have the following structure:
$ h5ls -r MLL_520.pty    
/                        Group
/R                       Dataset {119, 3}
/data                    Dataset {119, 516, 1556}
/mask                    Dataset {516, 1556}
/metadata                Group
/metadata/R_fs_scale     Dataset {SCALAR}
/metadata/R_ss_scale     Dataset {SCALAR}
/metadata/R_z_scale      Dataset {SCALAR}
/metadata/detector_distance Dataset {SCALAR}
/metadata/fs_pixel_size  Dataset {SCALAR}
/metadata/ss_pixel_size  Dataset {SCALAR}
/metadata/wavelength     Dataset {SCALAR}
/whitefield              Dataset {516, 1556}
"""

from Ptychography import utils
import numpy as np
import h5py
import os

def parse_cmdline_args():
    import argparse
    import os
    import ConfigParser
    parser = argparse.ArgumentParser(description='crop and then bin the ptychographic data in "filename"')
    parser.add_argument('filename', type=str, \
                        help="file name of the *.pty file")
    parser.add_argument('config', type=str, \
                        help="file name of the configuration file")
    
    args = parser.parse_args()
    
    # check that cxi file exists
    if not os.path.exists(args.filename):
        raise NameError('cxi file does not exist: ' + args.filename)

    # check that args.config exists
    if not os.path.exists(args.config):
        raise NameError('config file does not exist: ' + args.config)

    # process config file
    config = ConfigParser.ConfigParser()
    config.read(args.config)
    
    params = utils.parse_parameters(config)
    
    return args, params

def crop_to_nearest_pow2(mask):
    """
    return the indices of the smallest array of shape (n**2, m**2) that holds
    all of the nonzero elements of mask.
    """
    # find the left, right, top and bottom edges
    fs = np.sum(mask, axis=0)
    ss = np.sum(mask, axis=1)
    
    fs_min = next((i for i, x in enumerate(fs) if x), None)
    fs_max = len(fs) -1 - next((i for i, x in enumerate(fs[::-1]) if x), None)
    
    ss_min = next((i for i, x in enumerate(ss) if x), None)
    ss_max = len(ss) -1 - next((i for i, x in enumerate(ss[::-1]) if x), None)
    
    fs_size = fs_max - fs_min 
    ss_size = ss_max - ss_min 
        
    # now we find n and m
    for n in range(mask.shape[0]//2 + 1):
        if 2**n > ss_size :
            break 
    
    for m in range(mask.shape[1]//2 + 1):
        if 2**m > fs_size :
            break 

    # now centre the array as much as possible without hitting the edges
    i_min = ss_min + ss_size // 2 - 2**(n-1)
    
    for i in range(mask.shape[0]):
        # have we hit the top?
        if i_min < 0 :
            i_min = i_min + 1
        # have we hit the bottom?
        elif (i_min + 2**n) > mask.shape[0]-1 :
            i_min = i_min - 1
        else :
            break
    
    j_min = fs_min + fs_size // 2 - 2**(m-1)
    
    for j in range(mask.shape[1]):
        # have we hit the left?
        if j_min < 0 :
            j_min = j_min + 1
        # have we hit the right?
        elif (j_min + 2**m) > mask.shape[1]-1 :
            j_min = j_min - 1
        else :
            break
        
    print 'fs_min, fs_max',fs_min, fs_max
    print 'ss_min, ss_max',ss_min, ss_max
    print 'fs_size, ss_size',fs_size, ss_size
    print 'n, m', n, m
    print 'i_min, i_max',i_min, i_min + 2**n
    print 'j_min, j_max',j_min, j_min + 2**m
    
    # get the flattened array indices
    i = np.arange(mask.size).reshape(mask.shape)
    
    # cut out the ones we want
    indices = i[i_min : i_min + 2**n, j_min : j_min + 2**m].ravel()
    mask_out = mask.ravel()[indices].reshape((2**n, 2**m)).copy()
    return indices, mask_out

def bin_down(I, n = 2):
    if n == 1 :
        return I.copy()
    out = np.sum(I.reshape((I.shape[0], I.shape[1]/n, n)), axis=-1)
    out = np.sum(out.T.reshape((I.shape[1]/n, I.shape[0]/n, n)), axis=-1).T
    return out

def get_new_Rscale(old_shape, new_shape, bin_ss, bin_fs):
    """
    calculate the new R values given that the detector
    has been cropped and binned.

    N, dq --> N', dq'
    dq' = dq / bin
    
    Rpix = R / dx
    dx  = 1 / N  dq
    dx' = 1 / N' dq' = bin / N' dq
    
    dx' / dx = bin N / N'
    Rpix' / Rpix = N' / bin N 
    
    R = Rpix * scale
    R = Rpix' * scale'
    R = Rpix (N' / bin N) * scale'
    scale'/scale = bin N / N' 

    so Rscale = N' / bin N
    """
    return np.array(new_shape).astype(np.float) / (np.array([bin_ss, bin_fs], dtype=np.float) \
                                                 * np.array(old_shape).astype(np.float))
    
    

def overwrite_or_create(f, key, value):
    if key in f :
        del f[key]
    f[key] = value

if __name__ == '__main__':
    args, params = parse_cmdline_args()
    f = h5py.File(args.filename)
    
    mask = f['mask'][()]
    
    # crop function
    if params['crop_and_bin']['crop_to_mask'] is True :
        indices, mask_crop = crop_to_nearest_pow2(mask)
        
        crop = lambda x : x.ravel()[indices].reshape(mask_crop.shape)
    else :
        crop = lambda x : x
    
    # crop and bin function
    crop_bin = lambda x : bin_down(crop(x), n=params['crop_and_bin']['bin'])

    mask_crop_bin = crop_bin(mask)
    
    mask_norm = mask_crop_bin.copy()
    mask_norm[mask_norm==0] = 1
    mask_norm = mask_norm.astype(np.float)

    mask_crop_bin = mask_crop_bin > 0
    
    # crop and bin function with mask normalisation function
    crop_bin_I = lambda x : crop_bin(x.astype(np.float) * mask) / mask_norm

    # write the result 
    ##################
    if params['crop_and_bin']['output'] is None :
        fnam = os.path.abspath(args.filename).split('.')[0] + '_cropped_binned.pty'
    else :
        fnam = params['crop_and_bin']['output']

    g = h5py.File(fnam)
    
    # mask
    overwrite_or_create(g, 'mask', mask_crop_bin)

    # Rs 
    Rscale = get_new_Rscale(mask.shape, mask_crop_bin.shape, params['crop_and_bin']['bin'], params['crop_and_bin']['bin'])
    R = f['R'].value 
    R[:, 0] = R[:, 0] * Rscale[0]
    R[:, 1] = R[:, 1] * Rscale[1]
    overwrite_or_create(g, 'R', R)

    # whitefield
    overwrite_or_create(g, 'whitefield', crop_bin_I(f['whitefield'][()]))

    # metadata
    key = '/metadata/R_ss_scale'
    overwrite_or_create(g, key, f[key][()] / Rscale[0])

    key = '/metadata/R_fs_scale'
    overwrite_or_create(g, key, f[key][()] / Rscale[1])

    key = '/metadata/detector_distance'
    overwrite_or_create(g, key, f[key][()])

    key = '/metadata/fs_pixel_size'
    overwrite_or_create(g, key, f[key][()] * params['crop_and_bin']['bin'])

    key = '/metadata/ss_pixel_size'
    overwrite_or_create(g, key, f[key][()] * params['crop_and_bin']['bin'])
    
    key = '/metadata/wavelength'
    overwrite_or_create(g, key, f[key][()])

    key = '/metadata/grid'
    overwrite_or_create(g, key, f[key][()])

    key = '/metadata/steps'
    overwrite_or_create(g, key, f[key][()])
    
    # data
    if 'data' in g :
        del g['data'] 

    dset = g.create_dataset('data', dtype=np.float, shape=(f['data'].shape[0],) + mask_crop_bin.shape)
    for i, frame in enumerate(f['data']):
        g['data'][i] = crop_bin_I(frame)
        if i % 100 == 0 :
            print i, f['data'].shape[0]
        
    # copy the config file
    ######################
    import shutil
    outputdir = os.path.split(args.filename)[0]
    shutil.copy(args.config, outputdir)
