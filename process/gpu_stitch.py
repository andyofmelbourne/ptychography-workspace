"""
"""

import scipy.constants as sc
import h5py
import numpy as np
import afnumpy as ap
import arrayfire as af

def stitch_with_distortions_gpu():
    pass

def unstitch_with_distortions_gpu():
    pass

def calculate_distortions_from_aberrations():
    pass

fnam = '/asap3/petra3/gpfs/p11/2016/data/11002249/processed/cxi_files/MLL_391.cxi'
ROI  = [80, 426, 60, 450] 
df   = 200.0e-6

f = h5py.File('/asap3/petra3/gpfs/p11/2016/data/11002249/processed/cxi_files/MLL_391.cxi', 'r')

du = np.array([f['/entry_1/instrument_1/detector_1/y_pixel_size'][()], f['/entry_1/instrument_1/detector_1/x_pixel_size'][()]])
z  = f['/entry_1/instrument_1/detector_1/distance'][()]
E  = f['/entry_1/instrument_1/source_1/energy'][()]
wavelen = sc.h * sc.c / E

good_frames = list(f['/process_3/good_frames'][()])
b           = f['/entry_1/instrument_1/detector_1/basis_vectors'][good_frames]
R           = f['/entry_1/sample_3/geometry/translation'][good_frames]
data        = np.array([f['/entry_1/data_1/data'][fi][ROI[0]:ROI[1], ROI[2]:ROI[3]] for fi in good_frames])

whitefield  = f['/process_2/powder'][ROI[0]:ROI[1], ROI[2]:ROI[3]]
whitefield[whitefield==0] = 1.
whitefield  = whitefield.astype(np.float) / float(f['/entry_1/data_1/data'].shape[0])

# for now just divide the data by the whitefield
data = (data.astype(np.float) / whitefield.astype(np.float)).astype(np.float32)

# get the magnified sample-shifts 
# -------------------------------
# the x and y positions along the pixel directions
R_ss_fs = np.array([np.dot(b[i], R[i]) for i in range(len(R))])
R_ss_fs[:, 0] /= du[0]
R_ss_fs[:, 1] /= du[1]

# I want the x, y coordinates in scaled pixel units
# divide R by the scaled pixel size
R_ss_fs /= (df / z) * du

# offset the sample shifts so they start at zero
R_ss_fs[:, 0] -= np.max(R_ss_fs[:, 0])
R_ss_fs[:, 1] -= np.max(R_ss_fs[:, 1])

# get the coordinates of each pixel in each frame in pixel units
# --------------------------------------------------------------
# the regular pixel values
i, j = np.indices(data.shape[1 :])

# pixel offsets due to the phase gradients
dfs, dss = 0, 0

# make the object grid
Oss =  i.max() + np.max(dss) + np.max(np.abs(R_ss_fs[:, 0]))
Ofs =  j.max() + np.max(dfs) + np.max(np.abs(R_ss_fs[:, 1]))

O = np.zeros((int(round(Oss)), int(round(Ofs))), dtype=np.float32)
Oss, Ofs = np.indices(O.shape)

get_i_k = lambda k : Oss + dss + R_ss_fs[k, 0]
get_j_k = lambda k : Ofs + dfs + R_ss_fs[k, 1]

# now stitch send stuff to the gpu 
data_g    = ap.array(data.astype(np.float))
mask_g    = ap.array(np.ones_like(data[0]).astype(np.float))
Oss_g     = ap.array(Oss.astype(np.float).ravel())
Ofs_g     = ap.array(Ofs.astype(np.float).ravel())
R_ss_fs_g = ap.array(R_ss_fs.astype(np.float))
O_g       = ap.array(O.astype(np.float))

get_i_k_g = lambda k : Oss_g + dss + R_ss_fs_g[k, 0]
get_j_k_g = lambda k : Ofs_g + dfs + R_ss_fs_g[k, 1]

b    = None
norm = None
ds = []
for k in range(len(good_frames)):
    if b is None :
        b = af.approx2( data_g[k].d_array, get_j_k_g(k).d_array, get_i_k_g(k).d_array)
        norm = af.approx2( mask_g.d_array, get_j_k_g(k).d_array, get_i_k_g(k).d_array)
        #ds.append(np.array(b).reshape(O.shape))
    else :
        b += af.approx2( data_g[k].d_array, get_j_k_g(k).d_array, get_i_k_g(k).d_array)
        norm += af.approx2( mask_g.d_array, get_j_k_g(k).d_array, get_i_k_g(k).d_array)
        #ds.append(np.array(b).reshape(O.shape))
