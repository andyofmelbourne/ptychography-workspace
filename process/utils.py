import numpy as np
import scipy.constants as sc

def get_Fresnel_pixel_shifts_cxi(f, good_frames=None, df=None, offset_to_zero=True):
    du = np.array([f['/entry_1/instrument_1/detector_1/y_pixel_size'][()], f['/entry_1/instrument_1/detector_1/x_pixel_size'][()]])
    z  = f['/entry_1/instrument_1/detector_1/distance'][()]
    E  = f['/entry_1/instrument_1/source_1/energy'][()]
    wavelen = sc.h * sc.c / E
    
    if good_frames is None :
        good_frames = list(f['/entry_1/data_1/data'].shape[0])
    
    b           = f['/entry_1/instrument_1/detector_1/basis_vectors'][good_frames]
    R           = f['/entry_1/sample_3/geometry/translation'][good_frames]
    
    # get the magnified sample-shifts 
    # -------------------------------
    # the x and y positions along the pixel directions
    R_ss_fs = np.array([np.dot(b[i], R[i]) for i in range(len(R))])
    R_ss_fs[:, 0] /= du[0]
    R_ss_fs[:, 1] /= du[1]

    # get the focus to sample distance
    if df is None :
        df = R[0,2]
    
    # I want the x, y coordinates in scaled pixel units
    # divide R by the scaled pixel size
    R_ss_fs /= (df / z) * du
    
    # offset the sample shifts so they start at zero
    if offset_to_zero :
        R_ss_fs[:, 0] -= np.max(R_ss_fs[:, 0])
        R_ss_fs[:, 1] -= np.max(R_ss_fs[:, 1])
    return R_ss_fs, (df / z) * du
