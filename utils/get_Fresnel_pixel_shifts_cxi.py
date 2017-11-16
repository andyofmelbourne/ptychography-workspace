import numpy as np
import scipy.constants as sc

def get_Fresnel_pixel_shifts_cxi(f, good_frames=None, df=None, offset_to_zero=True):
    du = np.array([f['/entry_1/instrument_1/detector_1/y_pixel_size'][()], f['/entry_1/instrument_1/detector_1/x_pixel_size'][()]])
    z  = f['/entry_1/instrument_1/detector_1/distance'][()]
    E  = f['/entry_1/instrument_1/source_1/energy'][()]
    wavelen = sc.h * sc.c / E
    
    if good_frames is None :
        good_frames = range(f['/entry_1/data_1/data'].shape[0])
    
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

def get_Fresnel_pixel_shifts_cxi_inverse(R_ss_fs, f, good_frames=None, df=None, offset_to_zero=True, remove_affine = False):
    du = np.array([f['/entry_1/instrument_1/detector_1/y_pixel_size'][()], f['/entry_1/instrument_1/detector_1/x_pixel_size'][()]])
    z  = f['/entry_1/instrument_1/detector_1/distance'][()]
    E  = f['/entry_1/instrument_1/source_1/energy'][()]
    wavelen = sc.h * sc.c / E
    
    if good_frames is None :
        good_frames = range(f['/entry_1/data_1/data'].shape[0])
    
    b           = f['/entry_1/instrument_1/detector_1/basis_vectors'][good_frames]

    R_ss_fs_out = R_ss_fs.astype(np.float).copy()
    
    # un-offset
    if offset_to_zero :
        R_ss_fs0, dx = get_Fresnel_pixel_shifts_cxi(f, good_frames, df, offset_to_zero = False)
        R_ss_fs_out[:, 0] -= np.max(R_ss_fs_out[:, 0])
        R_ss_fs_out[:, 1] -= np.max(R_ss_fs_out[:, 1])
        
        R_ss_fs_out[:, 0] += np.max(R_ss_fs0[:, 0])
        R_ss_fs_out[:, 1] += np.max(R_ss_fs0[:, 1])
    
    # un-scale
    R_ss_fs_out *= (df / z) * du
    
    # unfortunately we cannot invert from pixel shifts to xyz 
    # this is only possible if the detector lies in the xy plane
    R_ss_fs_out *= du
    
    #print('\ninverting from sample coords to detector coords:')
    R0 = f['/entry_1/sample_3/geometry/translation'][()]
    R  = R0.copy()
    for i in range(R_ss_fs_out.shape[0]):
        Ri, r, rank, s = np.linalg.lstsq(b[i][:, :2], R_ss_fs_out[i])
        R[good_frames[i]][:2] = Ri
        #print(R_ss_fs_out[i], '-->', Ri)
    
    if remove_affine :
        R = remove_affine_transformation(R0, R)
    
    return R


def remove_affine_transformation(R0, R1):
    A = np.vstack([R1[:, 0], R1[:, 1], np.ones(len(R1))]).T
    ATx, r, rank, s = np.linalg.lstsq(A, R0[:, 0])
    ATy, r, rank, s = np.linalg.lstsq(A, R0[:, 1])
    
    Rout = R1.copy()
    Rout[:, 0] = np.dot(A, ATx)
    Rout[:, 1] = np.dot(A, ATy)

    print(ATx)
    print(ATy)
    return Rout
    

