# find the z-plane of maximum intensity
## fit a gaussian to the spot

# 2D parameter search in dx and dy to find the brightest spot without ast.
## fit a gaussian to the spot

# produce 1D seperable phase profiles
## remove tilt and defocus to produce the flattest profile in some subregion

# parameters
fnam      = '../hdf5/cmll_0001/cmll_0001_.cxi'
phase     = '/process_3/Zernike/phase'
intensity = '/process_2/powder'
h5_group  = '/process_3/profiles'
ROI       = [320,470,560,730]
df_range  = [-2.0e-3, 2.0e-3, 30]
subsample = 2
spot_window = [64,64]



import h5py
import numpy as np

def qgrid(shape, E, du):
    import scipy.constants as sc
    lamb = sc.h * sc.c / E
    
    dq = du / (lamb * z)
    i = np.fft.fftfreq(shape[0], 1/float(shape[0])) * dq[0]
    j = np.fft.fftfreq(shape[1], 1/float(shape[1])) * dq[1]
    return np.meshgrid(i, j, indexing='ij'), lamb

def interpolate_array(a, fac):
    # subsample to prevent wrapping in real-space
    i, j   = np.arange(a.shape[0]), np.arange(a.shape[1])
    i2, j2 = np.linspace(i[0], i[-1], subsample*len(i)), np.linspace(j[0], j[-1], subsample*len(j))
    i2, j2 = np.meshgrid(i2, j2, indexing='ij')

    from scipy.interpolate import interpn
    s = interpn((i,j), a, np.array([i2,j2]).T)
    return s

def zero_pad(a):
    # zero pad P for propagation
    P2 = np.zeros( (2*a.shape[0], 2*a.shape[1]), dtype=a.dtype)
    P2[:a.shape[0], :a.shape[1]] = a
    P2 = np.roll(P2, a.shape[0]//2, 0)
    P2 = np.roll(P2, a.shape[1]//2, 1)
    return P2

def defocus_sweep(P, df_range, E, z, du):
    (i, j), lamb = qgrid(P.shape, E, du)     
    ij         = i**2 + j**2
    
    Ps = []
    dfs = np.linspace(df_range[0], df_range[1], int(df_range[2]))
    for ii, df in enumerate(dfs):
        #print(ii, df) 
        exp  = np.exp(-1J * np.pi * lamb * df * ij)
        P2 = np.fft.ifftshift(P) * exp
        P2 = np.fft.fftshift(np.fft.ifftn(P2))
        Ps.append(P2.copy())
        
    dxy = lamb * z / (np.array(P.shape) * du)
    vox = np.array([dxy[0], dxy[1], dfs[1] - dfs[0]])
    return np.array(Ps), vox, dfs

def fit_gaus(a, dxy = np.array([1,1])):
    import math
    b    = a.ravel()
    i, j = np.indices(a.shape)
    ij   = np.vstack((i.ravel(), j.ravel())).astype(np.float)
    
    def gaus(params, ij = ij):
        """
        params = [scale, i-shift, j-shift, i-sigma, j-sigma, theta]
        """
        ij2    = ij.copy()
        
        #rotate
        t  = params[5]
        R  = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
        ij2 = np.dot(R, ij2)
        
        #shift
        ij2[0] = (ij2[0] - params[1]) / params[3]
        ij2[1] = (ij2[1] - params[2]) / params[4]
        
        # calc
        g = params[0] * np.exp(-(ij2[0]**2 + ij2[1]**2)/2.)
        return g
    
    def fun(params):
        # residuals
        return gaus(params) - b
    
    from scipy.optimize import least_squares
    x0  = [np.max(a), a.shape[0]/2., a.shape[1]/2., 1., 4., math.pi / 4.]
    res = least_squares(fun, x0)
    
    # calculate the major / minor sigma in real units
    t = res.x[5]
    R  = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    s1 = res.x[1]*np.dot(R, [1, 0])
    s2 = res.x[2]*np.dot(R, [0, 1])
    
    s1 = np.sqrt((dxy[0]*s1[0])**2 + (dxy[1]*s1[1])**2)
    s2 = np.sqrt((dxy[0]*s2[0])**2 + (dxy[1]*s2[1])**2)
    
    return res.x, [s1,s2], gaus(res.x).reshape(a.shape)
    
    
def get_spot_size_from_focus_sweep(P3D, vox, spot_window, maxs):
    # find the plane of maximum spot intensity
    k_min  = np.argmax(maxs)
    
    # find the pixel of maximum intensity
    ij_min = np.argmax(np.abs(P3D[k_min]))
    ij_min = np.unravel_index(ij_min, P3D[k_min].shape)

    # cut out a window 
    spot = P3D[k_min, ij_min[0] - spot_window[0]//2 : ij_min[0] + spot_window[0]//2, 
                      ij_min[1] - spot_window[1]//2 : ij_min[1] + spot_window[1]//2]

    params, spot_size, fit = fit_gaus(np.abs(spot)**2, vox[:2])
    
    spot_loc = np.array([params[1] + ij_min[0] - spot_window[0]//2, 
                         params[2] + ij_min[1] - spot_window[1]//2])
    return spot_size, np.array([np.abs(spot)**2, fit]), spot_loc


def corr_ast_by_max_spot_int(P, df_range, E, z, du, maxs):
    """
    It is assumed that both the horizontal and vertical foci are within 
    the bounds df_range[0] --> df_range[1].
    """
    dfs = np.linspace(df_range[0], df_range[1], df_range[2])
    
    maxs = np.max(np.abs(Ps), axis=(1,2))

    # fit 2 gaussians of the maxs line to estimate the horizontal and vertical foci spot separation
    def gaus(params, dfs = dfs):
        """
        params = [scale1, sig1, shift1, level1]
        """
        return params[0] * np.exp(-(dfs - params[2])**2 / (2. * params[1]**2))

    def gauss_1d(params, dfs = dfs):
        """
        params = [scale1, sig1, shift1, scale2, sig2, shift2, level]
        """
        return gaus(params[:3]) + gaus(params[3:-1]) + params[-1]

    def fun(params):
        return maxs - gauss_1d(params)
    
    sig = 0.1 * (df_range[1] - df_range[0])
    lev = np.min(maxs)
    scale = np.max(maxs) - lev
    shift1 = df_range[0] + 0.3 * (df_range[1] - df_range[0])
    shift2 = df_range[0] + 0.7 * (df_range[1] - df_range[0])
    x0  = [scale, sig, shift1, scale, sig, shift2, lev]

    from scipy.optimize import least_squares
    res = least_squares(fun, x0)

    # now figure out which foci is x and which is y
    i1 = np.argmin(np.abs(dfs - res.x[2]))
    i2 = np.argmin(np.abs(dfs - res.x[5]))

    I1, I2 = np.abs(Ps[i1])**2, np.abs(Ps[i2])**2

    if np.var(np.sum(I1, axis=0)) > np.var(np.sum(I1, axis=1)):
        dfss = res.x[2]
        dffs = res.x[5]
    else :
        dfss = res.x[5]
        dffs = res.x[2]

    # now do repeated through focal sweeps to find the x-y separation
    (i, j), lamb = qgrid(P.shape, E, du)     

    P2 = np.fft.ifftshift(P)

    def fun2(params, lamb = lamb, i = i, j = j, P2=P2):
        exp  = np.exp(1J * np.pi * lamb * (params[0] * i**2 + params[1] * j**2))
        P3   = np.fft.ifftn(P2 * exp)
        return -np.max( (P3 * P3.conj()).real )

    from scipy.optimize import minimize
    x0 = np.array([dfss, dffs])
    res = minimize(fun2, x0, bounds = [(dfs[0], dfs[-1]), (dfs[0], dfs[-1])])

    exp  = np.exp(1J * np.pi * lamb * (res.x[0] * i**2 + res.x[1] * j**2))
    return P * np.fft.fftshift(exp), res.x
        

if __name__ == '__main__':
    # Input
    f = h5py.File(fnam, 'r')
    p0    = f[phase][()][ROI[0]:ROI[1], ROI[2]:ROI[3]].astype(np.float)
    I0    = f[intensity][()][ROI[0]:ROI[1], ROI[2]:ROI[3]].astype(np.float)
    mask  = f['/process_3/mask'][()][ROI[0]:ROI[1], ROI[2]:ROI[3]]
    du0 = np.array([f['/entry_1/instrument_1/detector_1/y_pixel_size'][()], f['/entry_1/instrument_1/detector_1/x_pixel_size'][()]])
    z   = f['/entry_1/instrument_1/detector_1/distance'][()]
    E   = f['/entry_1/instrument_1/source_1/energy'][()]
    f.close()

    I   = interpolate_array(mask * I0, subsample)
    p   = interpolate_array(p0, subsample)
    du  = du0 / subsample

    P = np.sqrt(I) * np.exp(1.0J * p)
    P = zero_pad(P)
    
    Ps, vox, dfs = defocus_sweep(P, df_range, E, z, du)
    maxs = np.max(np.abs(Ps), axis=(1,2))

    spot_size, spot_data_fit, spot_loc = get_spot_size_from_focus_sweep(Ps, vox, spot_window, maxs)

    # map spot_loc into xy fftshifted coords
    spot_loc_xy = [(spot_loc[0] - (P.shape[0]//2))* vox[0], (spot_loc[1] - (P.shape[1]//2))* vox[1]]
    
    # add the phase offset corresponding to the maximum spot size
    df         = dfs[np.argmax(maxs)]
    (i, j), lamb = qgrid(p0.shape, E, du0)
    phase_def  = p0 + np.fft.fftshift(np.pi * lamb * df * (i**2 + j**2) \
                                      #- 2. * np.pi * i * spot_loc_xy[0] \
                                      #- 2. * np.pi * j * spot_loc_xy[1] \
                                      )
    
    P_corr, df_ss_fs = corr_ast_by_max_spot_int(P, df_range, E, z, du, maxs)

    Ps_corr, vox, dfs = defocus_sweep(P_corr, df_range, E, z, du)
    maxs = np.max(np.abs(Ps_corr), axis=(1,2))

    spot_size_corr, spot_data_fit_corr, spot_loc = get_spot_size_from_focus_sweep(Ps_corr, vox, spot_window, maxs)

    # map spot_loc into xy fftshifted coords
    spot_loc_xy = [(spot_loc[0] - (P.shape[0]//2))* vox[0], (spot_loc[1] - (P.shape[1]//2))* vox[1]]
    
    # add the phase offset corresponding to the maximum spot size
    (i, j), lamb = qgrid(p0.shape, E, du0)
    phase_corr   = p0 + np.fft.fftshift(np.pi * lamb * (df_ss_fs[0]*i**2 + df_ss_fs[1]*j**2)
                                        #- 2. * np.pi * i * spot_loc_xy[0] \
                                        #- 2. * np.pi * j * spot_loc_xy[1] \
                                        )
    ramp_corr_ss = - 2. * np.pi * i[:,0] * spot_loc_xy[0]
    ramp_corr_fs = - 2. * np.pi * j[0,:] * spot_loc_xy[1]

    # output
    import os
    g = h5py.File(fnam)
    outputdir = os.path.split(fnam)[0]

    group = h5_group
    if group not in g:
        print(g.keys())
        g.create_group(group)

    # 3D propagation profile
    key = h5_group+'/2D_focus_sweep/3D_profile'
    if key in g :
        del g[key]
    g[key] = (np.abs(np.array(Ps))**2).astype(np.float16)

    key = h5_group+'/2D_focus_sweep/vox'
    if key in g :
        del g[key]
    g[key] = vox

    key = h5_group+'/2D_focus_sweep/spot_size'
    if key in g :
        del g[key]
    g[key] = np.array(spot_size)

    key = h5_group+'/2D_focus_sweep/spot'
    if key in g :
        del g[key]
    g[key] = spot_data_fit

    key = h5_group+'/2D_focus_sweep/phase'
    if key in g :
        del g[key]
    g[key] = phase_def

    key = h5_group+'/2D_focus_sweep/phases_1d_ss'
    if key in g :
        del g[key]
    g[key] = np.mean(phase_def, axis=1)

    key = h5_group+'/2D_focus_sweep/phases_1d_fs'
    if key in g :
        del g[key]
    g[key] = np.mean(phase_def, axis=0)


    # 3D propagation profile, astigmatism corrected 
    key = h5_group+'/2D_focus_sweep_corr/3D_profile'
    if key in g :
        del g[key]
    g[key] = (np.abs(np.array(Ps_corr))**2).astype(np.float16)

    key = h5_group+'/2D_focus_sweep_corr/vox'
    if key in g :
        del g[key]
    g[key] = vox

    key = h5_group+'/2D_focus_sweep_corr/spot_size'
    if key in g :
        del g[key]
    g[key] = np.array(spot_size_corr)

    key = h5_group+'/2D_focus_sweep_corr/spot'
    if key in g :
        del g[key]
    g[key] = spot_data_fit_corr

    key = h5_group+'/2D_focus_sweep_corr/phase'
    if key in g :
        del g[key]
    g[key] = phase_corr

    key = h5_group+'/2D_focus_sweep_corr/phases_1d_ss'
    if key in g :
        del g[key]
    g[key] = np.mean(phase_corr, axis=1)

    key = h5_group+'/2D_focus_sweep_corr/phases_1d_fs'
    if key in g :
        del g[key]
    g[key] = np.mean(phase_corr, axis=0)


    g.close()

