import utils
from utils.utils import *
import numpy as np

def make_Zernike_grads(mask, roi = None, max_order = 100, return_grids = False, return_basis = False, yx_bounds = None, test = False):
    if return_grids :
        basis, basis_grid, y, x = make_Zernike_basis(mask, roi, max_order, return_grids, yx_bounds, test)
    else :
        basis = make_Zernike_basis(mask, roi, max_order, return_grids, yx_bounds, test)

    # calculate the x and y gradients
    from numpy.polynomial import polynomial as P
    
    # just a list of [(grad_ss, grad_fs), ...] where the grads are in a polynomial basis
    grads = [ (P.polyder(b, axis=0), P.polyder(b, axis=1)) for b in basis ]

    if return_grids :
        # just a list of [(grad_ss, grad_fs), ...] where the grads are evaluated on a y, x grid
        grad_grids = [(P.polygrid2d(y, x, g[0]), P.polygrid2d(y, x, g[1])) for g in grads]

        if return_basis :
            return grads, grad_grids, basis, basis_grid
        else :
            return grads, grad_grids
    else :
        if return_basis :
            return grads, basis
        else :
            return grads

def make_Zernike_basis(mask, roi = None, max_order = 100, return_grids = False, yx_bounds = None, test = False):
    """
    Make Zernike basis functions, such that:
        np.sum( Z_i * Z_j * mask) = delta_ij

    Returns
    -------
    basis_poly : list of arrays
        The basis functions in a polynomial basis.

    basis_grid : list of arrays
        The basis functions evaluated on the cartesian grid
    """
    shape = mask.shape
    
    # list the Zernike indices in the Noll indexing order:
    # ----------------------------------------------------
    Noll_indices = make_Noll_index_sequence(max_order)
    
    # set the x-y values and scale to the roi
    # ---------------------------------------
    if roi is None :
        roi = [0, shape[0]-1, 0, shape[1]-1]
    
    sub_mask  = mask[roi[0]:roi[1]+1, roi[2]:roi[3]+1] 
    sub_shape = sub_mask.shape
    
    if yx_bounds is None :
        if (roi[1] - roi[0]) > (roi[3] - roi[2]) :
            m = float(roi[1] - roi[0]) / float(roi[3] - roi[2])
            yx_bounds = [-m, m, -1., 1.]
        else :
            m = float(roi[3] - roi[2]) / float(roi[1] - roi[0])
            yx_bounds = [-1., 1., -m, m]
    
    dom = yx_bounds
    y = ((dom[1]-dom[0])*np.arange(shape[0]) + dom[0]*roi[1]-dom[1]*roi[0])/(roi[1]-roi[0])
    x = ((dom[3]-dom[2])*np.arange(shape[1]) + dom[2]*roi[3]-dom[3]*roi[2])/(roi[3]-roi[2])

    # define the area element
    # -----------------------
    dA = (x[1] - x[0]) * (y[1] - y[0])
    
    # generate the Zernike polynomials in a cartesian basis:
    # ------------------------------------------------------
    Z_polynomials = []
    for j in range(1, max_order+1):
        n, m, name           = Noll_indices[j]
        mat, A               = make_Zernike_polynomial_cartesian(n, m, order = max_order)
        Z_polynomials.append(mat * A * dA)
    
    # define the product method
    # -------------------------
    from numpy.polynomial import polynomial as P
    def product(a, b):
        c = P.polygrid2d(y[roi[0]:roi[1]+1], x[roi[2]:roi[3]+1], a)
        d = P.polygrid2d(y[roi[0]:roi[1]+1], x[roi[2]:roi[3]+1], b)
        v = np.sum(sub_mask * c * d)
        return v
    
    basis = Gram_Schmit_orthonormalisation(Z_polynomials, product)
    
    # test the basis function
    if test :
        print '\n\nbasis_i, basis_j, product(basis_i, basis_j)'
        for i in range(len(basis)) :
            for j in range(len(basis)) :
                print i, j, product(basis[i], basis[j])

    if return_grids :
        basis_grid = [P.polygrid2d(y, x, b) for b in basis]
        
        if test :
            print '\n\nbasis_i, basis_j, np.sum(mask * basis_i * basis_j)'
            for i in range(len(basis_grid)) :
                for j in range(len(basis_grid)) :
                    print i, j, np.sum(mask * basis_grid[i] * basis_grid[j])
        return basis, basis_grid, y, x
    else :
        return basis

def fit_Zernike_coefficients(phase, mask = 1, roi = None, max_order = 100, yx_bounds=None):
    """
    Find cof such that:
        \sum_n cof_n * Z_n[i, j] = phase[i, j]
    
    The Z_n are formed by orthonormalising the Zernike polynomials on the mask.
    The x, y coordinates are scaled and shifted inside the roi such that the 
    smallest dimension is scaled from -1 to 1 and the other in proportion.
    """
    if roi is None :
        roi = [0, shape[0]-1, 0, shape[1]-1]

    if mask is 1 :
        mask = np.ones_like(phase, dtype=np.bool)
    
    sub_mask = np.zeros_like(mask)
    sub_mask[roi[0]:roi[1]+1, roi[2]:roi[3]+1] = mask[roi[0]:roi[1]+1, roi[2]:roi[3]+1]

    basis, basis_grid, y, x = make_Zernike_basis(mask, roi = roi, \
                                           max_order = max_order, return_grids = True, \
                                           yx_bounds = yx_bounds)
    
    Zernike_coefficients = [np.sum(b * sub_mask * phase) for b in basis_grid]
    
    return Zernike_coefficients


if __name__ == '__main__':
    # ----------------------------------------------------------
    # fit Zernike coefficients to a phase profile for arbitrary 
    # aperture dimensions and with masked pixels
    # ----------------------------------------------------------
    print 'fiting Zernike coefficients to a phase profile for arbitrary'
    print 'aperture dimensions and with masked pixels...'
    shape = (256, 256)
    #roi   = [64, 192, 0, 256]
    roi   = [0, 255, 0, 255]
    
    # stretched domain
    dom_st   = [-1., 1., -1., 1.]
    
    # circle in rectangle domain
    dom_sm   = [-1., 1., -2., 2.]
    
    # rectangle in circle domain
    rat = float(roi[1]-roi[0])/float(roi[3]-roi[2])
    x   = np.sqrt(1. / (1. + rat**2))
    y   = rat * x
    dom_la = [-y, y, -x, x]

    dom = dom_la
    
    #mask  = np.ones(shape, dtype=np.bool)
    mask  = np.random.random( shape ) > 0.2

    # make the phase with the same basis functions as those that are
    # fit, in order to compare coefficients
    Zernike_coefficients = np.random.random((36,))
    basis, basis_grid, y, x = make_Zernike_basis(mask, roi = roi, \
                                           max_order = len(Zernike_coefficients), return_grids = True, \
                                           yx_bounds = dom)
    
    phase = np.sum([Z * b for Z, b in zip(Zernike_coefficients, basis_grid) ], axis=0)
    phase *= mask
    
    fit_coef  = fit_Zernike_coefficients(phase, mask = mask, max_order = 40, roi=roi, yx_bounds=dom)
    
    phase_ret = np.sum([Z * b for Z, b in zip(fit_coef, basis_grid) ], axis=0)
    
    print 'Success?'
    print 'coefficients == fit coefficients?', np.allclose(Zernike_coefficients, fit_coef[:len(Zernike_coefficients)])
    print 'phase        == fit phase       ?', np.allclose(phase, mask*phase_ret)

    # ----------------------------------------------------------
    # fit Zernike coefficients to phase gradient profiles for 
    # arbitrary aperture dimensions and with masked pixels
    # ----------------------------------------------------------
    #Bdy, Bdx, Bdy_grid, Bdx_grid = make_Zernike_grad(basis, y, x
    grads, grad_grids = make_Zernike_grads(mask, roi = roi, max_order = 36, return_grids = True, yx_bounds = dom, test = False)
