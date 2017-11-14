from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy
import optics
import pyximport; pyximport.install(setup_args={"include_dirs":numpy.get_include()})

from feature_matching import feature_map_cython
#from mean_filter import mean_filter
from get_Fresnel_pixel_shifts_cxi import get_Fresnel_pixel_shifts_cxi
from get_Fresnel_pixel_shifts_cxi import get_Fresnel_pixel_shifts_cxi_inverse

import numpy as np

def parse_parameters(config):
    """
    Parse values from the configuration file and sets internal parameter accordingly
    The parameter dictionary is made available to both the workers and the master nodes
    The parser tries to interpret an entry in the configuration file as follows:
    - If the entry starts and ends with a single quote, it is interpreted as a string
    - If the entry is the word None, without quotes, then the entry is interpreted as NoneType
    - If the entry is the word False, without quotes, then the entry is interpreted as a boolean False
    - If the entry is the word True, without quotes, then the entry is interpreted as a boolean True
    - If non of the previous options match the content of the entry, the parser tries to interpret the entry in order as:
        - An integer number
        - A float number
        - A string
      The first choice that succeeds determines the entry type
    """

    monitor_params = {}

    for sect in config.sections():
        monitor_params[sect]={}
        for op in config.options(sect):
            monitor_params[sect][op] = config.get(sect, op)
            if monitor_params[sect][op].startswith("'") and monitor_params[sect][op].endswith("'"):
                monitor_params[sect][op] = monitor_params[sect][op][1:-1]
                continue
            if monitor_params[sect][op] == 'None':
                monitor_params[sect][op] = None
                continue
            if monitor_params[sect][op] == 'False':
                monitor_params[sect][op] = False
                continue
            if monitor_params[sect][op] == 'True':
                monitor_params[sect][op] = True
                continue
            try:
                monitor_params[sect][op] = int(monitor_params[sect][op])
                continue
            except :
                try :
                    monitor_params[sect][op] = float(monitor_params[sect][op])
                    continue
                except :
                    # attempt to pass as an array of ints e.g. '1, 2, 3'
                    try :
                        l = monitor_params[sect][op].split(',')
                        temp = int(l[0])
                        monitor_params[sect][op] = np.array(l, dtype=np.int)
                        continue
                    except :
                        try :
                            l = monitor_params[sect][op].split(',')
                            temp = float(l[0])
                            monitor_params[sect][op] = np.array(l, dtype=np.float)
                            continue
                        except :
                            try :
                                l = monitor_params[sect][op].split(',')
                                if len(l) > 1 :
                                    monitor_params[sect][op] = [i.strip() for i in l]
                                continue
                            except :
                                pass

    return monitor_params

Zernike_index_names = {
        (0, 0)  : "Piston",
        (1, -1) : "tilt y",
        (1, 1)  : "tilt x",
        (2, -2) : "Astigmatism x",
        (2, 0)  : "Defocus",
        (2, 2)  : "Astigmatism y",
        (3, -3) : "Trefoil y",
        (3, -1) : "Primary y coma",
        (3,  1) : "Primary x coma",
        (3,  3) : "Trefoil x",
        (4, -4) : "Tetrafoil y",
        (4, -2) : "Secondary astigmatism y",
        (4,  0) : "Primary spherical",
        (4,  2) : "Secondary astigmatism x",
        (4,  4) : "Tetrafoil x",
        (5,  -5) : "Pentafoil y",
        (5,  -3) : "Secondary tetrafoil y",
        (5,  -1) : "Secondary coma y",
        (5,   1) : "Secondary coma x",
        (5,   3) : "Secondary tetrafoil x",
        (5,   5) : "Pentafoil x",
        (6,   -6) : "",
        (6,   -4) : "Secondary tetrafoil y",
        (6,   -2) : "Tertiary Astigmatism y",
        (6,    0) : "Secondary spherical",
        (6,    2) : "Tertiary Astigmatism y",
        (6,    4) : "Secondary tetrafoil x",
        (6,    6) : "",
        (7, -7) : "",
        (7, -5) : "",
        (7, -3) : "Tertiary trefoil y",
        (7, -1) : "Tertiary coma y",
        (7,  1) : "Tertiary coma x",
        (7,  3) : "Tertiary trefoil x",
        (7,  5) : "",
        (7,  7) : "",
        (8,  0) : "Tertiary spherical",
        }


def make_Noll_index_sequence(max_j):
    """
    Return a dictionary of tupples where each value is the 
    tupple (n, m), where (n, m) are the Zernike indices, and
    each key is the Noll index.
    
    The natural arrangement of the indices n (radial index) 
    and m (azimuthal index) of the Zernike polynomial Z(n,m) 
    is a triangle with row index n, in each row m ranging from 
    -n to n in steps of 2:
    (0,0)
    (1,-1) (1,1)
    (2,-2) (2,0) (2,2)
    (3,-3) (3,-1) (3,1) (3,3)
    (4,-4) (4,-2) (4,0) (4,2) (4,4)
    (5,-5) (5,-3) (5,-1) (5,1) (5,3) (5,5)
    (6,-6) (6,-4) (6,-2) (6,0) (6,2) (6,4) (6,6)
    (7,-7) (7,-5) (7,-3) (7,-1) (7,1) (7,3) (7,5) (7,7)
    
    For uses in linear algebra related to beam optics, a standard 
    scheme of assigning a single index j>=1 to each double-index 
    (n,m) has become a de-facto standard, proposed by Noll. The 
    triangle of the j at the equivalent positions reads
    1,
    3,2,
    5,4,6,
    9,7,8,10,
    15,13,11,12,14,
    21,19,17,16,18,20,
    27,25,23,22,24,26,28,
    35,33,31,29,30,32,34,36,
    which defines the OEIS entries. The rule of translation is that 
    odd j are assigned to m<0, even j to m>=0, and smaller j to smaller |m|.

    .. math:: Z^m_n(\rho, \theta) = R^m_n(\rho) e^{i m \theta}

    Parameters
    ----------
    max_j : int
        Maximum Noll index for the sequence.
    
    Returns
    -------
    Zernike_indices, dict
        A dictionary pair of {Noll_index : (n, m, name), ...} of length max_j, 
        where Noll_index is an int, and name is a string.

    Refernce
    --------
    https://oeis.org/A176988
    """
    Zernike_indices = {}
    n = 0
    j = 0
    js = []
    nms = []
    while j < max_j :
        # generate the sequence of ms for this row
        ms  = range(-n, n+1, 2)
        
        # append the list (n,m) tupples 
        nms += [(n, m) for m in ms]
        
        # generate the sequence of j's for this row
        jms = range(j+1, j+len(ms)+1)

        # remember the largest value
        j += len(ms)
        
        # assign js largest odd j --> smallest odd j
        js += [j for j in jms[::-1] if j % 2 == 1]
        
        # assign js smallest even j --> largest even j
        js += [j for j in jms if j % 2 == 0]
        
        # increment the row index
        n += 1
    
    # generate the dictionary 
    Zernike_indices = {}
    for j, nm in zip(js, nms):
        if nm in Zernike_index_names.keys() :
            Zernike_indices[j] = nm + (Zernike_index_names[nm],)
        else :
            Zernike_indices[j] = nm + ("",)
    
    return Zernike_indices
    
def make_Zernike_polynomial(n, m):
    """
    Given the Zernike indices n and m return the Zerike polynomial coefficients
    for the radial and azimuthal components.

    Z^m_n(r, \theta) = A^n_m cos(m \theta) R^m_n , for m >= 0
    
    Z^m_n(r, \theta) = A^n_m sin(m \theta) R^m_n , for m < 0
    
    R^m_n(r) = \sum_k=0^{(n-m)/2} \frac{(-1)^k (n-k)!}{k! ((n-m)/2 - k)! ((n-m)/2 - k))!} 
               r^{n-2k}

    A^n_m = \sqrt{(2n + 2)/(e_m \pi)}, e_m = 2 if m = 0, e_m = 1 if m != 0

    \iint Z^m_n(r, \theta) Z^m'_n'(r, \theta) r dr d\theta = \delta_{n-n'}\delta_{m-m'}

    Retruns :
    ---------
    p : list of integers
        The polynomial coefficients of R^n_m from largest to 0, e.g. if n = 4 and m = 2 then
        p_r = [4, 0, -3, 0, 0] representing the polynomial 4 r^4 - 3 r^3.

    A : float
        A^m_n, the normalisation factor, e.g. if n = 4 and m = 2 then
        A = \sqrt{10 / \pi}
    """
    if (n-m) % 2 == 1 or abs(m) > n or n < 0 :
        return [0], 0
    
    import math 
    fact = math.factorial 
    p = [0 for i in range(n+1)]

    for k in range((n-abs(m))//2+1):
        # compute the polynomial coefficient for order n - 2k 
        p[n-2*k] = (-1)**k * fact(n-k) / (fact(k) * fact((n+m)/2 - k) * fact((n-m)/2 - k))
    
    # compute the normalisation index
    if m is 0 :
        A = math.sqrt( float(n+1) / float(math.pi) )
    else :
        A = math.sqrt( float(2*n+2) / float(math.pi) )
    
    return p[::-1], A

def make_Zernike_phase(a = None, Noll = None, nms = None, shape=(256, 256), pixel_norm=False):
    """
    Evaluate the Zernike polynomials on the grid 'shape', by filling the 
    shape with a unit circle.
    
    if pixel_norm is True, then the Zernike polynomials are normalsed such that:
    np.sum(Z_n, Z_m) = 0, for n != m and
    np.sum(Z_n, Z_m) = 1, for n == m
    
    if pixel norm is False (default), then
    np.sum(Z_n, Z_m) = (number of pixels inside circular mask)**2, for n == m
    """
    x = np.linspace(-1, 1, shape[1])
    y = np.linspace(-1, 1, shape[0])
    ry, rx = np.meshgrid(y, x, indexing='ij')
    z      = rx + 1j * ry
    r      = np.abs(z)
    t      = np.angle(z)
    mask = (r <= 1.)
    
    unit_circle_rs = np.where(mask)
    Z = np.zeros_like(r)
    
    if pixel_norm :
        dA = np.sqrt( np.pi / float(np.sum(mask)))
    else :
        dA = 1

    if nms is None :
        j_nm_name = make_Noll_index_sequence(max(Noll))
        nms = []
        for j in Noll :
            nms.append(j_nm_name[j][:2])

    if a is None :
        a = np.ones((len(nms),), dtype=np.float)

    for ai, nm in zip(a, nms):
        # get the polynomial coefficients
        p, A = make_Zernike_polynomial(nm[0], nm[1])

        # evaluate the polynomial on the r theta grid
        R = np.polyval(p, r[unit_circle_rs])
        
        if nm[1] is 0 : 
            T = 1
        elif nm[1] < 0 :
            T = np.sin(abs(nm[1]) * t[unit_circle_rs])
        elif nm[1] > 0 :
            T = np.cos(abs(nm[1]) * t[unit_circle_rs])
        
        Z[unit_circle_rs] += ai * A * T * R * dA
    
    return r, t, Z

def make_Zernike_phase_cartesian(a = None, Noll = None, nms = None, shape=(256, 256), pixel_norm=False):
    """
    Evaluate the Zernike polynomials on the grid 'shape', by filling the 
    shape with a unit circle.
    
    if pixel_norm is True, then the Zernike polynomials are normalsed such that:
    np.sum(Z_n, Z_m) = 0, for n != m and
    np.sum(Z_n, Z_m) = 1, for n == m
    
    if pixel norm is False (default), then
    np.sum(Z_n, Z_m) = (number of pixels inside circular mask)**2, for n == m
    """
    x      = np.linspace(-1, 1, shape[1])
    y      = np.linspace(-1, 1, shape[0])
    ry, rx = np.meshgrid(y, x, indexing='ij')
    r      = np.abs(rx + 1j * ry)
    
    mask = (r <= 1.)
    
    if pixel_norm :
        dA = np.sqrt(np.pi / float(np.sum(mask)))
    else :
        dA = 1
    
    if nms is None :
        j_nm_name = make_Noll_index_sequence(max(Noll))
        nms = []
        for j in Noll :
            nms.append(j_nm_name[j][:2])
    
    # find the maximum order of the polynomials
    # -----------------------------------------
    n_max = max([nm[0] for nm in nms])
    
    # make the matrix mat[i, j] such that:
    #   Z[n,m](x, y) = mat[n, m] x**j y**i
    # ------------------------------------
    mat = np.zeros((n_max+1, n_max+1), dtype=np.float)
    
    if a is None :
        a = np.ones((len(nms),), dtype=np.float)
    
    for ai, nm in zip(a, nms):
        # get the polynomial coefficients
        p, A = make_Zernike_polynomial_cartesian(nm[0], nm[1], order = n_max+1)
        
        mat += p * ai * A * dA
    
    # evaluate the polynomial on the r theta grid
    Z = mask * np.polynomial.polynomial.polygrid2d(y, x, mat)
    
    return Z

def make_Zernike_phase_cartesian_rectangular(a = None, Noll = None, nms = None, shape=(256, 256), pixel_norm=False, dom=[-1, 1, -1, 1]):
    """
    Evaluate the Zernike polynomials on the grid 'shape', by filling the 
    shape with a unit circle.
    
    if pixel_norm is True, then the Zernike polynomials are normalsed such that:
    np.sum(Z_n, Z_m) = 0, for n != m and
    np.sum(Z_n, Z_m) = 1, for n == m
    
    if pixel norm is False (default), then
    np.sum(Z_n, Z_m) = (number of pixels inside circular mask)**2, for n == m
    """
    x      = np.linspace(dom[2], dom[3] , shape[1])
    y      = np.linspace(dom[0], dom[1], shape[0])
    
    if pixel_norm :
        dA = (x[-1] - x[0])/float(len(x)) * (y[-1] - y[0])/float(len(y)) #np.sqrt(np.pi / float(shape[0]*shape[1]))
    else :
        dA = 1
    
    if nms is None :
        j_nm_name = make_Noll_index_sequence(max(Noll))
        nms = []
        for j in Noll :
            nms.append(j_nm_name[j][:2])
    
    # find the maximum order of the polynomials
    # -----------------------------------------
    n_max = max([nm[0] for nm in nms])
    
    # make the matrix mat[i, j] such that:
    #   Z[n,m](x, y) = mat[n, m] x**j y**i
    # ------------------------------------
    mat = np.zeros((n_max+1, n_max+1), dtype=np.float)
    
    if a is None :
        a = np.ones((len(nms),), dtype=np.float)
    
    # generate the polynomial coefficients for a rectangular pupil
    basis = generate_rectangular_Zernike_polynomials(nms)
    
    for ai, b in zip(a, basis):
        # get the polynomial coefficients
        # p, A = make_Zernike_polynomial_cartesian(nm[0], nm[1], order = n_max+1)
        mat += b * ai * dA
    
    # evaluate the polynomial on the r theta grid
    Z = np.polynomial.polynomial.polygrid2d(y, x, mat)
    
    return y, x, mat, Z

def binomial(N, n):
    """ 
    Calculate binomial coefficient NCn = N! / (n! (N-n)!)

    Reference
    ---------
    PM 2Ring : http://stackoverflow.com/questions/26560726/python-binomial-coefficient
    """
    from math import factorial as fac
    try :
        binom = fac(N) // fac(n) // fac(N - n)
    except ValueError:
        binom = 0
    return binom

def pascal(m):
    """
    Print Pascal's triangle to test binomial()
    """
    for x in range(m + 1):
        print([binomial(x, y) for y in range(x + 1)])

def make_Zernike_polynomial_cartesian(n, m, order = None):
    """
    Given the Zernike indices n and m return the Zerike polynomial coefficients
    in a cartesian basis.

    The coefficients are stored in a yx matrix of the following form:
    
         1       x        x**2     x**3
    1    yx[0,0] yx[0, 1] yx[0, 2] yx[0, 3]
    y    yx[1,0] yx[1, 1] yx[1, 2] yx[1, 3]
    y**2 yx[2,0] yx[2, 1] yx[2, 2] yx[2, 3] ...
    ...
    
    such that Z^m_n = \sum_i \sum_j yx[i, j] y**i * x**j
    
    yx[i, j] is given by:

    Z^{m}_n  = R^m_n(r) cos(m \theta) 
    Z^{-m}_n = R^m_n(r) sin(m \theta) 
    
    Z^{m}_n  =  \sum_{k=0}^{(n-|m|)/2} (-1)^k (n - k)! / (k! ((n+|m|)/2 -k)! ((n-|m|)/2 -k)!) 
                \sum_{k'=0}^{|m|} binomial(|m|, k') * sin|cos((|m|-k') \pi/2) 
                \sum_{i=0}^{(n-|m|)/2 - k} binomial((n-|m|)/2 - k, i) 
                x^{2i + k'} y^{n - 2k - 2i - k'}
    
    where sin|cos = cos for m >= 0  
    and   sin|cos = sin for m < 0  
    
    Parameters
    ----------
    n, m : int, int
        Zernike indices
    
    order : int
        zero pads yx so that yx.shape = (order, order). If order is less than
        the maximum order of the polynomials then an error will be raised.
    
    Returns 
    -------
    yx : ndarray, int
    
    A : float
        A^m_n, the normalisation factor, e.g. if n = 4 and m = 2 then
        A = \sqrt{10 / \pi}
        
    Reference 
    -------
    For a slightly misguided approach see:
    Efficient Cartesian representation of Zernike polynomials in computer memory
    Hedser van Brug
    SPIE Vol. 3190 0277-786X/97
    """
    from math import factorial as fac
    import math
    
    if (n-m) % 2 == 1 or abs(m) > n or n < 0 :
        return np.array([0]), 0
    
    if m < 0 :
        t0 = math.pi / 2.
    else :
        t0 = 0
    
    m   = abs(m)
    
    if order is None :
        order = n + 1
    
    mat = np.zeros((order, order), dtype=np.int)
    
    for k in range((n-m)/2 + 1):
        a = (-1)**k * fac(n-k) / (fac(k) * fac((n+m)/2 - k) * fac( (n-m)/2 - k))
        for kk in range(m+1):
            b = int(round(math.cos((m-kk)*math.pi/2. - t0)))
            if b is 0 :
                continue
            b *= binomial(m, kk)
            ab = a*b
            l  = (n-m)/2 - k
            for i in range(l + 1):
                c    = binomial(l, i)
                abc  = ab*c
                powx = 2*i + kk
                powy = n - 2*k - 2*i - kk
                mat[powy, powx] += abc

    # compute the normalisation index
    if m is 0 :
        A = math.sqrt( float(n+1) / float(math.pi) )
    else :
        A = math.sqrt( float(2*n+2) / float(math.pi) )
    
    return mat, A

def polymul2d(a, b):
    # there must be a faster way...
    cl = []
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            for ii in range(b.shape[0]):
                for jj in range(b.shape[1]):
                    powy = i + ii
                    powx = j + jj
                    c = a[i,j]*b[ii,jj]
                    if abs(c) > 0 :
                        cl.append([powy, powx, c])
    cl = np.array(cl)
    c = np.zeros((int(np.max(cl[:,0]))+1, int(np.max(cl[:,1]))+1), dtype=a.dtype)
    for i, j, k in cl:
        c[int(i), int(j)] += k
    return c


def generate_rectangular_Zernike_polynomials(nms = None, Noll = None, order = None):
    # generate the n m indices from the Noll sequence
    # -----------------------------------------------
    if Noll is not None :
        j_nm_name = make_Noll_index_sequence(max(Noll))
        nms = []
        for j in Noll :
            nms.append(j_nm_name[j][:2])
    
    if order is not None :
        j_nm_name = make_Noll_index_sequence(order)
        nms = []
        for j in range(1, order+1) :
            nms.append(j_nm_name[j][:2])
    
    # find the maximum order of the polynomials
    # -----------------------------------------
    n_max = max([nm[0] for nm in nms])
    
    # Get the zernike polynomials in matrix form
    # ------------------------------------------
    vects = []
    for n, m in nms :
        mat, A = make_Zernike_polynomial_cartesian(n, m, order = n_max+1)
        vects.append(mat * A)
    
    # define the product method
    # -------------------------
    dom = [-1., 1., -1., 1.]
    from numpy.polynomial import polynomial as P
    def product(a, b):
        c = polymul2d(a, b)
        c = P.polyint(c, lbnd = dom[0], axis=0)
        c = P.polyint(c, lbnd = dom[2], axis=1)
        v = P.polyval2d(dom[1], dom[3], c)
        return v
    
    basis = Gram_Schmit_orthonormalisation(vects, product)
    #return basis, vects, product
    return basis


def Gram_Schmit_orthonormalisation(vects, product):
    """
    The following algorithm implements the stabilized Gram-Schmidt orthonormalization.
    
    The vectors 'vects' are replaced by orthonormal vectors which span the same subspace:
        vects = [v0, v1, ..., vN]
        
        u0 = v1
        ...
        uk = vk - \sum_j=0^k-1 proj(uj, vk)

        The basis vectors are then:
        ek = uk / norm(uk)

        where proj(uj, uk) = product(uk, uj) / product(uj, uj) * uj
        and   norm(uk)     = product(uj, uj)

    For the modified algorithm :
        uk = vk - \sum_j=0^k-1 proj(uj, vk)
        
        is replaced by:
                  uk       = uk_{k-2} - proj(uk-1, uk_{k-2})
            where uk_{0}   = vk       - proj(u0, vk)
            and   uk_{k+1} = uk_{k}   - proj(uk, uk_{k})
    
    Parameters
    ----------
    vects : sequence of objects
        The objects in the sequence 'O' must be acceptable to the function 'product'
        and they must have add/subtract/scalar multiplication and scalar division 
        methods.

    product : function of two arguments
        Must take two 'vectors' of type vn and uk and calculate the vector product.
        E.g. product([1,2,3], [4,5,6]) = 1*4 + 2*5 + 3*6. 
    
    Returns
    -------
    basis : sequence of objects
        The orthonormal basis vectors that span the subspace given by 'vects'.
    """
    import math
    import copy
    basis = [vects[0] / math.sqrt(product(vects[0], vects[0]))]
    for k in range(1, len(vects)):
        u = vects[k]
        for j in range(k):
            u = u - basis[j] * product(basis[j], u) 
           
        basis.append(u / math.sqrt(product(u, u)))
    
    return basis


def multiroll(x, shift, axis=None):
    """Roll an array along each axis.

    Thanks to: Warren Weckesser, 
    http://stackoverflow.com/questions/30639656/numpy-roll-in-several-dimensions
    
    
    Parameters
    ----------
    x : array_like
        Array to be rolled.
    shift : sequence of int
        Number of indices by which to shift each axis.
    axis : sequence of int, optional
        The axes to be rolled.  If not given, all axes is assumed, and
        len(shift) must equal the number of dimensions of x.

    Returns
    -------
    y : numpy array, with the same type and size as x
        The rolled array.

    Notes
    -----
    The length of x along each axis must be positive.  The function
    does not handle arrays that have axes with length 0.

    See Also
    --------
    numpy.roll

    Example
    -------
    Here's a two-dimensional array:

    >>> x = np.arange(20).reshape(4,5)
    >>> x 
    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19]])

    Roll the first axis one step and the second axis three steps:

    >>> multiroll(x, [1, 3])
    array([[17, 18, 19, 15, 16],
           [ 2,  3,  4,  0,  1],
           [ 7,  8,  9,  5,  6],
           [12, 13, 14, 10, 11]])

    That's equivalent to:

    >>> np.roll(np.roll(x, 1, axis=0), 3, axis=1)
    array([[17, 18, 19, 15, 16],
           [ 2,  3,  4,  0,  1],
           [ 7,  8,  9,  5,  6],
           [12, 13, 14, 10, 11]])

    Not all the axes must be rolled.  The following uses
    the `axis` argument to roll just the second axis:

    >>> multiroll(x, [2], axis=[1])
    array([[ 3,  4,  0,  1,  2],
           [ 8,  9,  5,  6,  7],
           [13, 14, 10, 11, 12],
           [18, 19, 15, 16, 17]])

    which is equivalent to:

    >>> np.roll(x, 2, axis=1)
    array([[ 3,  4,  0,  1,  2],
           [ 8,  9,  5,  6,  7],
           [13, 14, 10, 11, 12],
           [18, 19, 15, 16, 17]])

    """
    from itertools import product
    x = np.asarray(x)
    if axis is None:
        if len(shift) != x.ndim:
            raise ValueError("The array has %d axes, but len(shift) is only "
                             "%d. When 'axis' is not given, a shift must be "
                             "provided for all axes." % (x.ndim, len(shift)))
        axis = range(x.ndim)
    else:
        # axis does not have to contain all the axes.  Here we append the
        # missing axes to axis, and for each missing axis, append 0 to shift.
        missing_axes = set(range(x.ndim)) - set(axis)
        num_missing = len(missing_axes)
        axis = tuple(axis) + tuple(missing_axes)
        shift = tuple(shift) + (0,)*num_missing

    # Use mod to convert all shifts to be values between 0 and the length
    # of the corresponding axis.
    shift = [s % x.shape[ax] for s, ax in zip(shift, axis)]

    # Reorder the values in shift to correspond to axes 0, 1, ..., x.ndim-1.
    shift = np.take(shift, np.argsort(axis))

    # Create the output array, and copy the shifted blocks from x to y.
    y = np.empty_like(x)
    src_slices = [(slice(n-shft, n), slice(0, n-shft))
                  for shft, n in zip(shift, x.shape)]
    dst_slices = [(slice(0, shft), slice(shft, n))
                  for shft, n in zip(shift, x.shape)]
    src_blks = product(*src_slices)
    dst_blks = product(*dst_slices)
    for src_blk, dst_blk in zip(src_blks, dst_blks):
        y[dst_blk] = x[src_blk]

    return y
