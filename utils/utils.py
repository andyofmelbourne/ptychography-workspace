

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
    if (n-m) % 2 == 1 :
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
    import numpy as np
    i, j = np.indices(shape)
    i    = i - float(shape[0]-1)/2.
    j    = j - float(shape[1]-1)/2.
    ij   = i*1j + j
    r    = np.abs(   i*1j + j )
    r    = r / np.min(r[0, :])
    t    = np.angle( i*1j + j )

    unit_circle_rs = np.where(r <= 1.)
    r[r > 1]       = 0
    Z = np.zeros_like(r)
    
    if pixel_norm :
        dA = np.sqrt( np.pi / len(unit_circle_rs[0]))
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
            T = np.sin(nm[1] * t[unit_circle_rs])
        elif nm[1] > 0 :
            T = np.cos(nm[1] * t[unit_circle_rs])
        
        Z[unit_circle_rs] += ai * A * T * R * dA
    
    return r, t, Z


def make_Zernike_polynomial_cartesian(n, m):
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
    
    Returns 
    -------
    yx : ndarray, int
        
    """
    pass
