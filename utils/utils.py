

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
        
        print nms[-len(ms):]
        print js[-len(ms):]
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
    pass

def make_Zernike_phase(Noll = None, nms = None):
    pass
