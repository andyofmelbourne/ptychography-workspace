import numpy as np
cimport numpy as np

from libc.math cimport sin, cos, acos, exp, sqrt, fabs, M_PI

FLOAT = np.float
INT   = np.int

ctypedef np.float_t FLOAT_t
ctypedef np.int_t INT_t

cimport cython
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def feature_map_cython(np.ndarray[FLOAT_t, ndim=2] Od, np.ndarray[FLOAT_t, ndim=2] O, \
                np.ndarray[INT_t, ndim=3] X_ij, np.ndarray[INT_t, ndim=2] mask, \
                np.ndarray[INT_t, ndim=2] mask_O, \
                int i, int j, int window=10, int search_window=20, int steps = 1, \
                int offset_i = 0, int offset_j = 0):

    if Od.shape[0] > O.shape[0] or Od.shape[1] > O.shape[1] :
        raise ValueError("Od.shape should be less than O.shape for both axes")

    cdef np.ndarray[INT_t, ndim=3] X_ij_new   = np.zeros( (X_ij.shape[0], X_ij.shape[1], X_ij.shape[2]), dtype=INT)
    cdef np.ndarray[FLOAT_t, ndim=2] ncc      = np.zeros( (Od.shape[0], Od.shape[1]), dtype=FLOAT)
    cdef np.ndarray[FLOAT_t, ndim=2] data     = np.zeros( (window, window), dtype=FLOAT)
    cdef np.ndarray[FLOAT_t, ndim=2] data_O   = np.zeros( (window, window), dtype=FLOAT)
    cdef np.ndarray[INT_t, ndim=2] imap       = np.zeros( (Od.shape[0], Od.shape[1]), dtype=INT)#ii + X_ij[0] + i
    cdef np.ndarray[INT_t, ndim=2] jmap       = np.zeros( (Od.shape[0], Od.shape[1]), dtype=INT)#jj + X_ij[1] + j
    cdef np.ndarray[INT_t, ndim=1] i_d        = np.zeros((3,), dtype=INT)
    cdef np.ndarray[INT_t, ndim=1] j_d        = np.zeros((3,), dtype=INT)
    cdef int ii, jj, iii, jjj, k, l, u, v, Oi, Oj, ii_min, ii_max, jj_min, jj_max, Xi_min, Xj_min, k_min, l_min, masked
    cdef float X, XX, Y, YY, XY, ncc_w_norm, ncc_w, ncc_w_max, x, y, den

    imap, jmap = np.indices((Od.shape[0], Od.shape[1]))
    imap += X_ij[0] + i
    jmap += X_ij[1] + j

    for ii in range(offset_i, Od.shape[0], steps):
        for jj in range(offset_j, Od.shape[1], steps):
            # get the data segment
            ii_min = max(ii-window//2, 0)
            ii_max = min(ii+window//2, Od.shape[0])
            
            jj_min = max(jj-window//2, 0)
            jj_max = min(jj+window//2, Od.shape[1])
            d_norm = 0.
            X  = 0.
            XX = 0.
            x  = 0.
            for k in range(ii_max - ii_min):
                for l in range(jj_max - jj_min):
                    data[k, l] = mask[k+ii_min, l+jj_min] * Od[k+ii_min, l+jj_min]
                    X    += data[k, l] 
                    XX   += data[k, l] * data[k, l] 
                    if mask[k+ii_min, l+jj_min] != 0 :
                        x += 1.
            
            Xi_min  = 0
            Xj_min  = 0
            ncc_w_max = 0.
            ncc_w_norm = 0.
            k_min = 0
            l_min = 0
            for k in range(search_window):
                for l in range(search_window):
                    Y  = 0.
                    YY = 0.
                    XY = 0.
                    y  = 0.
                    masked = 0
                    for u in range(ii_max-ii_min):
                        for v in range(jj_max-jj_min):
                            # object index = warped index + offset
                            Oi  = imap[u+ii_min, v+jj_min] + k-search_window//2
                            Oj  = jmap[u+ii_min, v+jj_min] + l-search_window//2
                            data_O[u, v] = mask[u+ii_min, v+jj_min] * O[Oi, Oj] 
                            
                            Y     += data_O[u, v] 
                            YY    += data_O[u, v] * data_O[u, v]
                            XY    += data[u, v]   * data_O[u, v]
                            if mask[u + ii_min, v+jj_min] != 0 :
                                y += 1.
                            if mask_O[Oi, Oj] == 0 :
                                masked = 1
                    
                    if masked == 0 :
                        if abs(y - x) > 0.5 :
                            print x, y
                        
                        # calculate the pearson coefficient
                        den = (y*XX - X**2) * (y*YY - Y**2)
                        
                        # check if we are out of range
                        if den < 1.0e-15 :
                            ncc_w   = -1.
                            #print ii, jj, den
                        else :
                            ncc_w   = (y*XY - X*Y) / sqrt(den)
                                
                        ncc_w       = (1. + ncc_w) / 2.
                        ncc_w_norm += ncc_w
                        
                        #if ncc_w <= 0. :
                        #    print ii, jj, x, y, ncc_w
                        
                        if ncc_w > ncc_w_max : #and k != search_window//2 and l != search_window//2 :
                            ncc_w_max = ncc_w 
                            Xi_min  = k - search_window//2
                            Xj_min  = l - search_window//2
                            k_min = k
                            l_min = l
                    
            #print ii, jj, k_min, l_min, Xi_min, Xj_min, ncc_w_norm, ncc_w, X, XX, Y, YY, XY
                    
            if ncc_w_norm > 0. :
                ncc[ii, jj]  = ncc_w_max #/ ncc_w_norm
            else :
                ncc[ii, jj]  = 0.
            
            X_ij_new[0][ii, jj] = Xi_min
            X_ij_new[1][ii, jj] = Xj_min
            
            #print ii, jj, k_min, l_min, ncc[ii, jj], X_ij_new[0][ii, jj], X_ij_new[1][ii, jj]
    return X_ij_new, ncc

