"""
Phase the input file using Ptychography

should submit a batch job:
???
"""
import h5py
import time
import numpy as np

import Ptychography.ptychography.era as era
from Ptychography import DM
from Ptychography import ERA
from Ptychography import utils
from Ptychography import write_cxi

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def parse_cmdline_args():
    import argparse
    import os
    import ConfigParser
    parser = argparse.ArgumentParser(description='phase the "filename" data using ptychography')
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

def config_iters_to_alg_num(string):
    import re
    # split a string like '100ERA 200DM 50ERA' with the numbers
    steps = re.split('(\d+)', string)   # ['', '100', 'ERA ', '200', 'DM ', '50', 'ERA']
    
    # get rid of empty strings
    steps = [s for s in steps if len(s)>0] # ['100', 'ERA ', '200', 'DM ', '50', 'ERA']
    
    # pair alg and iters
    # [['ERA', 100], ['DM', 200], ['ERA', 50]]
    alg_iters = [ [steps[i+1].strip(), int(steps[i])] for i in range(0, len(steps), 2)]
    return alg_iters

if __name__ == '__main__':
    args, params = parse_cmdline_args()

    # read data into the master
    ###########################
    if rank == 0 :
        f = h5py.File(args.filename)
        
        # get the frames to process
        good_frames = list(f['good_frames'][()])
        
        I    = f['data'][good_frames]
        R    = f['R'][good_frames]
        mask = f['mask'][()]
        O    = f['O'][()]
        P    = f['P'][()]
        
        # get the Fresnel number 
        ########################
        if params['phase']['fresnel'] : 
            defocus = params['phase']['defocus']
            lamb = f['metadata/wavelength'][()]
            z    = f['metadata/detector_distance'][()]
            du   = f['metadata/fs_pixel_size'][()]
            dq   = du / (lamb * z)
            Fresnel = 1 / (dq**2 * lamb * defocus)
        else :
            Fresnel = False
        f.close()
        
        print 'Fresnel number : ', Fresnel
        print 'sum O : ', np.sum(O)
        
        Oout = O.copy()
        Pout = P.copy()
    else :
        Fresnel = O = Oout = Pout = I = R = P = mask = None
        
    comm.Barrier()
    
    alg_iters = config_iters_to_alg_num(params['phase']['iters'])
    
    d0 = time.time()

    in_params = params['phase']
    
    eMod = []
    for alg, iters in alg_iters:
        if alg == 'ERA' :
            Oout, Pout, info =  ERA(I, R, Pout, Oout, iters, OP_iters = in_params['op_iters'], \
                          mask = mask, Fresnel = Fresnel, background = None, method = in_params['method'], Pmod_probe = in_params['pmod_probe'] , \
                          probe_centering = in_params['probe_centering'], hardware = 'cpu', \
                          alpha = in_params['alpha'], dtype = in_params['dtype'], full_output = True, verbose = False, \
                          sample_blur = in_params['sample_blur'], output_h5file = args.filename, output_h5group = params['phase']['h5_group'], output_h5interval = 1)

            if rank == 0 : eMod += info['eMod']

        if alg == 'DM' :
            Oout, Pout, info =  DM(I, R, Pout, Oout, iters, OP_iters = in_params['op_iters'], \
                          mask = mask, Fresnel = Fresnel, background = None, method = in_params['method'], Pmod_probe = in_params['pmod_probe'] , \
                          probe_centering = in_params['probe_centering'], hardware = 'cpu', \
                          alpha = in_params['alpha'], dtype = in_params['dtype'], full_output = True, verbose = False, \
                          sample_blur = in_params['sample_blur'], output_h5file = args.filename, output_h5group = params['phase']['h5_group'], output_h5interval = 1)

            if rank == 0 : eMod += info['eMod']
    
    d1 = time.time()
    
    # Output
    ############
    if rank == 0 :
        print '\ntime:', d1-d0

        # output O P and eMod into the input file
        #########################################
        f = h5py.File(args.filename)
        # O
        ###
        key = params['phase']['h5_group']+'/O'
        if key in f :
            del f[key]
        f[key] = Oout

        # P
        ###
        key = params['phase']['h5_group']+'/P'
        if key in f :
            del f[key]
        f[key] = Pout

        # eMod
        ######
        key = params['phase']['h5_group']+'/eMod'
        if key in f :
            del f[key]
        f[key] = np.array(eMod)
        f.close()

        write_cxi(I, info['I'], P, Pout, O, Oout, \
                  R, R, None, None, mask, eMod, fnam = params['phase']['fnam'], compress = True)
