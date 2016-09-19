"""
Phase the input file using Ptychography

should submit a batch job:
???
"""
import h5py
import time

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
        if params['input']['fresnel'] : 
            defocus = params['input']['defocus']
            lamb = f['metadata/wavelength'][()]
            z    = f['metadata/detector_distance'][()]
            du   = f['metadata/fs_pixel_size'][()]
            dq   = du / (lamb * z)
            Fresnel = 1 / (dq**2 * lamb * defocus)
        else :
            Fresnel = False
        
        print 'Fresnel number : ', Fresnel
    else :
        Fresnel = O = I = R = P = mask = None
        
    comm.Barrier()
    
    alg_iters = config_iters_to_alg_num(params['ptychography']['iters'])
    
    d0 = time.time()

    in_params = params['ptychography']
    
    eMod = []
    for alg, iters in alg_iters:
        if alg == 'ERA' :
            Oout, Pout, info =  ERA(I, R, P, O, iters, OP_iters = in_params['op_iters'], \
                          mask = mask, Fresnel = Fresnel, background = None, method = in_params['method'], Pmod_probe = in_params['pmod_probe'] , \
                          probe_centering = in_params['probe_centering'], hardware = 'cpu', \
                          alpha = in_params['alpha'], dtype = in_params['dtype'], full_output = True, verbose = False, \
                          sample_blur = in_params['sample_blur'])

            if rank == 0 : eMod += info['eMod']
    
    d1 = time.time()
    
    # Output
    ############
    if rank == 0 :
        print '\ntime:', d1-d0
        write_cxi(I, info['I'], P, Pout, O, Oout, \
                  R, R, None, None, mask, eMod, fnam = params['output']['fnam'], compress = True)
