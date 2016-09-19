"""
Phase the input file using Ptychography

should submit a batch job:
???
"""
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



if __name__ == '__main__':
    args, params = parse_cmdline_args()

    # read data into the master
    ###########################
    if rank == 0 :
        f = h5py.File(args.filename)
         
        if params['input']['frames'] != 'all' :
            frames_i = np.array([int(i) for i in params['input']['frames'].split(',')])
        else :
            frames_i = np.arange(f['data'].shape[0])
        
        I    = f['data'][frames_i]
        R    = f['R'][frames_i]
        mask = f['mask'][()]
        if 'O' not in f :
            O = None
        else :
            O    = f['O'][()]
        P    = f['P'][()]
        
        defocus = params['input']['defocus']
        if (defocus is not None) and (defocus != 'metadata') :
            lamb = f['metadata/wavelength'][()]
            z    = f['metadata/detector_distance'][()]
            du   = f['metadata/fs_pixel_size'][()]
            dq   = du / (lamb * z)
            Fresnel = 1 / (dq**2 * lamb * defocus)
            dx_fresnel = du * params['input']['defocus'] / z
        else :
            Fresnel = False
        
        print 'Fresnel number : ', Fresnel
         
        # scale Rs if Fresnel is not False
        if Fresnel is not False :
            R[:, 0] *= f['/metadata/R_ss_scale'].value
            R[:, 1] *= f['/metadata/R_fs_scale'].value
            R = R / dx # this is the pixel unit shifts
            R = np.rint(R).astype(np.int)
    
    comm.Barrier()
