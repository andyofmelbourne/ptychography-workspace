import sys, os

import numpy as np
import h5py
import time
import ConfigParser

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

def parse_cmdline_args():
    import argparse
    import os
    parser = argparse.ArgumentParser(description='template python command')
    parser.add_argument('filename', type=str, \
                        help="file name of the *.h5 file")
    parser.add_argument('-c', '--config', type=str, \
                        help="file name of the configuration file")
    
    args = parser.parse_args()
    
    # check that cxi file exists
    if not os.path.exists(args.filename):
        raise NameError('file does not exist: ' + args.filename)
    
    # if config is non then read the default from the *.h5 dir
    if args.config is None :
        # guess that the ini file has the same name as this python file
        # but with an 'ini' extension
        ini_fnam = os.path.split(__file__)[1][:-2] + 'ini'
        
        # guess that the ini file is in the same directory as the h5 file
        args.config = os.path.join(os.path.split(args.filename)[0], ini_fnam)
        
        # if that does not exist then use the template in this directory
        if not os.path.exists(args.config):
            args.config = __file__[:-2] + 'ini'
    
    # check that args.config exists
    if not os.path.exists(args.config):
        raise NameError('config file does not exist: ' + args.config)
    
    # process config file
    config = ConfigParser.ConfigParser()
    config.read(args.config)
    
    params = parse_parameters(config)
    
    return args, params

def main(args, params):
    """
    I will output a random 2d image:
    
    Returns
    -------
    image : (6,6) float64
    
    and a random 1d array
    error : (100) float64
    """
    # make some random stuff
    image = np.random.random((6,6))
    error = np.random.random((100,))

    # open the input file (just for fun)
    f = h5py.File(args.filename)
    f.close()
    
    # open the output file (could be the same as input file)
    if params['template_command']['output_file'] is None :
        out = args.filename
    else :
        out = params['template_command']['output_file']
    
    f = h5py.File(out)
    
    if params['template_command']['output_group'] not in f.keys():
        g = f.create_group(params['template_command']['output_group'])
    else :
        g = f[params['template_command']['output_group']]
    
    # wack in our results
    if 'image' in g :
        del g['image']
    g['image'] = image
    if 'error' in g :
        del g['error']
    g['error'] = error
    f.close()
    
    # done
    print 'done!'


if __name__ == '__main__':
    args, params = parse_cmdline_args()

    main(args, params)
