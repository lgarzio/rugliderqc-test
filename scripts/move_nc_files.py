#!/usr/bin/env python

"""
Author: lgarzio on 12/7/2021
Last modified: lgarzio on 10/17/2025
Move quality controlled glider NetCDF files to the final data directory (out of qc_queue) to send to ERDDAP
"""

import os
import argparse
import sys
import glob
import subprocess
import time
from rugliderqc.common import find_glider_deployment_datapath, find_glider_deployments_rootdir
from rugliderqc.loggers import logfile_basename, setup_logger, logfile_deploymentname


def main(args):
# def main(deployments, mode, loglevel, test):
    loglevel = args.loglevel.upper()
    mode = args.mode
    test = args.test
    loglevel = loglevel.upper()

    # logFile_base = os.path.join(os.path.expanduser('~'), 'glider_proc_log')  # for debugging
    logFile_base = logfile_basename()
    logging_base = setup_logger('logging_base', loglevel, logFile_base)

    # wait 10 seconds before proceeding
    time.sleep(10)

    data_home, deployments_root = find_glider_deployments_rootdir(logging_base, test)
    if isinstance(deployments_root, str):

        for deployment in args.deployments:
        # for deployment in [deployments]:  # for debugging

            data_path, deployment_location = find_glider_deployment_datapath(logging_base, deployment, deployments_root, mode)

            if not data_path:
                logging_base.error(f'{deployment} data directory not found:')
                continue

            if not os.path.isdir(os.path.join(deployment_location, 'proc-logs')):
                logging_base.error(f'{deployment} deployment proc-logs directory not found:')
                continue

            logfilename = logfile_deploymentname(deployment, mode)
            logFile = os.path.join(deployment_location, 'proc-logs', logfilename)
            logging = setup_logger('logging', loglevel, logFile)

            # List the netcdf files in qc_queue
            ncfiles = sorted(glob.glob(os.path.join(data_path, 'qc_queue', '*.nc')))

            if len(ncfiles) == 0:
                logging.error(' 0 files found to move: {:s}'.format(os.path.join(data_path, 'qc_queue')))
                status = 1
                continue

            # move files to the parent directory
            logging.info(f'Attempting to move {len(ncfiles)} netcdf files')

            subprocess.call("mv " + os.path.join(data_path, 'qc_queue', '*.nc') + " " + data_path, shell=True)
            logging.info(f'Moved {len(ncfiles)} netcdf files to: {data_path}')
            logging.info('End of QC process')



if __name__ == '__main__':
    # deploy = 'ru39-20250423T1535'  #  ru44-20250306T0038 ru44-20250325T0438 ru39-20250423T1535
    # mode = 'delayed'  # delayed rt
    # ll = 'debug'
    # test = True
    # main(deploy, mode, ll, test)
    arg_parser = argparse.ArgumentParser(description=main.__doc__,
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument('deployments',
                            nargs='+',
                            help='Glider deployment name(s) formatted as glider-YYYYmmddTHHMM')

    arg_parser.add_argument('-m', '--mode',
                            help='Dataset mode: real-time (rt) or delayed-mode (delayed)',
                            choices=['rt', 'delayed'],
                            default='rt')

    arg_parser.add_argument('-l', '--loglevel',
                            help='Verbosity level',
                            type=str,
                            choices=['debug', 'info', 'warning', 'error'],
                            default='info')

    arg_parser.add_argument('-test', '--test',
                            help='Point to the environment variable key GLIDER_DATA_HOME_TEST for testing.',
                            action='store_true')

    parsed_args = arg_parser.parse_args()

    sys.exit(main(parsed_args))
