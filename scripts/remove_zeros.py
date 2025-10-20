#!/usr/bin/env python

"""
Author: lgarzio on 12/22/2021
Last modified: lgarzio on 9/12/2025
Converts CTD science variables to fill values if conductivity and temperature are both 0.000, 
and dissolved oxygen science variables to fill values if
oxygen_concentration and optode_water_temperature are both 0.000.
"""

import os
import argparse
import sys
import glob
import xarray as xr
import numpy as np
import rugliderqc.common as cf
from rugliderqc.loggers import logfile_basename, setup_logger, logfile_deploymentname


def check_zeros(varname_dict, dataset, ds_modified, var1, var2):
    """
    Find indices where values for 2 variables are 0.0000 and convert all defined variables to fill values
    :param varname_dict: dictionary containing variables to modify if condition is met
    :param dataset: xarray dataset
    :param ds_modified: int value indicating if the dataset was modified
    :param var1: first variable name to test for condition (e.g. 'conductivity' or 'oxygen_concentration')
    :param var2: second variable name to test for condition (e.g. 'temperature' or 'optode_water_temperature')
    returns int value indicating if the dataset was modified
    """
    for key, variables in varname_dict.items():
        try:
            check_var1 = dataset[variables[var1]]
            check_var2 = dataset[variables[var2]]
        except KeyError:
            continue

        var1_zero_idx = np.where(check_var1 == 0.0000)[0]
        if len(var1_zero_idx) > 0:
            var2_zero_idx = np.where(check_var2 == 0.0000)[0]
            dointersect_idx = np.intersect1d(var1_zero_idx, var2_zero_idx)
            if len(dointersect_idx) > 0:
                for cv, varname in variables.items():
                    dataset[varname][dointersect_idx] = dataset[varname].encoding['_FillValue']
                    ds_modified += 1

    return ds_modified


def main(args):
# def main(deployments, mode, loglevel, test):
    loglevel = args.loglevel.upper()
    mode = args.mode
    test = args.test
    loglevel = loglevel.upper()

    # logFile_base = os.path.join(os.path.expanduser('~'), 'glider_proc_log')  # for debugging
    logFile_base = logfile_basename()
    logging_base = setup_logger('logging_base', loglevel, logFile_base)

    data_home, deployments_root = cf.find_glider_deployments_rootdir(logging_base, test)
    if isinstance(deployments_root, str):

        # Set the default qc configuration path
        qc_config_root = os.path.join(data_home, 'qc', 'config')
        if not os.path.isdir(qc_config_root):
            logging_base.warning(f'Invalid QC config root: {qc_config_root}')
            return 1

        for deployment in args.deployments:
        # for deployment in [deployments]:  # for debugging

            # find the location of the files to be processed
            data_path, deployment_location = cf.find_glider_deployment_datapath(logging_base, deployment, deployments_root, mode)

            if not data_path:
                logging_base.error(f'{deployment} data directory not found:')
                continue

            if not os.path.isdir(os.path.join(deployment_location, 'proc-logs')):
                logging_base.error(f'{deployment} deployment proc-logs directory not found:')
                continue

            logfilename = logfile_deploymentname(deployment, mode)
            logFile = os.path.join(deployment_location, 'proc-logs', logfilename)
            logging = setup_logger('logging', loglevel, logFile)

            logging.info('Starting QC process')

            # Set the deployment qc configuration path
            deployment_qc_config_root = os.path.join(deployment_location, 'config', 'qc')
            if not os.path.isdir(deployment_qc_config_root):
                logging.warning(f'Invalid deployment QC config root: {deployment_qc_config_root}')

            # Determine if the test should be run or not
            qctests_config_file = os.path.join(deployment_qc_config_root, 'qctests.yml')
            if os.path.isfile(qctests_config_file):
                qctests_config_dict = cf.load_yaml_file(qctests_config_file, logger=logging)
                if not qctests_config_dict['remove_zeros']:
                    logging.warning(
                        f'Not removing zero fill values because test is turned off, check: {qctests_config_file}'
                        )
                    continue

            logging.info('Checking for TWRC fill values of 0.00: {:s}'.format(os.path.join(data_path, 'qc_queue')))

            # Get all of the possible CTD variable names from the config file
            ctd_config_file = os.path.join(qc_config_root, 'ctd_variables.yml')
            if not os.path.isfile(ctd_config_file):
                logging.error(f'Invalid CTD variable name config file: {ctd_config_file}.')
                continue

            ctd_vars = cf.load_yaml_file(ctd_config_file, logger=logging)

            # Get dissolved oxygen variable names
            oxygen_config_file = os.path.join(qc_config_root, 'oxygen_variables.yml')
            if not os.path.isfile(oxygen_config_file):
                logging.error(f'Invalid DO variable name config file: {oxygen_config_file}.')
                continue

            oxygen_vars = cf.load_yaml_file(oxygen_config_file, logger=logging)

            # List the netcdf files in qc_queue
            ncfiles = sorted(glob.glob(os.path.join(data_path, 'qc_queue', '*.nc')))

            if len(ncfiles) == 0:
                logging.error(' 0 files found to check: {:s}'.format(os.path.join(data_path, 'qc_queue')))
                continue

            # Iterate through files and check for values of 0.0
            zeros_removed = 0
            for f in ncfiles:
                logging.debug(f'{f}')
                modified = 0
                try:
                    with xr.open_dataset(f, decode_times=False) as ds:
                        ds = ds.load()
                except OSError as e:
                    logging.error(f'Error reading file {f} ({e})')
                    os.rename(f, f'{f}.bad')
                    continue
                except ValueError as e:
                    logging.error(f'Error reading file {f} ({e})')
                    os.rename(f, f'{f}.bad')
                    continue

                # Set CTD values to fill values where conductivity and temperature both = 0.00
                # Try all versions of CTD variable names
                modified = check_zeros(ctd_vars, ds, modified, 'conductivity', 'temperature')

                # Set DO values to fill values where oxygen_concentration and oxygen_saturation both = 0.00
                modified = check_zeros(oxygen_vars, ds, modified, 'oxygen_concentration', 'optode_water_temperature')

                # if zeros were removed from the ds, write the updated file and add the information to the log
                if modified > 0:
                    ds.to_netcdf(f)
                    zeros_removed += 1

            logging.info(f'Removed 0.00 values (TWRC fill values) for CTD and/or DO variables in {zeros_removed} files (of {len(ncfiles)} '
                         'total files)')


if __name__ == '__main__':
    # deploy = 'ru39-20250423T1535'  #  ru44-20250306T0038 ru44-20250325T0438 ru39-20250423T1535
    # mode = 'delayed'  # delayed rt
    # ll = 'info'
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
