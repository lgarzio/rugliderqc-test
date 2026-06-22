#!/usr/bin/env python

"""
Author: lgarzio on 5/1/2024
Last modified: lgarzio on 6/22/2026
Add a manual comment to a deployment and the option to convert values to nan using the manual_flag.yml config file
located in $HOME/deployments/YYYY/glider-YYYYMMDDTHHMM/config/qc
"""

import os
import argparse
import sys
import glob
import datetime as dt
import xarray as xr
import pandas as pd
import numpy as np
import rugliderqc.common as cf
from rugliderqc.loggers import logfile_basename, setup_logger, logfile_deploymentname
pd.set_option('display.width', 320, "display.max_columns", 10)


def main(args):
    loglevel = args.loglevel.upper()
    mode = args.mode
    test = args.test

    logFile_base = logfile_basename()
    logging_base = setup_logger('logging_base', loglevel, logFile_base)

    data_home, deployments_root = cf.find_glider_deployments_rootdir(logging_base, test)
    if isinstance(deployments_root, str):

        # Set the default qc configuration path
        qc_config_root = os.path.join(data_home, 'qc', 'config')
        if not os.path.isdir(qc_config_root):
            logging_base.warning(f'Invalid QC config root: {qc_config_root}')

        for deployment in args.deployments:

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

            # Set the deployment qc configuration path
            deployment_qc_config_root = os.path.join(deployment_location, 'config', 'qc')
            if not os.path.isdir(deployment_qc_config_root):
                logging.warning(f'Invalid deployment QC config root: {deployment_qc_config_root}')

            # Get the manual flag information from the deployment-specific manual_flag.yml file
            # If the file doesn't exist, no manual flags are added
            manflag_config_file = os.path.join(deployment_qc_config_root, 'manual_flag.yml')
            if not os.path.isfile(manflag_config_file):
                logging.error(f'manual_flag.yml file does not exist for this deployment: {manflag_config_file}.')
                continue

            manflags = cf.load_yaml_file(manflag_config_file)

            # List the netcdf files
            ncfiles = sorted(glob.glob(os.path.join(data_path, '*.nc')))

            if len(ncfiles) == 0:
                logging.error(f'0 files found: {data_path}')
                status = 1
                continue

            logging.info('Adding manual QC flags')

            # Iterate through files, and add manual QC flags
            for f in ncfiles:
                now = dt.datetime.now(dt.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')
                file_modified = 0
                try:
                    with xr.open_dataset(f) as ds:
                        ds = ds.load()
                except OSError as e:
                    logging.error(f'Error reading file {f} ({e})')
                    os.rename(f, f'{f}.bad')
                    continue
                except ValueError as e:
                    logging.error(f'Error reading file {f} ({e})')
                    os.rename(f, f'{f}.bad')
                    continue

                for k, v in manflags.items():
                    if k == 'deployment_flag':
                        # if there is a deployment flag, add the comment to global attrs (comment and processing_level)
                        if not hasattr(ds, 'comment'):
                            ds.attrs['comment'] = f'{now}: {v["comment"]}'
                        else:
                            ds.attrs['comment'] = f'{ds.attrs["comment"]} {now}: {v["comment"]}'

                        if not hasattr(ds, 'processing_level'):
                            ds.attrs['processing_level'] = f'{now}: {v["comment"]}'
                        else:
                            ds.attrs['processing_level'] = f'{ds.attrs["processing_level"]} {now}: {v["comment"]}'

                        file_modified += 1

                    elif k == 'sensor_flag':
                        for kk, vv in v.items():
                            try:
                                ds[kk]
                                if vv['convert_to_nan']:
                                    # convert the data to NaNs between the timestamps specified in the config file
                                    start_check = ds.time.values >= pd.to_datetime(vv['start'])
                                    end_check = ds.time.values <= pd.to_datetime(vv['end'])
                                    time_check = np.logical_and(start_check, end_check)
                                    ds[kk].values[time_check] = np.nan

                                    file_modified += 1

                                # add to the variable attrs comment
                                if not hasattr(ds[kk], 'comment'):
                                    ds[kk].attrs['comment'] = f'{now}: {vv["comment"]}'
                                else:
                                    ds[kk].attrs['comment'] = f'{ds[kk].attrs["comment"]} {now}: {vv["comment"]}'

                                file_modified += 1

                            except KeyError:
                                continue

                # if the file was modified, add the script to the file history, change modified date and save the file
                if file_modified > 0:
                    if not hasattr(ds, 'history'):
                        ds.attrs['history'] = f'{now}: {os.path.basename(__file__)}'
                    else:
                        ds.attrs['history'] = f'{ds.attrs["history"]} {now}: {os.path.basename(__file__)}'

                    ds.attrs['date_modified'] = now

                    ds.to_netcdf(f)

            logging.info('Finished adding manual QC flags')

        return status


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description=main.__doc__,
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument('deployments',
                            nargs='+',
                            help='Glider deployment name(s) formatted as glider-YYYYmmddTHHMM')

    arg_parser.add_argument('-m', '--mode',
                            help='Deployment dataset status',
                            choices=['rt', 'delayed'],
                            default='delayed')

    arg_parser.add_argument('-l', '--loglevel',
                            help='Verbosity level',
                            type=str,
                            choices=['debug', 'info', 'warning', 'error', 'critical'],
                            default='info')

    arg_parser.add_argument('-test', '--test',
                            help='Point to the environment variable key GLIDER_DATA_HOME_TEST for testing.',
                            action='store_true')

    parsed_args = arg_parser.parse_args()

    sys.exit(main(parsed_args))
