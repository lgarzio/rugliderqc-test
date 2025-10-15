#!/usr/bin/env python

"""
Author: lgarzio on 1/18/2022
Last modified: lgarzio on 10/8/2025
Summarize the QARTOD QC flags for each variable. Excludes climatology test.
"""

import os
import argparse
import sys
import datetime as dt
import glob
import numpy as np
import xarray as xr
import rugliderqc.common as cf
from rugliderqc.loggers import logfile_basename, setup_logger, logfile_deploymentname
np.set_printoptions(suppress=True)


def set_summary_qartod_attrs(sensor, ancillary_variables):
    """
    Define the QARTOD QC summary flag attributes
    :param sensor: sensor variable name (e.g. conductivity)
    :param ancillary_variables: variables included in the summary flag, format is a string attribute whose values
    are blank separated
    """

    flag_meanings = 'GOOD NOT_EVALUATED SUSPECT FAIL MISSING'
    flag_values = [1, 2, 3, 4, 9]
    standard_name = 'aggregate_quality_flag'
    long_name = 'QARTOD Summary Quality Flag'

    # Define variable attributes
    attrs = {
        'ancillary_variables': ancillary_variables,
        'standard_name': standard_name,
        'long_name': long_name,
        'flag_values': flag_values,
        'flag_meanings': flag_meanings,
        'valid_min': min(flag_values),
        'valid_max': max(flag_values),
        'ioos_qc_module': 'qartod',
        'ioos_qc_target': sensor,
        'comment': f'Highest QARTOD flag value for {sensor}, excluding 2/not evaluated. Excludes climatology test.'
    }

    return attrs


#def main(args):
def main(deployments, mode, loglevel, test):
    # loglevel = args.loglevel.upper()
    # mode = args.mode
    # test = args.test
    loglevel = loglevel.upper()  # for debugging

    logFile_base = os.path.join(os.path.expanduser('~'), 'glider_proc_log')  # for debugging
    # logFile_base = logfile_basename()
    logging_base = setup_logger('logging_base', loglevel, logFile_base)

    data_home, deployments_root = cf.find_glider_deployments_rootdir(logging_base, test)
    if isinstance(deployments_root, str):

        #for deployment in args.deployments:
        for deployment in [deployments]:  # for debugging

            data_path, deployment_location = cf.find_glider_deployment_datapath(logging_base, deployment, deployments_root, mode)

            if not data_path:
                logging_base.error(f'{deployment} data directory not found')
                continue

            if not os.path.isdir(os.path.join(deployment_location, 'proc-logs')):
                logging_base.error(f'{deployment} deployment proc-logs directory not found')
                continue

            logfilename = logfile_deploymentname(deployment, mode)
            logFile = os.path.join(deployment_location, 'proc-logs', logfilename)
            logging = setup_logger('logging', loglevel, logFile)

            # Set the deployment qc configuration path
            deployment_qc_config_root = os.path.join(deployment_location, 'config', 'qc')
            if not os.path.isdir(deployment_qc_config_root):
                logging.warning(f'Invalid deployment QC config root: {deployment_qc_config_root}')

            # Determine if the test should be run or not
            qctests_config_file = os.path.join(deployment_qc_config_root, 'qctests.yml')
            if os.path.isfile(qctests_config_file):
                qctests_config_dict = cf.load_yaml_file(qctests_config_file, logger=logging)
                if not qctests_config_dict['qartod_summary']:
                    logging.warning('Not summarizing QARTOD QC because test is turned off, check: {qctests_config_file}')
                    continue

            logging.info('Summarizing QARTOD flags: {:s}'.format(os.path.join(data_path, 'qc_queue')))

            # List the netcdf files in qc_queue
            ncfiles = sorted(glob.glob(os.path.join(data_path, 'qc_queue', '*.nc')))

            if len(ncfiles) == 0:
                logging.error(' 0 files found to QC: {:s}'.format(os.path.join(data_path, 'qc_queue')))
                continue

            # Iterate through files and summarize the QARTOD flags
            for f in ncfiles:
                try:
                    with xr.open_dataset(f, decode_times=False) as ds:
                        ds = ds.load()
                except OSError as e:
                    logging.error(f'Error reading file {f} ({e})')
                    continue

                # List the qartod flag variables, excluding the climatology test
                qartod_vars = [x for x in ds.data_vars if np.logical_and('_qartod_' in x, '_qartod_climatology' not in x)]

                if len(qartod_vars) == 0:
                    continue

                # List the sensors that were QC'd
                qc_vars = list(np.unique([x.split('_qartod_')[0] for x in qartod_vars]))

                # Iterate through each sensor that was QC'd and summarize the QARTOD flags
                for sensor in qc_vars:
                    summary_flag = np.empty(len(ds[sensor].values))
                    summary_flag[:] = 0
                    sensor_qartod_vars = [x for x in qartod_vars if f'{sensor}_qartod_' in x]
                    for sqv in sensor_qartod_vars:
                        # make a copy of the flags so the original array isn't changed
                        flag = ds[sqv].values.copy()

                        # turn 2/NOT_EVALUATED/UNKNOWN to 0
                        flag[flag == 2] = 0

                        summary_flag = np.maximum(summary_flag, flag)

                    # check if any flags are zero and turn those back to 2 (NOT_EVALUATED/UNKNOWN)
                    summary_flag[summary_flag == 0] = 2

                    qc_varname = f'{sensor}_qartod_summary_flag'
                    attrs = set_summary_qartod_attrs(sensor, ' '.join(sensor_qartod_vars))

                    # Create summary data array
                    da = xr.DataArray(summary_flag.astype('int32'), coords=ds[sensor].coords, dims=ds[sensor].dims,
                                      name=qc_varname, attrs=attrs)

                    # define variable encoding
                    cf.set_encoding(da)

                    # Add summary variable to dataset
                    ds[qc_varname] = da

                    # add the summary variable to the sensor ancillary_variables
                    if not hasattr(ds[sensor], 'ancillary_variables'):
                        ds[sensor].attrs['ancillary_variables'] = qc_varname
                    else:
                        ds[sensor].attrs['ancillary_variables'] = ' '.join((ds[sensor].ancillary_variables, qc_varname))

                # update the history attr
                now = dt.datetime.now(dt.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')
                if not hasattr(ds, 'history'):
                    ds.attrs['history'] = f'{now}: {os.path.basename(__file__)}'
                else:
                    ds.attrs['history'] = f'{ds.attrs["history"]} {now}: {os.path.basename(__file__)}'

                ds.to_netcdf(f)
                ds.close()

            logging.info(f'{deployment}: finished summarizing QARTOD flags')


if __name__ == '__main__':
    deploy = 'ru44-20250325T0438'  #  ru44-20250306T0038 ru44-20250325T0438 ru39-20250423T1535
    mode = 'delayed'  # delayed rt
    ll = 'info'
    test = True
    main(deploy, mode, ll, test)
    # arg_parser = argparse.ArgumentParser(description=main.__doc__,
    #                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # arg_parser.add_argument('deployments',
    #                         nargs='+',
    #                         help='Glider deployment name(s) formatted as glider-YYYYmmddTHHMM')

    # arg_parser.add_argument('-m', '--mode',
    #                         help='Deployment dataset status',
    #                         choices=['rt', 'delayed'],
    #                         default='rt')

    # arg_parser.add_argument('-l', '--loglevel',
    #                         help='Verbosity level',
    #                         type=str,
    #                         choices=['debug', 'info', 'warning', 'error', 'critical'],
    #                         default='info')

    # arg_parser.add_argument('-test', '--test',
    #                         help='Point to the environment variable key GLIDER_DATA_HOME_TEST for testing.',
    #                         action='store_true')

    # parsed_args = arg_parser.parse_args()

    # sys.exit(main(parsed_args))
