#!/usr/bin/env python

"""
Author: lnazzaro and lgarzio on 3/9/2022
Last modified: lgarzio on 10/13/2025
Calculate and apply optimal time shifts by segment for variables defined in config files (e.g. DO and pH voltages).
Each NetCDF file contains one glider segment and potentially multiple profile pairs.
"""

import os
import argparse
import sys
import glob
import datetime as dt
import numpy as np
import pandas as pd
import xarray as xr
import copy
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import polygonize
import rugliderqc.common as cf
from rugliderqc.loggers import logfile_basename, setup_logger, logfile_deploymentname
np.set_printoptions(suppress=True)


def apply_qc(dataset, varname):
    """
    Make a copy of a data array and convert values with not_evaluated (2) suspect (3) and fail (4) QC flags to nans.
    Convert pH voltage values of 0.0 to nan
    :param dataset: xarray dataset
    :param varname: sensor variable name (e.g. dissolved_oxygen)
    """
    datacopy = dataset[varname].copy()
    for qv in [x for x in dataset.data_vars if f'{varname}_qartod' in x]:
        qv_vals = dataset[qv].values
        qv_idx = np.where(np.logical_or(np.logical_or(qv_vals == 2, qv_vals == 3), qv_vals == 4))[0]
        datacopy[qv_idx] = np.nan
    if 'ph_ref_voltage' in varname:
        zeros = np.where(datacopy == 0.0)[0]
        datacopy[zeros] = np.nan
    return datacopy


def apply_time_shift(df, varname, shift_seconds, merge_original=False):
    """
    Apply a specified time shift to a variable.
    :param df: pandas dataframe containing the variable of interest (varname), depth, and time as the index
    :param varname: sensor variable name (e.g. dissolved_oxygen)
    :param shift_seconds: desired time shift in seconds
    :param merge_original: merge shifted dataframe with the original dataframe, default is False
    :returns: pandas dataframe containing the time-shifted variable, depth, and time as the index
    """
    # split off the variable and profile direction identifiers into a separate dataframe
    try:
        sdf = pd.DataFrame(dict(shifted_var=df[varname],
                                downs=df['downs']))
    except KeyError:
        sdf = pd.DataFrame(dict(shifted_var=df[varname]))

    # calculate the shifted timestamps
    tm_shift = df.index - dt.timedelta(seconds=shift_seconds)

    # append the shifted timestamps to the new dataframe and drop the original time index
    sdf['time_shift'] = tm_shift
    sdf.reset_index(drop=True, inplace=True)

    # rename the new columns and set the shifted timestamps as the index
    sdf = sdf.rename(columns={'time_shift': 'time',
                              'downs': 'downs_shifted'})
    sdf = sdf.set_index('time')

    if merge_original:
        # merge back into the original dataframe and drop rows with nans
        df2 = df.merge(sdf, how='outer', left_index=True, right_index=True)

        # drop the original variable
        df2.drop(columns=[varname, 'downs'], inplace=True)
        df2 = df2.rename(columns={'shifted_var': f'{varname}_shifted',
                                  'downs_shifted': 'downs'})
    else:
        df2 = sdf.rename(columns={'shifted_var': f'{varname}_shifted',
                                  'downs_shifted': 'downs'})

    return df2


def calculate_depth_range(df):
    """
    Calculate depth range for a dataframe
    :param df: pandas dataframe containing depth
    :returns: depth range
    """
    min_depth = np.nanmin(df.depth)
    max_depth = np.nanmax(df.depth)

    return max_depth - min_depth


def depth_bins(df, interval=0.25):
    """
    Bin data according to a specified depth interval, calculate median values for each bin.
    :param df: pandas dataframe containing depth and the time-shifted data, and time as the index
    :param interval: optional depth interval for binning, default is 0.25
    :returns: pandas dataframe containing depth-binned median data
    """
    # specify the bin intervals
    max_depth = np.nanmax(df.depth)
    bins = np.arange(0, max_depth, interval).tolist()
    bins.append(bins[-1] + interval)

    # calculate the bin for each row
    df['bin'] = pd.cut(df['depth'], bins)

    # calculate depth-binned median
    # used median instead of mean to account for potential unreasonable values not removed by QC
    df = df.groupby('bin', observed=False).median()

    return df


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

        # Set the default qc configuration path
        qc_config_root = os.path.join(data_home, 'qc', 'config')
        if not os.path.isdir(qc_config_root):
            logging_base.warning(f'Invalid QC config root: {qc_config_root}')

        #for deployment in args.deployments:
        for deployment in [deployments]:  # for debugging

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

            logging.info('Calculating optimal time shift: {:s}'.format(os.path.join(data_path, 'qc_queue')))

            # Set the deployment qc configuration path
            deployment_qc_config_root = os.path.join(deployment_location, 'config', 'qc')
            if not os.path.isdir(deployment_qc_config_root):
                logging.warning(f'Invalid deployment QC config root: {deployment_qc_config_root}')

            # Determine if the test should be run or not
            qctests_config_file = os.path.join(deployment_qc_config_root, 'qctests.yml')
            if os.path.isfile(qctests_config_file):
                qctests_config_dict = cf.load_yaml_file(qctests_config_file, logger=logging)
                if not qctests_config_dict['time_shift']:
                    logging.warning(f'Not calculating time shifts because test is turned off, check: {qctests_config_file}')
                    continue

            # Get the variable names for time shifting from the config file for the deployment. If not provided,
            # optimal time shifts aren't calculated
            config_file = os.path.join(deployment_qc_config_root, 'time_shift.yml')
            if not os.path.isfile(config_file):
                logging.warning(f'Time shifts not calculated because deployment config file not specified: {config_file}.')
                continue

            shift_dict = cf.load_yaml_file(config_file, logger=logging)

            # keep track of each segment's optimal time shift
            segment_shifts = dict()
            for k, v in shift_dict.items():
                segment_shifts[k] = np.array([])

            # List the netcdf files
            ncfiles = sorted(glob.glob(os.path.join(data_path, 'qc_queue', '*.nc')))

            if len(ncfiles) == 0:
                logging.error(' 0 files found to QC: {:s}'.format(os.path.join(data_path, 'qc_queue')))
                continue

            # define shifts in seconds to test
            seconds = 60
            shifts = np.arange(0, seconds, 1).tolist()
            shifts.append(seconds)

            files_tested = 0

            logging.info('Calculating optimal time shift by segment')

            # Iterate through each file (contains a full segment) and calculate optimal shift for that segment
            for f in ncfiles:
                try:
                    with xr.open_dataset(f, decode_times=False) as ds:
                        ds = ds.load()
                except OSError as e:
                    logging.error(f'Error reading file {f} ({e})')
                    continue

                fname = f.split('/')[-1]

                # Iterate through the test variables
                for testvar in shift_dict:

                    try:
                        ds[testvar]
                    except KeyError:
                        logging.debug(f'{testvar} not found in file {fname})')
                        continue

                    data_idx = np.where(~np.isnan(ds[testvar].values))[0]

                    if len(data_idx) == 0:
                        logging.debug(f'{testvar} values are all nan: {fname})')
                        continue

                    # can't calculate time shift if there are no profiles
                    if np.all(ds.profile_direction.values == 0):
                        logging.debug(f'No profiles indexed for segment {fname}, optimal time shift not calculated')
                        optimal_shift = np.nan
                
                    # can't calculate time shift if there are only ups or downs
                    elif len(np.unique(ds['profile_direction'])) == 1:
                        logging.debug(f'Only ups or downs available for segment {fname}, optimal time shift not calculated')
                        optimal_shift = np.nan

                    else:
                        # make a copy of the data and apply QARTOD QC flags
                        data_copy = apply_qc(ds, testvar)

                        # convert to dataframe with profile direction
                        df = data_copy.to_dataframe().merge(ds.profile_direction.to_dataframe(), on=['time', 'depth'])

                        # Drop lat and lon columns
                        df = df.drop(columns=[col for col in df.columns if 'latitude' in col or 'longitude' in col])
                        df = df.dropna(subset=[testvar])

                        # Drop rows where profile_direction = 0 (hovering or at surface)
                        df = df[df['profile_direction'] != 0]

                        # make a dataframe without QC removed for time shifting after the optimal shift is calculated
                        df_all = ds[testvar].to_dataframe()
                        df_all = df_all.dropna(subset=[testvar])

                        # convert profile_direction to a down identifier (1 = down, 0 = up)
                        df['profile_direction'] = df['profile_direction'].replace(-1, 0)
                        df.rename(columns={'profile_direction': 'downs'}, inplace=True)

                        # convert timestamps
                        df.index = cf.convert_epoch_ts(df.index)
                        df_all.index = cf.convert_epoch_ts(df_all.index)

                        min_time = pd.to_datetime(np.nanmin(df.index)).strftime('%Y-%m-%dT%H:%M:%S')
                        max_time = pd.to_datetime(np.nanmax(df.index)).strftime('%Y-%m-%dT%H:%M:%S')

                        # removes duplicates and syncs the dataframes so they can be merged when shifted
                        df_resample = df.resample('1s').mean()

                        # For each shift, shift the master dataframes by x seconds, bin data by 0.25 dbar,
                        # calculate area between curves
                        areas = []
                        for shift in shifts:
                            kwargs = dict()
                            kwargs['merge_original'] = True
                            df_shift = apply_time_shift(df_resample, testvar, shift, **kwargs)

                            # interpolate depth
                            df_shift['depth'] = df_shift['depth'].interpolate(method='linear', limit_direction='both')
                            df_shift.dropna(subset=[f'{testvar}_shifted'], inplace=True)

                            # find down identifiers that were averaged in the resampling and reset
                            downs = np.array(df_shift['downs'])
                            ind = np.argwhere(downs == 0.5).flatten()
                            downs[ind] = downs[ind - 1]
                            df_shift['downs'] = downs

                            # after shifting and interpolating depth, divide df into down and up profiles
                            downs_df = df_shift[df_shift['downs'] == 1].copy()
                            ups_df = df_shift[df_shift['downs'] == 0].copy()

                            # don't calculate area if a down or up profile group is missing
                            if np.logical_or(len(downs_df) == 0, len(ups_df) == 0):
                                area = np.nan
                            else:
                                # check the depth range
                                downs_depth_range = calculate_depth_range(downs_df)
                                ups_depth_range = calculate_depth_range(ups_df)

                                # don't calculate area if either profile grouping spans <5m
                                if np.logical_or(downs_depth_range < 5, ups_depth_range < 5):
                                    area = np.nan
                                else:
                                    # bin data frames
                                    downs_binned = depth_bins(downs_df)
                                    downs_binned.dropna(inplace=True)
                                    ups_binned = depth_bins(ups_df)
                                    ups_binned.dropna(inplace=True)

                                    downs_ups = pd.concat([downs_binned, ups_binned.iloc[::-1]])  # downs_ups = downs_binned.append(ups_binned.iloc[::-1])

                                    # calculate area between curves
                                    polygon_points = downs_ups.values.tolist()
                                    polygon_points.append(polygon_points[0])
                                    polygon = Polygon(polygon_points)
                                    polygon_lines = polygon.exterior
                                    polygon_crossovers = polygon_lines.intersection(polygon_lines)
                                    polygons = polygonize(polygon_crossovers)
                                    valid_polygons = MultiPolygon(polygons)
                                    area = valid_polygons.area

                            areas.append(area)

                        # if >50% of the values are nan, return nan
                        fraction_nan = np.sum(np.isnan(areas)) / len(areas)
                        if fraction_nan > .5:
                            optimal_shift = np.nan

                            logging.debug(f'Optimal time shift for {testvar} {min_time} to {max_time}: undetermined')
                        else:
                            # find the shift that results in the minimum area between the curves
                            optimal_shift = int(np.nanargmin(areas))

                            # if the optimal shift is zero or last shift tested (couldn't find a minimal
                            # area within the times tested), use the closest non-nan shift from the
                            # previous segments
                            if np.logical_or(optimal_shift == 0, optimal_shift == np.nanmax(seconds)):
                                non_nans = ~np.isnan(segment_shifts[testvar])
                                try:
                                    optimal_shift = int(segment_shifts[testvar][non_nans][-1])
                                except IndexError:
                                    # if there are no previous non-nan optimal shifts, use the default
                                    # value from the config file
                                    optimal_shift = shift_dict[testvar]['default_shift']
                    
                    # keep track of optimal time shifts
                    segment_shifts[testvar] = np.append(segment_shifts[testvar], optimal_shift)
                    
                    # define new variable names for shifted data to add back into the .nc file
                    data_shift_varname = f'{testvar}_shifted'
                    shift_varname = f'{testvar}_optimal_shift'

                    # if there is no optimal shift calculated, the shifted data array is the same as the original
                    # otherwise, apply the time shift to the data
                    if np.isnan(optimal_shift):
                        shifted_data = ds[testvar].values.copy()
                    else:
                        # shift the data in the non-QC'd dataframe by the optimal time shift calculated
                        segment_shifted = apply_time_shift(df_all, testvar, optimal_shift)
                        data_time = cf.convert_epoch_ts(ds[testvar].time)
                        df_file = segment_shifted[(segment_shifted.index >= np.nanmin(data_time)) & (segment_shifted.index <= np.nanmax(data_time))].copy()
                        df_file.dropna(inplace=True)
                        data_df = ds[testvar].to_dataframe()
                        data_df[data_shift_varname] = np.nan

                        # convert timestamps to date times in dataframe
                        data_df.index = data_time

                        # insert the shifted data in the location of the closest timestamp from the original file
                        for name, row in df_file.iterrows():
                            name_idx = np.argmin(abs(data_df.index - name))
                            data_df.loc[data_df.index[name_idx], data_shift_varname] = row[data_shift_varname]

                        # create data array of shifted values
                        shifted_data = np.array(data_df[data_shift_varname])

                    # insert the array of shifted values into the original dataset
                    attrs = ds[testvar].attrs.copy()
                    attrs['long_name'] = shift_dict[testvar]['long_name']
                    comment = '{} shifted by the optimal time shift (seconds) determined by grouping down ' \
                                'and up profiles for one glider segment, then minimizing the areas between the ' \
                                'profiles by testing time shifts between 0 and {} seconds. This is a preliminary ' \
                                'variable currently under development.'.format(testvar, seconds)
                    attrs['comment'] = comment

                    # Create data array of shifted data
                    da = xr.DataArray(shifted_data.astype(ds[testvar].dtype), coords=ds[testvar].coords, 
                                      dims=ds[testvar].dims, name=data_shift_varname, attrs=attrs)

                    # use the encoding from the original variable that was time shifted
                    cf.set_encoding(da, original_encoding=ds[testvar].encoding)

                    # Add the shifted data to the dataset
                    ds[data_shift_varname] = da

                    # create data array of the optimal shift (seconds) and insert in original data file
                    shift_vals = optimal_shift * np.ones(np.shape(ds[testvar].values))

                    comment = 'Optimal time shift (seconds) determined by grouping down and up profiles for one ' \
                                'glider segment, then minimizing the area between the ' \
                                'profiles by testing time shifts between 0 and {} seconds.  This is a preliminary ' \
                                'variable currently under development.'.format(seconds)

                    # set attributes
                    attrs = {
                        'comment': comment,
                        'units': 'sec',
                        'valid_min': 0,
                        'valid_max': seconds - 1,
                        'qc_target': testvar
                    }

                    # Create data array of optimal shift
                    da = xr.DataArray(shift_vals.astype('float32'), coords=ds[testvar].coords, dims=ds[testvar].dims,
                                      name=shift_varname, attrs=attrs)

                    # define variable encoding
                    cf.set_encoding(da)

                    # Add the optimal shift to the original dataset
                    ds[shift_varname] = da

                # update the history attr
                now = dt.datetime.now(dt.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')
                if not hasattr(ds, 'history'):
                    ds.attrs['history'] = f'{now}: {os.path.basename(__file__)}'
                else:
                    ds.attrs['history'] = f'{ds.attrs["history"]} {now}: {os.path.basename(__file__)}'

                ds.to_netcdf(f)
                ds.close()
                files_tested += 1

            logging.info(f'{deployment}: {files_tested} of {len(ncfiles)} files tested.')


if __name__ == '__main__':
    deploy = 'ru44-20250325T0438'  #  ru44-20250306T0038 ru44-20250325T0438 ru39-20250423T1535
    mode = 'rt'  # delayed rt
    ll = 'debug'
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
