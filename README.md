# rugliderqc-test
A collection of python tools to quality control real-time and delayed-mode profile-based glider NetCDF files. It uses a modified version of the [ioos_qc](https://ioos.github.io/ioos_qc/) package to add quality flag variables to the datasets that are outlined in the Required and Strongly Recommended tests in the [IOOS QC Manual of Glider Data](https://cdn.ioos.noaa.gov/media/2017/12/Manual-for-QC-of-Glider-Data_05_09_16.pdf). We have also developed additional tests to further quality control glider data, such as a [test](https://github.com/rucool/rugliderqc/blob/master/scripts/ctd_hysteresis_test.py) that flags CTD profile pairs that are severely lagged, which can be an indication of CTD pump issues.

This code is designed to run on glider NetCDF files with indexed profiles with a specific file and directory structure.

## Note: this repository is under development

## Installation

`git clone https://github.com/lgarzio/rugliderqc-test.git`

Also clone my forked version of ioos_qc
`git clone https://github.com/lgarzio/ioos_qc.git`

Navigate to the cloned repo on your local machine
`cd rugliderqc-test`

Create the environment
`conda env create -f environment.yml`

Activate the environment
`conda activate rugliderqc-test`

Install the local package in your environment
`pip install .`

Navigate to the forked version of ioos_qc
`cd ioos_qc`

Install the forked version of ioos_qc in the environment in editable mode (if you want to make edits)
`pip install -e .`

## Usage

`python run_glider_qc.py glider-YYYYmmddTHHMM`

This wrapper script runs:

1. [remove_zeros.py](https://github.com/lgarzio/rugliderqc-test/blob/master/scripts/remove_zeros.py)
2. [glider_qartod_qc.py](https://github.com/lgarzio/rugliderqc-test/blob/master/scripts/glider_qartod_qc.py)
3. [ctd_hysteresis_test.py](https://github.com/lgarzio/rugliderqc-test/blob/master/scripts/ctd_hysteresis_test.py)
4. [summarize_qartod_flags.py](https://github.com/lgarzio/rugliderqc-test/blob/master/scripts/summarize_qartod_flags.py)
5. [time_shift.py](https://github.com/lgarzio/rugliderqc-test/blob/master/scripts/time_shift.py)
6. [add_derived_variables.py](https://github.com/lgarzio/rugliderqc-test/blob/master/scripts/add_derived_variables.py)
7. [move_nc_files.py](https://github.com/lgarzio/rugliderqc-test/blob/master/scripts/move_nc_files.py)

## Acknowledgements

Development was supported in part by the [Mid-Atlantic Regional Association Coastal Ocean Observing System](https://maracoos.org/).
