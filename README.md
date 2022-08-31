NSIDC Sea Ice Concentrations from Passive Microwave Data
---

This code package is in development and the API is subject to change without
notice. There is no guarantee that this code works as expected. The addition of
tests and verification of outputs is still in progress.

The code here creates sea ice concentration estimates from passive microwave
data using code adapted from NASA Goddard's Bootstrap and NASA Team algorithms.

For more informaton about Bootstrap and NASA Team algorithms, see [Descriptions
of and Differences between the NASA Team and Bootstrap
Algorithms](https://nsidc.org/data/user-resources/help-center/descriptions-and-differences-between-nasa-team-and-bootstrap)


# Directory contents:

## `./cdr_amsr2/bt/`

Here are python routines which replace Goddard's original Fortran code
for production sea ice concentration.

## `./cdr_amsr2/nt/`

Python code related to the nasateam algorithm.


## `./legacy/SB2_NRT_programs/`

This directory contains modifications of the original Bootstrap Fortran code
and should yield exactly the same results as Goddard produces, given identical
input and proper (hard-coded?) local file names.

## `./legacy/nt_orig`

Contains original nasateam code.

## `./cdr_testdata/`  <-- symbolic link

This symbolic link points to a local file system which contains large files
used with the code, either as ancillary input or to verify proper operation
of the code.

Creating this symbolic link allows codes to "hard-code" the relative directory
name without forcing that directory to be maintained in version control.

E.g,:

```
$ ln -s /share/apps/amsr2-cdr/cdr_testdata /path/to/cdr_amsr2/repo/
```


# Setting up for initial run:

First, create and activate the `cdr_amsr2` conda environment:

    conda env create
    conda activate cdr_amsr2

Create symbolic links for original ancillary, input, and output directories.

In ./legacy/SB2_NRT_programs/, create:

    ln -sfn ../../cdr_testdata/bt_goddard_ANCILLARY/ ANCILLARY

    ln -sfn ../../cdr_testdata/bt_goddard_orig_input_tbs/ orig_input_tbs

    ln -sfn ../../cdr_testdata/bt_goddard_orig_output/ orig_output

Generate the fortran output:

in ./legacy/SB2_NRT_programs/, execute:

    ./gen_sample_nh_ic_for.sh


# Misc. Development notes

This project uses `invoke` as a task runner. To see all of the available tasks:

```
$ invoke -l
Available tasks:

  format.format (format)       Apply formatting standards to the codebase.
  test.all (test)              Run all of the tests.
  test.ci                      Run tests in CircleCI.
  test.lint (test.flake8)      Run flake8 linting.
  test.regression              Run regression tests.
  test.typecheck (test.mypy)   Run mypy typechecking.
  test.unit                    Run unit tests.
  test.vulture                 Use `vulture` to detect dead code.
```


# Running the python code

## Bootstrap

### Scripting

Users can write a script using the functions provided in this repo to run the
bootstrap algorithm. The main entrypoint to the bootstrap algorithm is the
`bootstrap` function defined in `cdr_amsr2/bt/compute_bt_ic.py`.

For an example of how to write a script to convert a2l1c tbs into a
concentration field, see `scripts/example_bt_script.py`.

Additional examples are in `cdr_amsr2/bt/api.py`. Note that as of this time, all
functions defined in the the `api` module are specifically setup to use
hard-coded defaults for testing purposes at NSIDC. This includes paths to data
on NSIDC infrastructure that are not available to the public.


### CLI

There is a command line interface defined for the Bootstrap algoirthm using
common defaults for testing purposes at NSIDC.

NOTE: the CLI relies on hard-coded paths to mask files on NSIDC's virtual
machine infrastructure. This CLI will not currently work for those outside of
NSIDC. We plan to change this in the future.

The CLI can be interacted with via `scripts/cli.sh`:

```
$ ./scripts/cli.sh --help
Usage: python -m cdr_amsr2.bt.cli [OPTIONS] COMMAND [ARGS]...

  Run the bootstrap algorithm.

Options:
  --help  Show this message and exit.

Commands:
  a2l1c  Run the bootstrap algorithm with 'a2l1c' data.
  amsr2  Run the bootstrap algorithm with ASMR2 data.
```

E.g., to create a NetCDF file with a `conc` variable containing concentration
values from AU_SI12 data:

```
$ ./scripts/cli.sh amsr2 --date 2022-08-01 --hemisphere north --output-dir /tmp/ --resolution 12
2022-08-29 13:34:49.344 | INFO     | __main__:amsr2:78 - Wrote AMSR2 concentration field: /tmp/bt_NH_20220801_u2_12km.nc
```

## Nasateam

This is a work in progress. More details to come...
