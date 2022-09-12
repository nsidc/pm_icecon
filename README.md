CDR_AMSR2
---

This code package is in development and the API is subject to change without
notice. There is no guarantee that this code works as expected. The addition of
tests and verification of outputs is still in progress.

The code here creates sea ice concentration estimates for the NOAA CDR using
code adapted from Goddard's Bootstrap and NASA Team algorithms.


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

## Running the tests

All of the tests (Unit, lint, typechecker, regression) can be run:

```
$ inv test
```

If a linting error occurs, the `format` task might fix the issue via the `black`
formatter:

```
inv format
```

# Running the python code

## CLI

There is a command line interface defined using common defaults for testing
purposes at NSIDC.

NOTE: the CLI relies on hard-coded paths to mask files on NSIDC's virtual
machine infrastructure. This CLI will not currently work for those outside of
NSIDC. We plan to change this in the future.

The CLI can be interacted with via `scripts/cli.sh`:

```
$ ./scripts/cli.sh --help
Usage: python -m cdr_amsr2.cli [OPTIONS] COMMAND [ARGS]...

  Run the nasateam or bootstrap algorithm.

Options:
  --help  Show this message and exit.

Commands:
  bootstrap  Run the bootstrap algorithm.
  nasateam   Run the nasateam algorithm.
```

E.g., to create a NetCDF file with a `conc` variable containing concentration
values from AU_SI12 from the bootstrap algorithm:

```
$ ./scripts/cli.sh bootstrap amsr2 --date 2022-08-01 --hemisphere north --output-dir /tmp/ --resolution 12
2022-09-12 15:21:44.482 | INFO     | cdr_amsr2.bt.cli:amsr2:82 - Wrote AMSR2 concentration field: /tmp/bt_NH_20220801_u2_12km.nc
```

E.g., to create a NetCDF file with a `conc` variable containing concentration
values from AU_SI25 from the nasateam algorithm:

```
$ ./scripts/cli.sh nasateam amsr2 --date 2022-08-01 --hemisphere south --output-dir /tmp/ --resolution 25
2022-09-12 15:23:34.993 | INFO     | cdr_amsr2.nt.cli:amsr2:82 - Wrote AMSR2 concentration field: /tmp/nt_SH_20220801_u2_25km.nc
```

## Scripting
### Bootstrap


Users can write a script using the functions provided in this repo to run the
bootstrap algorithm. The main entrypoint to the bootstrap algorithm is the
`bootstrap` function defined in `cdr_amsr2/bt/compute_bt_ic.py`.

For an example of how to write a script to convert a2l1c tbs into a
concentration field, see `scripts/example_bt_script.py`.

Additional examples are in `cdr_amsr2/bt/api.py`. Note that as of this time, all
functions defined in the the `api` module are specifically setup to use
hard-coded defaults for testing purposes at NSIDC. This includes paths to data
on NSIDC infrastructure that are not available to the public.


### Nasateam

The main entrypoint to running the nasateam code on input Tbs is the `nasateam`
function defined in `cdr_amsr2/nt/compute_nt_ic.py`.

An API has also been defined for common use cases. See `cdr_amsr/nt/api.py` for
more information. NOTE: the API is currently defined with hard-coded defaults
that expect access to NSIDC's virtual machine infrastructure and should be used
with caution.
