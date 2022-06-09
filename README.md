CDR_AMSR2
---

This code package is in development.

The code here creates sea ice concentration estimates for the NOAA CDR using
code adapted from Goddard's Bootstrap and NASA Team algorithms.


# Directory contents:

## `./cdr_amsr2/bt/`

Here are python routines which replace Goddard's original Fortran code
for production sea ice concentration.

Initially, the code is simply a translation of the original Fortran code,
and is intended to reproduce those results as similarly as possible.

As it develops, this code will become more and more general

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

Note that this will create .json files that the python code will read

Output that will be compared to an original file in cdr_testdata/

    ./legacy/SB2_NRT_programs/NH_20180217_SB2_NRT_f18.ic


Additionally, json files are created. Symlink the json files to the bt
directory:

```
cd ../../cdr_amsr2/bt
ln -sfn ../../legacy/SB2_NRT_programs/*.json .
```

Now generate the initial Python output


```
cd ../../
./scripts/gen_sample_nh_ic_py.sh
```

Output that will be compared to the output in the fortran directory:

    ./cdr_amsr2/bt/NH_20180217_SB2_NRT_f18.ic

# Other routines

Two short comparison scripts in cdr_amsr2/bt/ used to compare 4-byte float and
2-byte int raw binary files respectively are: fpcomp.py and i2comp.py


# Misc. Development notes

This project uses `invoke` as a task runner. To see all of the available tasks:

```
$ invoke -l
Available tasks:

  format.format (format)    Apply formatting standards to the codebase.
  test.lint (test.flake8)   Run flake8 linting.
```
