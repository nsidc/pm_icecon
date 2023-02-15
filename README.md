![NSIDC logo](/images/NSIDC_logo_2018_poster-1.png)
![NOAA logo](/images/noaa_at_nsidc.jpg)
![NASA logo](/images/nasa_color-1.gif)

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


## Level of Support

This repository is not actively supported by NSIDC but we welcome issue
submissions and pull requests in order to foster community contribution.

See the [LICENSE](LICENSE) for details on permissions and warranties. Please
contact nsidc@nsidc.org for more information.

## Requirements

This code relies on the python packages defined in the included
`environment.yml` file.

Use [conda](https://docs.conda.io/en/latest/) to install the requirements:

```
$ conda env create
```

## Installation

First, Activate the conda environment:

```
$ conda activate pm_icecon
```

## Usage

### CLI

There is a command line interface defined using common defaults for testing
purposes at NSIDC.

NOTE: the CLI relies on hard-coded paths to mask files on NSIDC's virtual
machine infrastructure. This CLI will not currently work for those outside of
NSIDC. We plan to change this in the future.

The CLI can be interacted with via `scripts/cli.sh`:

```
$ ./scripts/cli.sh --help
Usage: python -m pm_icecon.cli.entrypoint [OPTIONS] COMMAND [ARGS]...

  Run the nasateam or bootstrap algorithm.

Options:
  --help  Show this message and exit.

Commands:
  bootstrap  Run the bootstrap algorithm.
  cdr        Run the CDR algorithm with AMSR2 data.
  nasateam   Run the nasateam algorithm.
```

E.g., to create a NetCDF file with a `conc` variable containing concentration
values from AU_SI12 from the bootstrap algorithm:

```
$ ./scripts/cli.sh bootstrap amsr2 --date 2022-08-01 --hemisphere north --output-dir /tmp/ --resolution 12
2022-09-12 15:21:44.482 | INFO     | pm_icecon.bt.cli:amsr2:82 - Wrote AMSR2 concentration field: /tmp/bt_NH_20220801_u2_12km.nc
```

E.g., to create a NetCDF file with a `conc` variable containing concentration
values from AU_SI25 from the nasateam algorithm:

```
$ ./scripts/cli.sh nasateam amsr2 --date 2022-08-01 --hemisphere south --output-dir /tmp/ --resolution 25
2022-09-12 15:23:34.993 | INFO     | pm_icecon.nt.cli:amsr2:82 - Wrote AMSR2 concentration field: /tmp/nt_SH_20220801_u2_25km.nc
```

### Scripting
#### Bootstrap


Users can write a script using the functions provided in this repo to run the
bootstrap algorithm. The main entrypoint to the bootstrap algorithm is the
`bootstrap` function defined in `pm_icecon/bt/compute_bt_ic.py`.

For an example of how to write a script to convert a2l1c tbs into a
concentration field, see `scripts/example_bt_script.py`.

Additional examples are in `pm_icecon/bt/api.py`. Note that as of this time, all
functions defined in the the `api` module are specifically setup to use
hard-coded defaults for testing purposes at NSIDC. This includes paths to data
on NSIDC infrastructure that are not available to the public.


#### Nasateam

The main entrypoint to running the nasateam code on input Tbs is the `nasateam`
function defined in `pm_icecon/nt/compute_nt_ic.py`.

An API has also been defined for common use cases. See `cdr_amsr/nt/api.py` for
more information. NOTE: the API is currently defined with hard-coded defaults
that expect access to NSIDC's virtual machine infrastructure and should be used
with caution.


#### CDR

The `pm_icecon/cdr.py` module provides code related to creating a CDR-like
concentration field from the outputs of bootstrap and nasateam algorithms.

The `amsr2_cdr` function is the primary entrypoint that can be used to generate
concentrations programatically:

```
import datetime as dt

from pm_icecon.cdr import amsr2_cdr


nh_cdr_20210101 = amsr2_cdr(
    date=dt.date(2021, 1, 1),
    hemisphere='north',
    resolution='12',
)
```

NOTE: the CDR code is currently defined with hard-coded defaults that expect
access to NSIDC's virtual machine infrastructure.

### Misc. Development notes

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


## Making a new release

To make a new release:

1. Make a feature branch and commit your changes.
2. Update the CHANGELOG.md and bump the version (`bumpversion {major|minor|patch}`)
3. Open a PR and have it reviewed.
4. If tests pass and the PR has been approved, merge with `main`.
5. Tag the last commit in `main` with the version you bumped to. This will
   automatically trigger a circleCI build that includes deployment of the
   pacakge to anaconda.org.


## License

See [LICENSE](LICENSE).

## Code of Conduct

See [Code of Conduct](CODE_OF_CONDUCT.md).

## Credit

This software was developed by the National Snow and Ice Data Center with
funding from NASA and NOAA.

The original Bootstrap and NASA Team algorithms were developed by researchers at
the NASA Goddard Space Flight Center. NASA Goddard provided the original code to
NSIDC where it has been adapted and modernized for processing at NSIDC.

For more information, please consult the following references:

Cavalieri, D. J., Gloersen, P., and Campbell, W. J. (1984), Determination of sea ice parameters with the NIMBUS 7 SMMR, J. Geophys. Res., 89( D4), 5355– 5369, doi:10.1029/JD089iD04p05355. 

Comiso, J. C. 1995. SSM/I Concentrations Using the Bootstrap Algorithm. NASA Reference Publication 1380. 40 pg. Available from: https://www.geobotany.uaf.edu/library/pubs/ComisoJC1995_nasa_1380_53.pdf
