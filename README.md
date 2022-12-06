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


## Level of Support (TODO)

_(Choose one of the following bullets to describe USO Level of Support, then
delete this instructional message along with the unchosen support bullet)_

* This repository is fully supported by NSIDC. If you discover any problems or
  bugs, please submit an Issue. If you would like to contribute to this
  repository, you may fork the repository and submit a pull request.
* This repository is not actively supported by NSIDC but we welcome issue
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

## License

See [LICENSE](LICENSE).

## Code of Conduct

See [Code of Conduct](CODE_OF_CONDUCT.md).

## Credit (TODO: credit Goddard. Specific language for NOAA?)

This software was developed by the National Snow and Ice Data Center with
funding from multiple sources.
