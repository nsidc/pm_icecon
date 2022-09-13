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
$ conda activate cdr_amsr2
```

## Usage

### Bootstrap

#### Scripting

Users can write a script using the functions provided in this repo to run the
bootstrap algorithm. The main entrypoint to the bootstrap algorithm is the
`bootstrap` function defined in `cdr_amsr2/bt/compute_bt_ic.py`.

For an example of how to write a script to convert a2l1c tbs into a
concentration field, see `scripts/example_bt_script.py`.

Additional examples are in `cdr_amsr2/bt/api.py`. Note that as of this time, all
functions defined in the the `api` module are specifically setup to use
hard-coded defaults for testing purposes at NSIDC. This includes paths to data
on NSIDC infrastructure that are not available to the public.


#### CLI

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

### Nasateam

This is a work in progress. More details to come...


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
