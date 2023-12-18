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

### Bootstrap


Users can write a script using the functions provided in this repo to run the
bootstrap algorithm. The main entrypoint to the bootstrap algorithm is the
`bootstrap` function defined in `pm_icecon/bt/compute_bt_ic.py`.

For an example of how to write a script to convert a2l1c tbs into a
concentration field, see `scripts/example_bt_script.py`.

Additional examples are in `pm_icecon/bt/api.py`. Note that as of this time, all
functions defined in the the `api` module are specifically setup to use
hard-coded defaults for testing purposes at NSIDC. This includes paths to data
on NSIDC infrastructure that are not available to the public.


### Nasateam

The main entrypoint to running the nasateam code on input Tbs is the `nasateam`
function defined in `pm_icecon/nt/compute_nt_ic.py`.

An API has also been defined for common use cases. See `pm_icecon/nt/api.py` for
more information. NOTE: the API is currently defined with hard-coded defaults
that expect access to NSIDC's virtual machine infrastructure and should be used
with caution.


## Development/contributing

See [doc/development.md](doc/development.md) for more information.


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

For more information on the NASA Team and Bootstrap algorithms, please consult
the following references, respectively:

Cavalieri, D. J., Gloersen, P., and Campbell, W. J. (1984), Determination of sea ice parameters with the NIMBUS 7 SMMR, J. Geophys. Res., 89( D4), 5355– 5369, doi:10.1029/JD089iD04p05355. 

Comiso, J. C. 1995. SSM/I Concentrations Using the Bootstrap Algorithm. NASA Reference Publication 1380. 40 pg. Available from: https://www.geobotany.uaf.edu/library/pubs/ComisoJC1995_nasa_1380_53.pdf
