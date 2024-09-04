# v0.3.2

* Update `pm_tb_data` dependency to v0.4.0


# v0.3.1

* Fix some nasateam tiepoints to reflect values found in `cdralgos`.


# v0.3.0

* Extract CDR-specific logic from this repository. Moved to the `seaice_ecdr`
  (https://github.com/nsidc/seaice_ecdr/)
* Remove `Hemisphere` type. Import from `pm_tb_data` instead.
* Remove CLI for this library. It is now the responsiblity of programs utlizing
  this library to define how inputs/outputs are handled. This library purely
  focuses on the implementation of the sea ice concentration algorithms.
* Related to the above, remove code related to reading/handling ancillary files
  that define e.g., valid ice masks. This code was very specific to NSIDC's
  internal infrastructure and unpublished data. It is now the responsibility of
  other programs utlizing this library to provide masks, input TBs, etc.
* Replace `BootstrapAlgError` in `get_linfit` with a logged warning. Use default
  slope and offset values instead of failing when there are less than 125 valid
  pixels
* Separate AMSR2 and AMSRE parameter coefficient specifications.

# v0.2.0

* Bootstrap: stop wrapping data computations in functions that cast data to
  `np.float32`. This was done originally to exactly match the outputs of the
  Fortran code provided to us by GSFC.
* Replace custom `linfit` function used by bootstrap algorithm with
  `numpy.polyfit`.
* Refactor of Bootstrap and NASA Team algorithms to allow CDR to apply weather
  filters and land spillover adjustments independently of the sea ice
  concentration calculation.
* Replace `fetch` subpackage with usage of `pm_tb_data`. See the [`pm_tb_data`
  repository](https://github.com/nsidc/pm_tb_data) for more information.
* Bootstrap parameters:
  * Begin refactoring of how parameters are defined (e.g., begin to remove
    concept of `TiepointSet`)
  * DRY out duplicate paramters and make the chain of inheritance more clear
    (e.g., the initial BT params used by the CDR for `AU_SI12` was provided by
    Goddard for AMSRU and adapted for our use-case (weather filter parameters
    updated for `AU_SI12`).
  * Define tests that assert pre-defined initial bootstrap params are not
    changed. These sets of parameters should be immutable. New sets of
    parameters that inherit from a previous set should make it clear that they
    do so instead of overwriting existing values

# 0.1.1

* Packaging fixup: include `py.typed` in built package so that other libraries
  (e.g., `nise` can run type checking against types defined in this package)

# 0.1.0

* Initial dev release.
