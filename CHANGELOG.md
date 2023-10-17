# NEXT_RELEASE

* Extract CDR-specific logic from this repository. Moved to the `seaice_ecdr`
  (https://github.com/nsidc/seaice_ecdr/)

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
