# NEXT_RELEASE

* Bootstrap: stop wrapping data computations in functions that cast data to
  `np.float32`. This was done originally to exactly match the outputs of the
  Fortran code provided to us by GSFC.
* Replace custom `linfit` function used by bootstrap algorithm with
  `numpy.polyfit`.
* Refactor of Bootstrap and NASA Team algorithms to allow CDR to apply weather
  filters and land spillover adjustments independently of the sea ice
  concentration calculation.

# 0.1.1

* Packaging fixup: include `py.typed` in built package so that other libraries
  (e.g., `nise` can run type checking against types defined in this package)

# 0.1.0

* Initial dev release.
