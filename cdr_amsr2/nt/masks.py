import numpy as np
import numpy.typing as npt

from cdr_amsr2._types import Hemisphere
from cdr_amsr2.constants import PACKAGE_DIR
from cdr_amsr2.util import get_ps25_grid_shape


def get_ps25_sst_mask(
    *,
    hemisphere: Hemisphere,
    sst_threshold=2780,
) -> npt.NDArray[np.bool_]:
    """Read and return the ps25 SST mask.

    `True` elements are those which are masked as invalid ice.
    """
    # TODO: why are the northern hemisphere files 'fixed' while the southern
    # hemisphere are not (except in one case)?
    if hemisphere == 'north':
        sst_fn = 'jan.temp.zdf.ssmi_fixed_25fill.fixed'
    else:
        sst_fn = 'jan.temp.zdf.ssmi_25fill'

    sst_path = (
        PACKAGE_DIR
        / '../legacy'
        / f'nt_orig/DATAFILES/data36/SST/{hemisphere.capitalize()}/'
        / sst_fn
    )
    sst_field = np.fromfile(sst_path, dtype='>i2')[150:].reshape(
        get_ps25_grid_shape(hemisphere=hemisphere)
    )

    where_sst_high = sst_field >= sst_threshold

    return where_sst_high
