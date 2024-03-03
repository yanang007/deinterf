import numpy as np
from numpy import ndarray
from numpy.typing import ArrayLike
from sklearn.utils._param_validation import validate_params
from sklearn.utils.validation import check_array


@validate_params(
    {
        "magvec": ["array-like"],
        "copy": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def magvec2modulus(magvec: ArrayLike, copy=True) -> ndarray:
    """
    根据磁矢量计算磁总场

    Parameters
    ----------
    magvec : array-like of shape (n_samples, 3)
        磁矢量数据，第二维度对应 x, y, z 三轴

    Returns
    -------
    tmi : ndarray of shape (n_samples,)
        磁总场数据
    """
    magvec = check_array(magvec, ensure_min_features=3, copy=copy)
    return np.linalg.norm(magvec, axis=1)


@validate_params(
    {
        "magvec": ["array-like"],
        "copy": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def magvec2dircosine(magvec: ArrayLike, copy=True) -> ndarray:
    """
    根据磁矢量计算姿态方向余弦

    Parameters
    ----------
    magvec : array-like of shape (n_samples, 3)
        磁矢量数据，第二维度对应 x, y, z 三轴

    Returns
    -------
    dir_cosine : ndarray of shape (n_samples, 3)
        姿态方向余弦，第二维度对应 x, y, z 三轴
    """
    magvec = check_array(magvec, ensure_min_features=3, copy=copy)

    bx, by, bz = np.transpose(magvec)

    modulus = magvec2modulus(magvec)

    dir_conise_x = bx / modulus
    dir_conise_y = by / modulus
    dir_conise_z = bz / modulus

    return np.column_stack((dir_conise_x, dir_conise_y, dir_conise_z))
