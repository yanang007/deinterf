from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from deinterf.foundation._data import Data, DataIOC
from deinterf.utils.transform import magvec2dircosine, magvec2modulus


class MagVector(Data):
    def __init__(self, bx: ArrayLike, by: ArrayLike, bz: ArrayLike) -> None:
        super().__init__(bx, by, bz)


class MagModulus(Data):
    def __init__(self, modulus: ArrayLike) -> None:
        super().__init__(modulus)

    @classmethod
    def __build__(cls, container: DataIOC, id: int) -> Data:
        mag_vec = container[MagVector][id]
        modulus = magvec2modulus(mag_vec.data)
        return cls(modulus=modulus)


class Tmi(Data):
    def __init__(self, tmi: ArrayLike) -> None:
        super().__init__(tmi)

    @classmethod
    def __build__(cls, container: DataIOC, id=0) -> Tmi:
        raise NotImplementedError


class DirectionalCosine(Data):
    def __init__(
        self, dir_cosine_x: ArrayLike, dir_cosine_y: ArrayLike, dir_cosine_z: ArrayLike
    ) -> None:
        super().__init__(dir_cosine_x, dir_cosine_y, dir_cosine_z)

    @classmethod
    def __build__(cls, container: DataIOC, id=0) -> DirectionalCosine:
        dir_cosine = magvec2dircosine(container[MagVector][id].data)
        dir_cosine_x, dir_cosine_y, dir_cosine_z = np.transpose(dir_cosine)
        return DirectionalCosine(
            dir_cosine_x=dir_cosine_x,
            dir_cosine_y=dir_cosine_y,
            dir_cosine_z=dir_cosine_z,
        )
