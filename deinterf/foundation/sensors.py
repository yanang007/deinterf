from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from deinterf.utils.data_ioc import DataNDArray, DataIOC
from deinterf.utils.transform import magvec2dircosine, magvec2modulus


class MagVector(DataNDArray):
    def __new__(cls, bx: ArrayLike, by: ArrayLike, bz: ArrayLike, **kwargs):
        return super().__new__(cls, bx, by, bz, **kwargs)

    @property
    def bx(self):
        return self[:, 0]

    @property
    def by(self):
        return self[:, 1]

    @property
    def bz(self):
        return self[:, 2]


class MagModulus(DataNDArray):
    @classmethod
    def __build__(cls, container: DataIOC):
        modulus = magvec2modulus(container[MagVector])
        return cls(modulus)


class Tmi(DataNDArray):
    def __new__(cls, tmi: ArrayLike):
        return super().__new__(cls, tmi)


class DirectionalCosine(DataNDArray):
    def __new__(cls, dir_cosine_x: ArrayLike, dir_cosine_y: ArrayLike, dir_cosine_z: ArrayLike):
        return super().__new__(cls, dir_cosine_x, dir_cosine_y, dir_cosine_z)

    @classmethod
    def __build__(cls, container: DataIOC) -> DirectionalCosine:
        dir_cosine = magvec2dircosine(container[MagVector])
        dir_cosine_x, dir_cosine_y, dir_cosine_z = np.transpose(dir_cosine)
        return cls(
            dir_cosine_x=dir_cosine_x,
            dir_cosine_y=dir_cosine_y,
            dir_cosine_z=dir_cosine_z,
        )

    @property
    def dcosx(self):
        return self[:, 0]

    @property
    def dcosy(self):
        return self[:, 1]

    @property
    def dcosz(self):
        return self[:, 2]
