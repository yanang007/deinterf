from __future__ import annotations

import numpy as np

from deinterf.foundation._data import Data, DataIOC
from deinterf.foundation._term import ComposableTerm
from deinterf.foundation.sensors import DirectionalCosine, MagModulus


class Permanent(ComposableTerm):
    @classmethod
    def __build__(cls, container: DataIOC, id=0) -> Data:
        return container[DirectionalCosine][id]


class Induced6(ComposableTerm):
    @classmethod
    def __build__(cls, container: DataIOC, id=0) -> Data:
        modulus = container[MagModulus][id].data
        cos_x, cos_y, cos_z = container[DirectionalCosine][id].data.T
        # (12301, 1) * (12301, 6) -> (12301, 6)
        feats = np.einsum(
            "ij,ij->ij",
            modulus,
            np.column_stack(
                (
                    cos_x * cos_x,
                    cos_x * cos_y,
                    cos_x * cos_z,
                    cos_y * cos_y,  # removed in Induced_5 version
                    cos_y * cos_z,
                    cos_z * cos_z,
                )
            ),
        )
        return Data(*feats.T)


class Induced5(ComposableTerm):
    @classmethod
    def __build__(cls, container: DataIOC, id=0) -> Data:
        feats = Induced6.__build__(container, id).data
        feats = np.delete(feats, feats.shape[1] // 2, 1)
        return Data(*feats.T)


class Induced(Induced5): ...


class Eddy9(ComposableTerm):
    @classmethod
    def __build__(cls, container: DataIOC, id=0) -> Data:
        modulus = container[MagModulus][id].data
        cos_x, cos_y, cos_z = container[DirectionalCosine][id].data.T
        cos_x_dot = np.gradient(cos_x)
        cos_y_dot = np.gradient(cos_y)
        cos_z_dot = np.gradient(cos_z)
        feats = np.einsum(
            "ij,ij->ij",
            modulus,
            np.column_stack(
                (
                    cos_x * cos_x_dot,
                    cos_x * cos_y_dot,
                    cos_x * cos_z_dot,
                    cos_y * cos_x_dot,
                    cos_y * cos_y_dot,  # removed in Eddy_8 version
                    cos_y * cos_z_dot,
                    cos_z * cos_x_dot,
                    cos_z * cos_y_dot,
                    cos_z * cos_z_dot,
                )
            ),
        )
        return Data(*feats.T)


class Eddy8(ComposableTerm):
    @classmethod
    def __build__(cls, container: DataIOC, id=0) -> Data:
        feats = Eddy9.__build__(container, id).data
        feats = np.delete(feats, feats.shape[1] // 2, 1)
        return Data(*feats.T)


class Eddy(Eddy8): ...
