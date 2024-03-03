from __future__ import annotations

import numpy as np
from typing_extensions import List, Union

from deinterf.foundation._data import Data, DataIOC
from deinterf.foundation.sensors import DirectionalCosine, MagModulus


class ComposableTerm(Data):
    def __init__(self) -> None: ...

    def __or__(self, other: ComposableTerm):
        return Composition(self, other)


class Composition(ComposableTerm):
    def __init__(
        self,
        term1: Union[ComposableTerm, Composition],
        term2: Union[ComposableTerm, Composition],
    ):
        terms1 = self._validate(term1)
        terms2 = self._validate(term2)
        self.terms = terms1 + terms2

    def __build__(self, container: DataIOC, id=0) -> Data:
        terms_d = np.column_stack([container[term][id].data for term in self.terms])
        return Data(*terms_d.T)

    # validate term input
    def _validate(
        self, term: Union[ComposableTerm, Composition]
    ) -> List[ComposableTerm]:
        if not isinstance(term, ComposableTerm):
            raise TypeError(
                f"Term must be an instance of ComposableTerm, not {type(term)}"
            )
        if isinstance(term, Composition):
            return term.terms
        else:
            return [term]


class Permanent(ComposableTerm):
    @classmethod
    def __build__(cls, container: DataIOC) -> Data:
        return container[DirectionalCosine][0]


class Induced6(ComposableTerm):
    @classmethod
    def __build__(cls, container: DataIOC) -> Data:
        modulus = container[MagModulus][0].data
        cos_x, cos_y, cos_z = container[DirectionalCosine][0].data.T
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
    def __build__(cls, container: DataIOC) -> Data:
        feats = Induced6.__build__(container).data
        feats = np.delete(feats, feats.shape[1] // 2, 1)
        return Data(*feats.T)


class Induced(Induced5): ...


class Eddy9(ComposableTerm):
    @classmethod
    def __build__(cls, container: DataIOC) -> Data:
        modulus = container[MagModulus][0].data
        cos_x, cos_y, cos_z = container[DirectionalCosine][0].data.T
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
                    cos_y * cos_y_dot,  # removed in Terms.Eddy_8 version
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
    def __build__(cls, container: DataIOC) -> Data:
        feats = Eddy9.__build__(container).data
        feats = np.delete(feats, feats.shape[1] // 2, 1)
        return Data(*feats.T)


class Eddy(Eddy8): ...


Terms16 = Permanent() | Induced5() | Eddy8()
