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
