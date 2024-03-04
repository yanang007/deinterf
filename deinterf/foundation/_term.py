from __future__ import annotations

import numpy as np
from typing_extensions import Union

from deinterf.utils.data_ioc import DataDescriptor, DataIOC


class ComposableTerm(DataDescriptor[np.ndarray]):
    def __or__(self, other: ComposableTerm):
        return Composition(self, other)


class Composition(ComposableTerm):
    __slots__ = ['terms']

    def __init__(
            self,
            terms: Union[ComposableTerm, Composition],
            *other_terms: Union[ComposableTerm, Composition],
            **kwargs
    ):
        super().__init__(**kwargs)

        self.terms = []
        for term in [terms, *other_terms]:
            if isinstance(term, (tuple, list)):
                self.terms.extend(term)
            if isinstance(term, Composition):
                self.terms.extend(term.terms)
            else:
                self.terms.append(term)

        self.terms = tuple(self.terms)

    def __getitem__(self, item):
        return type(self)(*(term[item] for term in self.terms))

    def __build__(self, container: DataIOC):
        return np.column_stack([container[term] for term in self.terms])
