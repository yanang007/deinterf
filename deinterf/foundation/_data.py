from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.utils.validation import check_consistent_length, column_or_1d
from typing_extensions import Any, Dict, Protocol, Self, Type, Union


class Data:
    def __init__(self, *data: ArrayLike) -> None:
        check_consistent_length(*data)
        self.data = np.column_stack([column_or_1d(v, dtype=np.float64) for v in data])


class SupportsBuild(Protocol):
    def __build__(self, container: DataIOC, id: int) -> Data: ...


class AutoBuildDict(dict):
    def __init__(self, factory, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.factory = factory

    def __missing__(self, key: int):
        if not isinstance(key, int):
            raise TypeError(f"Key must be an integer, not {type(key)}")
        if self.factory is not None:
            self[key] = self.factory(key)
            return self[key]
        else:
            raise KeyError(key)

    def __setitem__(self, key: int, value: Data):
        if not isinstance(key, int):
            raise TypeError(f"Key must be an integer, not {type(key)}")
        if not issubclass(type(value), Data):
            raise TypeError(f"Value must be a subclass of Data, not {type(value)}")
        super().__setitem__(key, value)


class DataIOC:
    def __init__(self):
        self._collection: Dict[Type[Data], Dict[int, Data]] = {}

    def __getitem__(self, dtype: Union[Type[Data], Data]) -> Dict[int, Data]:
        if isinstance(dtype, Data):
            dtype_instance = dtype
            dtype = type(dtype)
            if dtype not in self._collection:
                self._collection[dtype] = AutoBuildDict(
                    lambda id: self.build(dtype_instance, id)
                )
        elif issubclass(dtype, Data):
            if dtype not in self._collection:
                self._collection[dtype] = AutoBuildDict(
                    lambda id: self.build(dtype, id)
                )
        else:
            raise TypeError(f"{dtype} must be a subclass or instance of Data")
        return self._collection[dtype]

    def build(self, dtype: Union[Type[Data], Data], id: int) -> Data:
        builder = getattr(dtype, "__build__", None)
        if builder:
            return builder(self, id)
        raise KeyError(f"{dtype} not found and no builder available")

    def add(self, data: Data, id=0) -> Self:
        if (dtype := type(data)) not in self._collection:
            self._collection[dtype] = {}
        self._collection[dtype][id] = data
        return self

    def _is_dtype(self, dtype: type) -> bool:
        return issubclass(dtype, Data)

    def _is_dinstance(self, data: Any) -> bool:
        return isinstance(data, Data)
