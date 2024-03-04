from __future__ import annotations

import copy
import inspect
from functools import partial
from typing import Any, Dict, Protocol, Type, runtime_checkable, Callable, Generic, TypeVar, overload

from typing_extensions import Self

DataT = TypeVar('DataT')


@runtime_checkable
class SupportsBuild(Protocol):
    def __build__(self, container: DataIOC) -> Any: ...


class DataDescriptor(Generic[DataT]):
    __slots__ = ['_id']

    def __init__(self, id=0) -> None:
        self._id = id

    @property
    def id(self):
        return self._id

    def __build__(self, container: DataIOC) -> DataT: ...

    def __getitem__(self, index) -> DataDescriptor[DataT]:
        ret = copy.copy(self)
        ret._id = index

        return ret

    def __class_getitem__(cls, id, *args):
        if isinstance(id, int):
            return cls(*args, id=id)
        else:
            return super().__class_getitem__(id, *args)

    def __hash__(self):
        return hash(tuple(getattr(self, k) for k in self.keys))

    def __eq__(self, other):
        if type(self) is not type(other):
            return False

        for k in self.keys:
            if getattr(self, k) != getattr(other, k):
                return False
        else:
            return True

    @property
    def keys(self):
        keys = []
        for c in reversed(inspect.getmro(type(self))):
            keys += getattr(c, '__slots__', [])

        keys.extend(getattr(self, '__dict__', []))

        return set(keys)

    @property
    def params(self):
        return {k.strip("_"): getattr(self, k) for k in self.keys}

    def __repr__(self):
        params = self.params
        del params['id']

        id_str = f'[{self._id}]' if self.id > 0 else ''
        param_str = ', '.join([f'{k}={repr(v)}' for k, v in params.items()])

        return f'{type(self).__name__}{id_str}({param_str})'

    def __copy__(self):
        return type(self)(**self.params)


class IndexedDataTypeDescriptor(DataDescriptor[DataT]):
    __slots__ = ['_dtype']

    def __new__(cls, dtype, *args, **kwargs):
        if issubclass(dtype, DataDescriptor):
            return dtype(*args, **kwargs)
        else:
            return super().__new__(cls)

    def __init__(self, dtype: Type, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._dtype = dtype

    @property
    def dtype(self):
        return self._dtype

    def __build__(self, container: DataIOC) -> DataT:
        ret = _extract_builder(self.dtype, self.id)(container)
        return ret


class IndexedType(type):
    def __getitem__(self, id, *args):
        return IndexedDataTypeDescriptor(self, *args, id=id)

    def __repr__(self):
        return f'{self.__name__}'


class DataIOC:
    def __init__(self, allow_implicit_register=True):
        """
        :param allow_implicit_register: 允许隐式注册Data，若为False则只能获取/构建已注册的Data
        """
        self._collection: Dict[DataDescriptor | Type, Any] = {}
        self._lazy_collection: Dict[DataDescriptor | Type, Callable[[DataIOC], Any]] = {}
        self.allow_implicit_register = allow_implicit_register

    def with_data(self, *data: Any) -> Self:
        for d in data:
            self[type(d)] = d

        return self

    def add(self, data_type: DataT | Type[DataT] | DataDescriptor[DataT], data: DataT | None = None) -> Self:
        builder = None
        if data is None:
            if isinstance(data_type, DataDescriptor):
                builder = _extract_builder(data_type)
            else:
                data_type, data = type(data_type), data_type

        if builder is None:
            self[data_type] = data
        else:
            self._lazy_collection[data_type] = builder

        return self

    @overload
    def __getitem__(self, dtype: DataDescriptor[DataT]) -> DataT: ...

    @overload
    def __getitem__(self, dtype: Type[DataDescriptor[DataT]]) -> DataT: ...

    @overload
    def __getitem__(self, dtype: Type[DataT]) -> DataT: ...

    def __getitem__(self, dtype: Type[DataT] | DataDescriptor[DataT]) -> DataT:
        ret = self._collection.get(dtype, None)
        if ret is None:
            builder = self._lazy_collection.get(dtype, None)
            if builder is None:
                if not self.allow_implicit_register:
                    raise RuntimeError(f'Builder for {repr(dtype)} not found in {DataIOC.__name__}.')
                else:
                    self.add(dtype)
                    ret = self[dtype]
            else:
                ret = builder(self)

            self[dtype] = ret

        return ret

    def __setitem__(self, data_type: Type[DataT] | DataDescriptor[DataT], data: DataT):
        self._collection[data_type] = data
        if isinstance(data_type, IndexedDataTypeDescriptor) and data_type.id == 0:
            self._collection[data_type.dtype] = data
        if isinstance(data_type, type):
            # 对于直接用类名绑定，则默认同时绑定对应的0号数据
            self._collection[IndexedDataTypeDescriptor(dtype=data_type)] = data


class IndexedDataIOC(DataIOC):
    def __init__(self, base_container: DataIOC, id=0):
        super().__init__()
        self._base_container = base_container
        self._id = id

    @property
    def id(self):
        return self._id

    def __getattr__(self, item):
        return getattr(self._base_container, item)

    def __getitem__(self, item: Type[DataT] | DataDescriptor[DataT]) -> DataT:
        if self._id != 0 and isinstance(item, DataDescriptor):
            return self._base_container[item[self._id]]
        elif isinstance(item, type):
            return self._base_container[IndexedDataTypeDescriptor(dtype=item, id=self._id)]
        else:
            return self._base_container[item]


def _extract_builder(dtype: Type | DataDescriptor, id=0):
    if isinstance(dtype, DataDescriptor):
        id = dtype.id

    if isinstance(dtype, SupportsBuild):
        builder = dtype.__build__
    elif isinstance(dtype, type):
        builder = dtype
    else:
        builder = None

    if builder is None:
        return builder
    else:
        return partial(_bind_default_index, id, builder)


def _bind_default_index(id: int, builder, container: DataIOC):
    return builder(IndexedDataIOC(container, id=id))
