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
    """用于在DataIOC中唯一标识一个数据和/或构造数据的描述符

    Parameters
    ----------
    id
        数据id

    Notes
    -----
    `DataDescriptor` 需要作为字典的键，因此子类需要保证可哈希与可比较，如果每个成员变量均满足条件，则子类自动满足该条件。

    如果id为负，则代表该描述符为弱id（非用户手动指定的id），在 `__build__` 过程中会自动重绑定到对应的id上，

    如果id为正，则代表该描述符为强id（用户手动指定的id或重绑定后的id），在 `__build__` 过程中不会被重绑定。

    References
    ----------
    id重绑定过程 : :class:`IndexedDataIOC` 。
    """

    __slots__ = ['_id']

    def __init__(self, id=-1) -> None:
        self.id = id

    @property
    def id(self):
        if self._id < 0:
            return -self._id - 1
        else:
            return self._id

    @id.setter
    def id(self, val):
        self._id = val

    @property
    def is_implicit_id(self):
        return self._id < 0

    def __build__(self, container: DataIOC) -> DataT: ...

    def __getitem__(self, index) -> DataDescriptor[DataT]:
        if not self.is_implicit_id:
            raise RuntimeError(f'Trying to rebind strong id {self.id} to {index}.')

        ret = copy.copy(self)
        ret.id = index

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
            keys.extend(getattr(c, '__slots__', []))
        keys.extend(getattr(self, '__dict__', []))

        keys = set(keys)
        keys.remove('_id')
        keys.add('id')  # 保证 id 值恒为正

        return keys

    @property
    def params(self):
        return {k.strip("_"): getattr(self, k) for k in self.keys}

    def __repr__(self):
        params = self.params
        del params['id']

        id_str = f'[{self.id}]' if self.id > 0 else ''
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

    def __call__(self, *args, **kwargs):
        return DescribedData(
            self,
            self.dtype(*args, **kwargs)
        )


class DescribedData:
    def __init__(self, desc: DataDescriptor[DataT], data: DataT):
        self.desc = desc
        self.data = data


class IndexedType(type):
    def __getitem__(self, id, *args):
        return IndexedDataTypeDescriptor(self, *args, id=id)

    def __repr__(self):
        return f'{self.__name__}'


class DataIOC:
    def __init__(self, allow_implicit_registering=True):
        """
        :param allow_implicit_registering: 允许隐式注册Data，若为False则只能获取/构建已注册的Data
        """
        self._collection: Dict[DataDescriptor | Type, Any] = {}
        self._lazy_collection: Dict[DataDescriptor | Type, Callable[[DataIOC], Any]] = {}
        self.allow_implicit_register = allow_implicit_registering

    def with_data(self, *data: Any) -> Self:
        for d in data:
            if isinstance(d, DescribedData):
                self[d.desc] = d.data
            else:
                self[type(d)] = d

        return self

    def add(self, data_type: DataT | Type[DataT] | DataDescriptor[DataT], data: DataT | None = None) -> Self:
        if data is None:
            if isinstance(data_type, DataDescriptor) or isinstance(data_type, type):
                builder = _extract_builder(data_type)
                self._lazy_collection[data_type] = builder
            else:
                self.with_data(data_type)
        else:
            self[data_type] = data

        return self

    def add_provider(
            self,
            data_type: Type[DataT] | DataDescriptor[DataT],
            provider: SupportsBuild | Callable
    ) -> Self:
        default_id = None
        if isinstance(data_type, DataDescriptor):
            default_id = data_type.id

        self._lazy_collection[data_type] = _extract_builder(provider, id=default_id)

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
            builder = self.find_builder(dtype)
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

    def find_builder(self, dtype: Type[DataT] | DataDescriptor[DataT]):
        """查找构造器

        * 直接以类别指定的构造器：适用于所有id下的 IndexedDataTypeDescriptor ，可以通用

        * 以特定 DataDescriptor 指定的构造器：只适用于特定的 DataDescriptor
        """
        builder = self._lazy_collection.get(dtype, None)
        if builder is None:
            if isinstance(dtype, IndexedDataTypeDescriptor):
                builder = self._lazy_collection.get(dtype.dtype, None)
                if builder is not None:
                    builder = _bind_builder_context(dtype.id, builder)

        return builder


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
        if self._id != 0 and isinstance(item, DataDescriptor) and item.is_implicit_id:
            return self._base_container[item[self._id]]
        elif isinstance(item, type):
            return self._base_container[IndexedDataTypeDescriptor(dtype=item, id=self._id)]
        else:
            return self._base_container[item]


def _extract_builder(dtype: DataDescriptor | SupportsBuild | Callable, id=None):
    if isinstance(dtype, DataDescriptor):
        id = dtype.id

    if isinstance(dtype, SupportsBuild):
        builder = dtype.__build__
    elif callable(dtype):
        # 类型的构造函数
        # 或者直接的构造器函数
        # TODO: 添加校验，验证可以用于DataIOC
        builder = dtype
    else:
        builder = None

    if builder is None:
        return builder
    else:
        return _bind_builder_context(id, builder)


def _bind_builder_context(id, builder):
    if id is None:
        return builder
    else:
        return partial(_bind_default_index, id, builder)


def _bind_default_index(id: int, builder, container: DataIOC):
    return builder(IndexedDataIOC(container, id=id))
