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
    """用于在DataIOC中唯一标识一个数据和/或构造数据的数据类型描述符

    Parameters
    ----------
    id
        数据id

    Notes
    -----
    `DataDescriptor` 需要作为字典的键，因此子类需要保证可哈希与可比较，如果每个成员变量均满足条件，则子类自动满足该条件。

    相同类型的多个不同数据可以通过id来区分，不显式指定时默认为 `0` 。
    例如三个个同类型的传感器读数可以使用 `Sensor` 、 `Sensor[1]` 和 `Sensor[2]` 来标识，
    此时他们在 `DataIOC` 中可以映射到不同的数据上。

    Examples
    --------
    默认创建的数据类型描述符例如 `Sensor` 具有弱id，代表他们是用于指示数据类型而非具体的某个数据，
    如果在 __build__ 中使用这类描述符从 `DataIOC` 中提取数据，则会自动重映射到具体的id索引。
    >>> from deinterf.utils.data_ioc import DataNDArray
    >>>
    >>> class SensorData(DataNDArray):
    >>>     def __new__(cls, data, **kwargs):
    >>>         return super().__new__(cls, data, **kwargs)
    >>>
    >>> class Sum(DataDescriptor):
    >>>     def __build__(self, container: DataIOC):
    >>>        return container[SensorData].sum()  # 不指定id，则获取的 SensorData 和 Sum 具有相同的id
    >>>
    >>> container = DataIOC().with_data(SensorData([1, 1, 1]), SensorData[1]([2, 2, 2]))  # 不显式指定时默认为 `0` 号数据
    >>> print(container[Sum[0]], container[Sum[1]])
    3 6

    手动指定索引的数据类型描述符例如 `Sensor[1]` 具有强id，代表他们是用于指示某组具体的数据，
    在 __build__ 中不会被重新映射。

    >>> class OffsetSensor0(DataDescriptor):
    >>>     def __build__(self, container: DataIOC):
    >>>        base = container[SensorData[0]].sum()  # 指定id，显式获取 0 号 SensorData
    >>>        return base + container[SensorData]
    >>>
    >>> print(container[OffsetSensor0[0]], container[OffsetSensor0[1]])
    [4, 4, 4] [5, 5, 5]
    """
    __slots__ = ['_id']
    DefaultWeakID = -1

    def __init__(self, id=DefaultWeakID) -> None:
        self.id = id

    @property
    def id(self):
        if self.signed_id < 0:
            return -self.signed_id - 1
        else:
            return self.signed_id

    @id.setter
    def id(self, val):
        self._id = val

    @property
    def signed_id(self):
        return self._id

    @property
    def is_weak_id(self):
        return self.signed_id < 0

    def __build__(self, container: DataIOC) -> DataT: ...

    def index_weak(self, new_index, check_overriding=True):
        return self.index(new_index, weak=True, check_overriding=check_overriding)

    def index(self, new_index, weak=False, check_overriding=True):
        """将当前数据描述符绑定到新的id索引，以实现关联同类别的不同数据

        例如三个个同类型的传感器读数可以使用：

        * Sensor[0]
        * Sensor[1]
        * Sensor[2]

        标识

        Parameters
        ----------
        new_index
            新的数据索引
        weak
            是否为弱索引
        check_overriding
            是否检查强索引重绑定

        Returns
        -------
        新的索引id的数据类型
        """
        if new_index < 0:
            weak = True
        elif weak and new_index >= 0:
            new_index = -new_index - 1

        ret = self
        if not weak or self.is_weak_id:
            # 如果为强索引，或者当前为弱索引，则可以覆盖，重映射到新索引位置下的数据
            ret = copy.copy(ret)
            ret.id = new_index
        else:
            if check_overriding and not weak and not self.is_weak_id:
                raise RuntimeError(f'Trying to rebind strong id {self.id} to weak id {new_index}.')

        return ret

    def __getitem__(self, index) -> DataDescriptor[DataT]:
        """直接强制绑定为新的id索引

        Notes
        -----
        内部代码应优先使用 `index_weak` 以支持弱id的自动重绑定。
        """
        return self.index(index, check_overriding=False)

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
        ret = _extract_builder(self.dtype, self.signed_id)(container)
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
        """数据IOC容器

        Parameters
        ----------
        allow_implicit_registering
            允许隐式注册Data，若为False则只能获取/构建已注册的Data
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
            default_id = data_type.signed_id

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
                # 如果是带索引的类型，则进一步搜索该类型的通用构造器
                builder = self._lazy_collection.get(dtype.dtype, None)
                if builder is not None:
                    builder = _bind_builder_context(dtype.id, builder)

        return builder


class IndexedDataIOC(DataIOC):
    def __init__(self, base_container: DataIOC, id=DataDescriptor.DefaultWeakID):
        super().__init__()
        self._base_container = base_container
        self._id = id

    @property
    def id(self):
        return self._id

    def __getattr__(self, item):
        return getattr(self._base_container, item)

    def __getitem__(self, item: Type[DataT] | DataDescriptor[DataT]) -> DataT:
        if isinstance(item, DataDescriptor):
            return self._base_container[item.index_weak(self.id)]
        elif isinstance(item, type):
            return self._base_container[IndexedDataTypeDescriptor(dtype=item, id=self.id)]
        else:
            return self._base_container[item]


def _extract_builder(dtype: DataDescriptor | SupportsBuild | Callable, id=None):
    if isinstance(dtype, DataDescriptor):
        id = dtype.signed_id

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
