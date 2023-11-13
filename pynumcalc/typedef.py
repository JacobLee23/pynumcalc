"""
"""

import functools
import typing

import numpy as np


DomainElement: typing.TypeAlias = typing.Union[float, np.ndarray]


@typing.runtime_checkable
class RealFunction(typing.Protocol):
    """
    """
    @typing.overload
    def __call__(self, element: float) -> float: ...

    @typing.overload
    def __call__(self, element: np.ndarray) -> np.ndarray: ...

    def __call__(self, element: DomainElement) -> DomainElement: ...


class FiniteDifferenceC:
    """
    :param func:
    """
    @typing.overload
    def __init__(self, func: typing.Callable[[RealFunction, float, DomainElement], RealFunction]): ...

    @typing.overload
    def __init__(self, func: typing.Callable[[RealFunction, float, DomainElement, int], RealFunction]): ...

    def __init__(self, func: typing.Callable):
        self.func = func

        functools.update_wrapper(self, self.func)

    @typing.overload
    def __call__(self, f: RealFunction, h: float) -> RealFunction: ...

    @typing.overload
    def __call__(self, f: RealFunction, h: float, n: int) -> RealFunction: ...

    def __call__(self, f: RealFunction, h: float, n: int = None) -> RealFunction:
        """
        :param f:
        :param h:
        :param n:
        :return:
        """
        def fdiff(x: DomainElement) -> DomainElement:
            """
            :param x:
            :return:
            """
            return self.func(f, h, x) if n is None else self.func(f, h, x, n)
        
        fdiff.__doc__ = self.func.__doc__

        return fdiff
    

class PFiniteDifferenceC:
    """
    :param func:
    """
    @typing.overload
    def __init__(self, func: typing.Callable[[RealFunction, float, int, np.ndarray], RealFunction]): ...

    @typing.overload
    def __init__(self, func: typing.Callable[[RealFunction, float, int, np.ndarray, int], RealFunction]): ...

    def __init__(self, func: typing.Callable):
        self.func = func

        functools.update_wrapper(self, self.func)

    @typing.overload
    def __call__(self, f: RealFunction, h: float, dim: int) -> typing.Sequence[RealFunction]: ...

    @typing.overload
    def __call__(self, f: RealFunction, h: float, dim: int, n: int) -> typing.Sequence[RealFunction]: ...

    def __call__(self, f: RealFunction, h: float, dim: int, n: int = None) -> typing.Sequence[RealFunction]:
        """
        :param f:
        :param h:
        :param dim:
        :param n:
        :return:
        """
        @typing.overload
        def fdiff(x: np.ndarray) -> typing.Sequence[np.ndarray]: ...

        @typing.overload
        def fdiff(x: np.ndarray, *, ndim: int) -> np.ndarray: ...

        def fdiff(x: np.ndarray, *, ndim: int = None) -> np.ndarray:
            """
            :param x:
            :param ndim:
            :return:
            :raise ValueError:
            """
            if not isinstance(x, np.ndarray):
                if len(x) != dim:
                    raise ValueError
                x = np.array(x)
            else:
                if x.shape != (dim,):
                    raise ValueError
            
            if ndim is not None:
                return self.func(f, h, ndim, x) if n is None else self.func(f, h, ndim, x, n)

            return np.array(
                [
                    self.func(f, h, ndim, x) for ndim in range(dim)
                ] if n is None else [
                    self.func(f, h, ndim, x, n) for ndim in range(dim)
                ]
            )
        
        fdiff.__doc__ = self.func.__doc__

        return fdiff
