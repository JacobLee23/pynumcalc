"""
"""

import functools
import typing

import numpy as np


@typing.runtime_checkable
class RealFunction(typing.Protocol):
    """
    """
    def __call__(self, element: float) -> float: ...


@typing.runtime_checkable
class RealFunctionN(typing.Protocol):
    """
    """
    def __call__(self, element: np.ndarray) -> np.ndarray: ...


class FiniteDifferenceC:
    """
    :param func:
    """
    @typing.overload
    def __init__(
        self, func: typing.Callable[[RealFunction, float, float], RealFunction]
    ): ...

    @typing.overload
    def __init__(
        self, func: typing.Callable[[RealFunction, float, float, int], RealFunction]
    ): ...

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
        def fdiff(x: float) -> float:
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
    def __init__(
        self, func: typing.Callable[[RealFunction, float, int, np.ndarray], RealFunction]
    ): ...

    @typing.overload
    def __init__(
        self, func: typing.Callable[[RealFunction, float, int, np.ndarray, int], RealFunction]
    ): ...

    def __init__(self, func: typing.Callable):
        self.func = func

        functools.update_wrapper(self, self.func)

    @typing.overload
    def __call__(self, f: RealFunction, h: float, dim: int) -> RealFunctionN: ...

    @typing.overload
    def __call__(self, f: RealFunction, h: float, dim: int, n: int) -> RealFunctionN: ...

    def __call__(self, f: RealFunction, h: float, dim: int, n: int = None) -> RealFunctionN:
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
            x = np.array(x) if not isinstance(x, np.ndarray) else x
            if x.shape != (dim,):
                raise ValueError(
                    f"Expected np.ndarray object of shape {(dim,)}, received {x.shape}"
                )

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


class DifferenceQuotientC:
    """
    :param func:
    """
    @typing.overload
    def __init__(
        self, func: typing.Callable[[RealFunction, float, float], RealFunction]
    ): ...

    @typing.overload
    def __init__(
        self, func: typing.Callable[[RealFunction, float, float, int], RealFunction]
    ): ...

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
        def fdiff(x: float) -> float:
            """
            :param x:
            :return:
            """
            return self.func(f, h, x) if n is None else self.func(f, h, x, n)

        fdiff.__doc__ = self.func.__doc__
        return fdiff


class PDifferenceQuotientC:
    """
    :param func:
    """
    @typing.overload
    def __init__(
        self, func: typing.Callable[[RealFunctionN, float, int, np.ndarray, int], np.ndarray]
    ): ...

    @typing.overload
    def __init__(
        self, func: typing.Callable[[RealFunctionN, float, int, np.ndarray, int, int], np.ndarray]
    ): ...

    def __init__(self, func: typing.Callable):
        self.func = func

        functools.update_wrapper(self, self.func)

    @typing.overload
    def __call__(self, f: RealFunctionN, h: float, dim: int) -> RealFunctionN: ...

    @typing.overload
    def __call__(self, f: RealFunctionN, h: float, dim: int, n: int) -> RealFunctionN: ...

    def __call__(self, f: RealFunctionN, h: float, dim: int, n: int = None) -> RealFunctionN:
        """
        :param f:
        :param h:
        :param dim:
        :param n:
        :return:
        """
        def fdiff(x: np.ndarray, *, ndim: int = None) -> np.ndarray:
            """
            :param x:
            :param ndim:
            :return:
            """
            return self.func(f, h, dim, x, ndim=ndim) if n is None else self.func(f, h, dim, x, n, ndim=ndim)

        fdiff.__doc__ == self.func.__doc__
        return fdiff
