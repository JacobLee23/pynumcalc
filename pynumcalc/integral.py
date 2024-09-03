"""
Numerical integral calculus.
"""

import functools
import itertools
import operator
import typing

import numpy as np

from .types import RealFunction, RealFunctionN


class Interval:
    """
    :param lower: The lower bound of the summation interval
    :param upper: The upper bound of the summation interval
    :param n: The number of partitions to divide the interval
    """
    def __init__(self, lower: float, upper: float, n: int):
        self._lower = lower
        self._upper = upper
        self._n = n

    def __repr__(self) -> str:
        attributes = ("lower", "upper", "n")
        arguments = ", ".join(f"{k}={getattr(self, k)}" for k in attributes)

        return f"{self.__class__.__name__}({arguments})"

    def __str__(self) -> str:
        return f"[{self.lower}, {self.upper}] / {self.n}"

    def __iter__(self):
        return iter((self.lower, self.upper, self.n))

    def __add__(self, other: float) -> "Interval":
        res = self.__class__.__new__(self.__class__)
        res.__init__(self.lower + other, self.upper + other, self.n)
        return res

    def __sub__(self, other: float) -> "Interval":
        res = self.__class__.__new__(self.__class__)
        res.__init__(self.lower - other, self.upper - other, self.n)
        return res

    def __mul__(self, other: float) -> "Interval":
        res = self.__class__.__new__(self.__class__)
        res.__init__(self.lower * other, self.upper * other, self.n)
        return res

    def __truediv__(self, other: float) -> "Interval":
        res = self.__class__.__new__(self.__class__)
        res.__init__(self.lower / other, self.upper / other, self.n)
        return res

    def __floordiv__(self, other: float) -> "Interval":
        res = self.__class__.__new__(self.__class__)
        res.__init__(self.lower // other, self.upper // other, self.n)
        return res

    def __mod__(self, other: float) -> "Interval":
        res = self.__class__.__new__(self.__class__)
        res.__init__(self.lower % other, self.upper % other, self.n)
        return res

    @property
    def lower(self) -> float:
        """
        The lower bound of the summation interval.
        """
        return self._lower

    @lower.setter
    def lower(self, value: float) -> None:
        self._lower = value

    @property
    def upper(self) -> float:
        """
        The upper bound of the summation interval.
        """
        return self._upper

    @upper.setter
    def upper(self, value: float) -> None:
        self._upper = value

    @property
    def n(self) -> int:
        """
        The number of partitions to divide the interval.
        """
        return self._n

    @n.setter
    def n(self, value: int) -> None:
        """
        """
        self._n = value

    @property
    def delta(self) -> float:
        """
        """
        return (self.upper - self.lower) / self.n

    @property
    def left(self) -> np.ndarray:
        r"""
        .. math::
        
            x_{i}^{*} = x_{i-1} = a + i \Delta x
        """
        return self.lower + np.arange(self.n) * self.delta

    @property
    def right(self) -> np.ndarray:
        r"""
        .. math::
        
            x_{i}^{*} = x_{i} = a + (i + 1) \Delta x
        """
        return self.lower + (np.arange(self.n) + 1) * self.delta

    @property
    def middle(self) -> np.ndarray:
        r"""
        .. math::
        
            x_{i}^{*} = \frac{x_{i-1} + x_{i}}{2} = a + (i + \frac{i}{2}) \Delta x
        """
        return self.lower + (np.arange(self.n) + 1 / 2) * self.delta


class RiemannSum:
    """
    :param f: A callable representation of a real-valued function
    :param dim: The dimension of the domain of ``f``
    """
    _interval = None

    @typing.overload
    def __init__(self, f: RealFunction, interval: Interval, dim: int = None): ...

    @typing.overload
    def __init__(self, f: RealFunctionN, interval: typing.Sequence[Interval], dim: int): ...

    def __init__(
        self, f: typing.Union[RealFunction, RealFunctionN],
        interval: typing.Union[Interval, typing.Sequence[Interval]],
        dim: typing.Optional[int] = None
    ):
        self._f = f
        self._dim = dim

        self.interval = interval

    @property
    def f(self) -> typing.Union[RealFunction, RealFunctionN]:
        r"""
        A real-valued function defined by :math:`f: {\mathbb{R}}^{n} \to \mathbb{R}` if
        :py:attr:`dim` is not ``None``, otherwise :math:`f: \mathbb{R} \to \mathbb{R}`.
        """
        return self._f

    @property
    def interval(self) -> typing.Union[Interval, typing.Sequence[Interval]]:
        """
        """
        return self._interval

    @interval.setter
    def interval(self, value: typing.Union[Interval, typing.Sequence[Interval]]) -> None:
        if self.dim is None and not isinstance(value, Interval):
            raise TypeError(value)
        elif self.dim is not None:
            if len(value) != self.dim:
                raise ValueError(value)
            if not all(isinstance(x, Interval) for x in value):
                raise TypeError(value)

        self._interval = value

    @property
    def dim(self) -> int:
        """
        The dimension of the domain of :py:attr:`f`. If ``None``, then :py:attr:`f` is a
        one-dimensional function.
        """
        return self._dim

    @property
    def delta(self) -> float:
        """
        Computes the unit length/area/volume/hypervolume to be used in the Riemann summation.

        +-----------------------+---------------+
        | :py:attr:`dim`        | Unit          |
        +=======================+===============+
        | ``None``              | Length        |
        +-----------------------+---------------+
        | ``1``                 | Length        |
        +-----------------------+---------------+
        | ``2``                 | Area          |
        +-----------------------+---------------+
        | ``3``                 | Volume        |
        +-----------------------+---------------+
        | Greater than ``3``    | Hypervolume   |
        +-----------------------+---------------+
        """
        if self.dim is None:
            return self.interval.delta
        return functools.reduce(
            operator.mul, (i.upper - i.lower for i in self.interval)
        ) / functools.reduce(
            operator.mul, (i.n for i in self.interval)
        )

    def summation(self) -> float:
        """
        :return:
        """
        if self.dim is None:
            return self.delta * (
                self.f(self.interval.left).sum() + self.f(self.interval.right).sum()
            ) / 2

        coordinates = map(np.array, itertools.product([i.left, i.right] for i in self.interval))
        return self.interval.delta * sum(self.f(c) for c in coordinates) / pow(2, self.dim)
