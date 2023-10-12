"""
"""

import itertools
import typing

import numpy as np


class Integrate:
    """
    """
    @staticmethod
    def left(
        lower: float, upper: float, npartitions: int
    ) -> np.ndarray[typing.Any, np.dtype[np.float64]]:
        r"""
        .. math::

            x_{i}^{*} = x_{i-1} = a + i \Delta x

        :param lower: The lower bound of the summation interval
        :param upper: The upper bound of the summation interval
        :param npartitions: The number of partitions dividing the interval
        :return:
        """
        length = (upper - lower) / npartitions
        return lower + np.arange(npartitions) * length

    @staticmethod
    def middle(
        lower: float, upper: float, npartitions: int
    ) -> np.ndarray[typing.Any, np.dtype[np.float64]]:
        r"""
        .. math::

            x_{i}^{*} = \frac{x_{i-1} + x_{i}}{2} = a + (i + \frac{1}{2}) \Delta x

        :param lower: The lower bound of the summation interval
        :param upper: The upper bound of the summation interval
        :param npartitions: The number of partitions dividing the interval
        :return:
        """
        length = (upper - lower) / npartitions
        return lower + (np.arange(npartitions) + 1 / 2) * length

    @staticmethod
    def right(
        lower: float, upper: float, npartitions: int
    ) -> np.ndarray[typing.Any, np.dtype[np.float64]]:
        r"""
        .. math::

            x_{i}^{*} = x_{i} = a + (i + 1) \Delta x

        :param lower: The lower bound of the summation interval
        :param upper: The upper bound of the summation interval
        :param npartitions: The number of partitions dividing the interval
        :return:
        """
        length = (upper - lower) / npartitions
        return lower + (np.arange(npartitions) + 1) * length

    @classmethod
    def riemann_sum(
        cls, function: typing.Callable[[typing.Sequence], float], axes: np.ndarray[typing.Any, np.dtype[np.float64]]
    ) -> float:
        """
        :param function:
        :param axes:
        :param method:
        :return:
        """
        delta = (axes[:, 1] - axes[:, 0]).prod()
        coordinates = np.dstack(np.meshgrid(*axes)).reshape(-1, 2)
        return np.apply_along_axis(function, coordinates, 1).sum() * delta
    
    @classmethod
    def integrate(
        cls, function: typing.Callable[[typing.Sequence], float], *intervals: typing.Tuple[float, float, int]
    ) -> float:
        """
        :param function:
        :param intervals:
        :return:
        """
        dimensions = np.array([[cls.left(*x), cls.right(*x)] for x in intervals])

        return sum(
            cls.riemann_sum(function, np.array([a[i] for a, i in zip(dimensions, indices)]))
            for indices in itertools.product((0, 1), repeat=len(intervals))
        ) / pow(2, len(intervals))
