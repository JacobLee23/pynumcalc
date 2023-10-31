"""
"""

import typing

import numpy as np
import scipy.special


class FiniteDifference:
    """
    """
    @staticmethod
    def forward(f: typing.Callable[[float], float], h: float) -> typing.Callable[[float], float]:
        r"""
        .. math

            {\Delta}_{h}[f](x) = f(x + h) - f(x)

        :param f:
        :param h:
        :return:
        """
        return lambda x: f(x + h) - f(x)
    
    @staticmethod
    def forward2(f: typing.Callable[[float], float], h: float) -> typing.Callable[[float], float]:
        r"""
        :param f:
        :param h:
        :return:
        """
        return lambda x: f(x + 2 * h) - 2 * f(x + h) + f(x)
    
    @staticmethod
    def forwardn(
        f: typing.Callable[[float], float], n: int, h: float
    ) -> typing.Callable[[float], float]:
        r"""
        :param f:
        :param n:
        :param h:
        :return:
        """
        array = np.arange(0, n)
        return lambda x: (
            (-1) ** (n - array) * scipy.special.comb(n, array) * f(x + array * h)
        ).sum()

    @staticmethod
    def backward(f: typing.Callable[[float], float], h: float) -> typing.Callable[[float], float]:
        r"""
        .. math

            {\nabla}_{h}[f](x) = f(x) - f(x - h)

        :param f:
        :param h:
        :return:
        """
        return lambda x: f(x) - f(x - h)
    
    @staticmethod
    def backward2(f: typing.Callable[[float], float], h: float) -> typing.Callable[[float], float]:
        r"""
        :param f:
        :param h:
        :return:
        """
        return lambda x: f(x) - 2 * f(x - h) + f(x - 2 * h)

    @staticmethod
    def backwardn(
        f: typing.Callable[[float], float], n: int, h: float
    ) -> typing.Callable[[float], float]:
        r"""
        :param f:
        :param n:
        :param h:
        :return:
        """
        array = np.arange(0, n)
        return lambda x: (
            (-1) ** array * scipy.special.comb(n, array) * f(x - array * h)
        ).sum()

    @staticmethod
    def central(f: typing.Callable[[float], float], h: float) -> typing.Callable[[float], float]:
        r"""
        .. math

            {\delta}_{h}[f](x) = f(x + \frac{h}{2}) - f(x - \frac{h}{2})

        :param f:
        :param h:
        :return:
        """
        return lambda x: f(x + h / 2) - f(x - h / 2)
    
    @staticmethod
    def central2(f: typing.Callable[[float], float], h: float) -> typing.Callable[[float], float]:
        r"""
        :param f:
        :param h:
        :return:
        """
        return lambda x: f(x + h) - 2 * f(x) + f(x - h)
    
    @staticmethod
    def centraln(
        f: typing.Callable[[float], float], n: int, h: float
    ) -> typing.Callable[[float], float]:
        r"""
        :param f:
        :param n:
        :param h:
        :return:
        """
        array = np.arange(0, n)
        return lambda x: (
            (-1) ** array * scipy.special.comb(n, array) * f(x - (n / 2 - array) * h)
        ).sum()
