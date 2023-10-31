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
        .. math

            {\Delta}_{h}^{2}[f](x) = f(x + 2h) - 2f(x + h) + f(x)

        :param f:
        :param h:
        :return:
        """
        return lambda x: f(x + 2 * h) - 2 * f(x + h) + f(x)
    
    @staticmethod
    def forwardn(
        f: typing.Callable[[float], float], h: float, n: int
    ) -> typing.Callable[[float], float]:
        r"""
        .. math

            {\Delta}_{h}^{n}[f](x) = \sum_{i = 0}^{n} {(-1)}^{n - i} {{n}\choose{i}} f(x + ih)

        :param f:
        :param h:
        :parma n:
        :return:
        """
        array = np.arange(0, n + 1)
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
        .. math

            {\nabla}_{h}^{2}[f](x) = f(x) - 2f(x - h) + f(x - 2h)

        :param f:
        :param h:
        :return:
        """
        return lambda x: f(x) - 2 * f(x - h) + f(x - 2 * h)

    @staticmethod
    def backwardn(
        f: typing.Callable[[float], float], h: float, n: int
    ) -> typing.Callable[[float], float]:
        r"""
        .. math

            {\nabla}_{h}^{n}[f](x) = \sum_{i = 0}^{n} {(-1)}^{i} {{n}\choose{i}} f(x - ih)

        :param f:
        :param h:
        :param n:
        :return:
        """
        array = np.arange(0, n + 1)
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
        .. math

            {\delta}_{h}^{2}[f](x) = f(x + h) - 2f(x) + f(x - h)

        :param f:
        :param h:
        :return:
        """
        return lambda x: f(x + h) - 2 * f(x) + f(x - h)
    
    @staticmethod
    def centraln(
        f: typing.Callable[[float], float], h: float, n: int
    ) -> typing.Callable[[float], float]:
        r"""
        .. math

            {\delta}_{h}^{n}[f](x) = \sum_{i = 0}^{n} {(-1)}^{i} {{n}\choose{i}} f(x + (\frac{n}{2} - i)h)

        :param f:
        :param h:
        :param n:
        :return:
        """
        array = np.arange(0, n + 1)
        return lambda x: (
            (-1) ** array * scipy.special.comb(n, array) * f(x + (n / 2 - array) * h)
        ).sum()
