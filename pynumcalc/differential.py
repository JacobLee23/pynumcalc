"""
"""

import typing

import numpy as np
import scipy.special


class FiniteDifference:
    """
    """
    @classmethod
    def forward(
        cls, f: typing.Callable[[float], float], h: float
    ) -> typing.Callable[[float], float]:
        r"""
        .. math

            {\Delta}_{h}[f](x) = f(x + h) - f(x)

        :param f:
        :param h:
        :return:
        """
        return lambda x: f(x + h) - f(x)
    
    @classmethod
    def forward2(
        cls, f: typing.Callable[[float], float], h: float
    ) -> typing.Callable[[float], float]:
        r"""
        .. math

            {\Delta}_{h}^{2}[f](x) = f(x + 2h) - 2f(x + h) + f(x)

        :param f:
        :param h:
        :return:
        """
        return lambda x: f(x + 2 * h) - 2 * f(x + h) + f(x)
    
    @classmethod
    def forwardn(
        cls, f: typing.Callable[[float], float], h: float, n: int
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

    @classmethod
    def backward(
        cls, f: typing.Callable[[float], float], h: float
    ) -> typing.Callable[[float], float]:
        r"""
        .. math

            {\nabla}_{h}[f](x) = f(x) - f(x - h)

        :param f:
        :param h:
        :return:
        """
        return lambda x: f(x) - f(x - h)
    
    @classmethod
    def backward2(
        cls, f: typing.Callable[[float], float], h: float
    ) -> typing.Callable[[float], float]:
        r"""
        .. math

            {\nabla}_{h}^{2}[f](x) = f(x) - 2f(x - h) + f(x - 2h)

        :param f:
        :param h:
        :return:
        """
        return lambda x: f(x) - 2 * f(x - h) + f(x - 2 * h)

    @classmethod
    def backwardn(
        cls, f: typing.Callable[[float], float], h: float, n: int
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

    @classmethod
    def central(
        cls, f: typing.Callable[[float], float], h: float
    ) -> typing.Callable[[float], float]:
        r"""
        .. math

            {\delta}_{h}[f](x) = f(x + \frac{h}{2}) - f(x - \frac{h}{2})

        :param f:
        :param h:
        :return:
        """
        return lambda x: f(x + h / 2) - f(x - h / 2)
    
    @classmethod
    def central2(
        cls, f: typing.Callable[[float], float], h: float
    ) -> typing.Callable[[float], float]:
        r"""
        .. math

            {\delta}_{h}^{2}[f](x) = f(x + h) - 2f(x) + f(x - h)

        :param f:
        :param h:
        :return:
        """
        return lambda x: f(x + h) - 2 * f(x) + f(x - h)
    
    @classmethod
    def centraln(
        cls, f: typing.Callable[[float], float], h: float, n: int
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


class DifferenceQuotient:
    """
    """
    @classmethod
    def quotient(cls, f: typing.Callable[[float], float], h: float) -> typing.Callable[[float], float]:
        """
        :param f:
        :param h:
        :return:
        """
        try:
            return FiniteDifference.central(f, h) / h
        except ValueError:
            try:
                return FiniteDifference.forward(f, h) / h
            except ValueError:
                return FiniteDifference.backward(f, h) / h

    @classmethod
    def quotient2(cls, f: typing.Callable[[float], float], h: float) -> typing.Callable[[float], float]:
        """
        :param f:
        :param h:
        :return:
        """
        try:
            return FiniteDifference.central2(f, h) / h
        except ValueError:
            try:
                return FiniteDifference.forward2(f, h) / h
            except ValueError:
                return FiniteDifference.backward2(f, h) / h
            
    @classmethod
    def quotientn(cls, f: typing.Callable[[float], float], h: float, n: int) -> typing.Callable[[float], float]:
        """
        :param f:
        :param h:
        :param n:
        :return:
        """
        try:
            return FiniteDifference.centraln(f, h, n) / h
        except ValueError:
            try:
                return FiniteDifference.forwardn(f, h, n) / h
            except ValueError:
                return FiniteDifference.backwardn(f, h, n) / h
