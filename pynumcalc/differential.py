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
    def quotient(
        cls, f: typing.Callable[[float], float], h: float
    ) -> typing.Callable[[float], float]:
        """
        :param f:
        :param h:
        :return:
        """
        try:
            fdiff = FiniteDifference.central(f, h)
        except ValueError:
            try:
                fdiff = FiniteDifference.foward(f, h)
            except ValueError:
                fdiff = FiniteDifference.backward(f, h)

        return lambda x: fdiff(x) / h

    @classmethod
    def quotient2(
        cls, f: typing.Callable[[float], float], h: float
    ) -> typing.Callable[[float], float]:
        """
        :param f:
        :param h:
        :return:
        """
        try:
            fdiff = FiniteDifference.central2(f, h)
        except ValueError:
            try:
                fdiff = FiniteDifference.foward2(f, h)
            except ValueError:
                fdiff = FiniteDifference.backward2(f, h)

        return lambda x: fdiff(x) / (h ** 2)

    @classmethod
    def quotientn(
        cls, f: typing.Callable[[float], float], h: float, n: int
    ) -> typing.Callable[[float], float]:
        """
        :param f:
        :param h:
        :param n:
        :return:
        """
        try:
            fdiff = FiniteDifference.centraln(f, h, n)
        except ValueError:
            try:
                fdiff = FiniteDifference.fowardn(f, h, n)
            except ValueError:
                fdiff = FiniteDifference.backwardn(f, h, n)

        return lambda x: fdiff(x) / (h ** n)
