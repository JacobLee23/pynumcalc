"""
"""

import typing


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
    def central(f: typing.Callable[[float], float], h: float) -> typing.Callable[[float], float]:
        r"""
        .. math

            {\delta}_{h}[f](x) = f(x + \frac{h}{2}) - f(x - \frac{h}{2})

        :param f:
        :param h:
        :return:
        """
        return lambda x: f(x + h / 2) - f(x - h / 2)
