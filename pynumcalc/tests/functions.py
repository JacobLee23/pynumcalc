"""
Test case functions for the unit tests in :py:mod:`pynumcalc.tests.test_differential`.
"""

from numbers import Number
import typing

import numpy as np

from pynumcalc.types import RealFunction


DOMAIN = np.linspace(-10, 10)


class RealFunctionCase(typing.NamedTuple):
    r"""
    .. py:attribute:: f

        A real-valued function defined by :math:`f: {\mathbb{R}}^{n} \to \mathbb{R}`.

    .. py:attribute:: forward1

        A second-order function that returns a callable representation of the explicit definition
        of the first-order forward finite difference of :py:attr:`f`.

    .. py:attribute:: forward2

        A second-order function that returns a callable representation of the explicit definition
        of the second-order forward finite difference of :py:attr:`f`.

    .. py:attribute:: backward1

        A second-order function that returns a callable representation of the explicit definition
        of the first-order backward finite difference of :py:attr:`f`.

    .. py:attribute:: backward2

        A second-order function that returns a callable representation of the explicit definition
        of the second-order backward finite difference of :py:attr:`f`.

    .. py:attribute:: central1

        A second-order function that returns a callable representation of the explicit definition
        of the first-order central finite difference of :py:attr:`f`.

    .. py:attribute:: central2

        A second-order function that returns a callable representation of the explicit definition
        of the second-order central finite difference of :py:attr:`f`.

    .. py:attribute:: dquotient1

        A second-order function that returns a callable representation of the explicit definition of
        the first-order difference quotient of :py:attr:`f`.

    .. py:attribute:: dquotient2

        A second-order function that returns a callable representation of the explicit definition of
        the second-order difference quotient of :py:attr:`f`.

    .. py:attribute:: domain

        An array representing a subset of the interval over which to test :py:attr:`f`.
    """
    f: RealFunction
    forward1: typing.Callable[[float], RealFunction]
    forward2: typing.Callable[[float], RealFunction]
    backward1: typing.Callable[[float], RealFunction]
    backward2: typing.Callable[[float], RealFunction]
    central1: typing.Callable[[float], RealFunction]
    central2: typing.Callable[[float], RealFunction]
    dquotient1: typing.Callable[[float], RealFunction]
    dquotient2: typing.Callable[[float], RealFunction]
    domain: np.ndarray = DOMAIN
    domain: typing.Tuple[Number, Number] = DOMAIN


FUNCTIONS = [
    RealFunctionCase(
        lambda x: 0,
        lambda h: (lambda x: 0), lambda h: (lambda x: 0),
        lambda h: (lambda x: 0), lambda h: (lambda x: 0),
        lambda h: (lambda x: 0), lambda h: (lambda x: 0),
        lambda h: (lambda x: 0), lambda h: (lambda x: 0)
    ), RealFunctionCase(
        lambda x: 1,
        lambda h: (lambda x: 0), lambda h: (lambda x: 0),
        lambda h: (lambda x: 0), lambda h: (lambda x: 0),
        lambda h: (lambda x: 0), lambda h: (lambda x: 0),
        lambda h: (lambda x: 0), lambda h: (lambda x: 0)
    ), RealFunctionCase(
        lambda x: -1,
        lambda h: (lambda x: 0), lambda h: (lambda x: 0),
        lambda h: (lambda x: 0), lambda h: (lambda x: 0),
        lambda h: (lambda x: 0), lambda h: (lambda x: 0),
        lambda h: (lambda x: 0), lambda h: (lambda x: 0)
    ), RealFunctionCase(
        lambda x: x,
        lambda h: (lambda x: h), lambda h: (lambda x: 0),
        lambda h: (lambda x: h), lambda h: (lambda x: 0),
        lambda h: (lambda x: h), lambda h: (lambda x: 0),
        lambda h: (lambda x: 1), lambda h: (lambda x: 0)
    ), RealFunctionCase(
        lambda x: -x,
        lambda h: (lambda x: -h), lambda h: (lambda x: 0),
        lambda h: (lambda x: -h), lambda h: (lambda x: 0),
        lambda h: (lambda x: -h), lambda h: (lambda x: 0),
        lambda h: (lambda x: -1), lambda h: (lambda x: 0)
    ), RealFunctionCase(
        lambda x: x ** 2,
        lambda h: (lambda x: 2 * x * h + h ** 2), lambda h: (lambda x: 2 * h ** 2),
        lambda h: (lambda x: 2 * x * h - h ** 2), lambda h: (lambda x: 2 * h ** 2),
        lambda h: (lambda x: 2 * x * h), lambda h: (lambda x: 2 * h ** 2),
        lambda h: (lambda x: 2 * x), lambda h: (lambda x: 2)
    ), RealFunctionCase(
        lambda x: -x ** 2,
        lambda h: (lambda x: -2 * x * h - h ** 2), lambda h: (lambda x: -2 * h ** 2),
        lambda h: (lambda x: -2 * x * h + h ** 2), lambda h: (lambda x: -2 * h ** 2),
        lambda h: (lambda x: -2 * x * h), lambda h: (lambda x: -2 * h ** 2),
        lambda h: (lambda x: -2 * x), lambda h: (lambda x: -2)
    ), RealFunctionCase(
        lambda x: x ** 3,
        lambda h: (lambda x: 3 * x ** 2 * h + 3 * x * h ** 2 + h ** 3),
        lambda h: (lambda x: 6 * x * h ** 2 + 6 * h ** 3),
        lambda h: (lambda x: 3 * x ** 2 * h - 3 * x * h ** 2 + h ** 3),
        lambda h: (lambda x: 6 * x * h ** 2 - 6 * h ** 3),
        lambda h: (lambda x: 3 * x ** 2 * h + h ** 3 / 4),
        lambda h: (lambda x: 6 * x * h ** 2),
        lambda h: (lambda x: 3 * x ** 2 + h ** 2 / 4),
        lambda h: (lambda x: 6 * x)
    ), RealFunctionCase(
        lambda x: -x ** 3,
        lambda h: (lambda x: -3 * x ** 2 * h - 3 * x * h ** 2 - h ** 3),
        lambda h: (lambda x: -6 * x * h ** 2 - 6 * h ** 3),
        lambda h: (lambda x: -3 * x ** 2 * h + 3 * x * h ** 2 - h ** 3),
        lambda h: (lambda x: -6 * x * h ** 2 + 6 * h ** 3),
        lambda h: (lambda x: -3 * x ** 2 * h - h ** 3 / 4),
        lambda h: (lambda x: -6 * x * h ** 2),
        lambda h: (lambda x: -3 * x ** 2 - h ** 2 / 4),
        lambda h: (lambda x: -6 * x)
    ), RealFunctionCase(
        lambda x: pow(x, 1 / 2),
        lambda h: (lambda x: pow(x + h, 1 / 2) - pow(x, 1 / 2)),
        lambda h: (lambda x: pow(x + 2 * h, 1 / 2) - 2 * pow(x + h, 1 / 2) + pow(x, 1 / 2)),
        lambda h: (lambda x: pow(x, 1 / 2) - pow(x - h, 1 / 2)),
        lambda h: (lambda x: pow(x, 1 / 2) - 2 * pow(x - h, 1 / 2) + pow(x - 2 * h, 1 / 2)),
        lambda h: (lambda x: pow(x + h / 2, 1 / 2) - pow(x - h / 2, 1 / 2)),
        lambda h: (lambda x: pow(x + h, 1 / 2) - 2 * pow(x, 1 / 2) + pow(x - h, 1 / 2)),
        lambda h: (lambda x: (pow(x + h / 2, 1 / 2) - pow(x - h / 2, 1 / 2)) / h),
        lambda h: (
            lambda x: (pow(x + h, 1 / 2) - 2 * pow(x, 1 / 2) + pow(x - h, 1 / 2)) / pow(h, 2)
        ), np.linspace(1, 10)
    ), RealFunctionCase(
        lambda x: pow(-x, 1 / 2),
        lambda h: (lambda x: pow(-x - h, 1 / 2) - pow(-x, 1 / 2)),
        lambda h: (lambda x: pow(-x - 2 * h, 1 / 2) - 2 * pow(-x - h, 1 / 2) + pow(-x, 1 / 2)),
        lambda h: (lambda x: pow(-x, 1 / 2) - pow(-x + h, 1 / 2)),
        lambda h: (lambda x: pow(-x, 1 / 2) - 2 * pow(-x + h, 1 / 2) + pow(-x + 2 * h, 1 / 2)),
        lambda h: (lambda x: pow(-x - h / 2, 1 / 2) - pow(-x + h / 2, 1 / 2)),
        lambda h: (lambda x: pow(-x - h, 1 / 2) - 2 * pow(-x, 1 / 2) + pow(-x + h, 1 / 2)),
        lambda h: (lambda x: (pow(-x - h / 2, 1 / 2) - pow(-x + h / 2, 1 / 2)) / h),
        lambda h: (
            lambda x: (pow(-x - h, 1 / 2) - 2 * pow(-x, 1 / 2) + pow(-x + h, 1 / 2)) / pow(h, 2)
        ), np.linspace(-10, -1)
    ), RealFunctionCase(
        lambda x: -pow(x, 1 / 2),
        lambda h: (lambda x: -pow(x + h, 1 / 2) + pow(x, 1 / 2)),
        lambda h: (lambda x: -pow(x + 2 * h, 1 / 2) + 2 * pow(x + h, 1 / 2) - pow(x, 1 / 2)),
        lambda h: (lambda x: -pow(x, 1 / 2) + pow(x - h, 1 / 2)),
        lambda h: (lambda x: -pow(x, 1 / 2) + 2 * pow(x - h, 1 / 2) - pow(x - 2 * h, 1 / 2)),
        lambda h: (lambda x: -pow(x + h / 2, 1 / 2) + pow(x - h / 2, 1 / 2)),
        lambda h: (lambda x: -pow(x + h, 1 / 2) + 2 * pow(x, 1 / 2) - pow(x - h, 1 / 2)),
        lambda h: (lambda x: (-pow(x + h / 2, 1 / 2) + pow(x - h / 2, 1 / 2)) / h),
        lambda h: (
            lambda x: (-pow(x + h, 1 / 2) + 2 * pow(x, 1 / 2) - pow(x - h, 1 / 2)) / pow(h, 2)
        ), np.linspace(1, 10)
    ), RealFunctionCase(
        lambda x: -pow(-x, 1 / 2),
        lambda h: (lambda x: -pow(-x - h, 1 / 2) + pow(-x, 1 / 2)),
        lambda h: (lambda x: -pow(-x - 2 * h, 1 / 2) + 2 * pow(-x - h, 1 / 2) - pow(-x, 1 / 2)),
        lambda h: (lambda x: -pow(-x, 1 / 2) + pow(-x + h, 1 / 2)),
        lambda h: (lambda x: -pow(-x, 1 / 2) + 2 * pow(-x + h, 1 / 2) - pow(-x + 2 * h, 1 / 2)),
        lambda h: (lambda x: -pow(-x - h / 2, 1 / 2) + pow(-x + h / 2, 1 / 2)),
        lambda h: (lambda x: -pow(-x - h, 1 / 2) + 2 * pow(-x, 1 / 2) - pow(-x + h, 1 / 2)),
        lambda h: (lambda x: (-pow(-x - h / 2, 1 / 2) + pow(-x + h / 2, 1 / 2)) / h),
        lambda h: (
            lambda x: (-pow(-x - h, 1 / 2) + 2 * pow(-x, 1 / 2) - pow(-x + h, 1 / 2)) / pow(h, 2)
        ), np.linspace(-10, -1)
    ), RealFunctionCase(
        lambda x: pow(x, 1 / 3),
        lambda h: (lambda x: pow(x + h, 1 / 3) - pow(x, 1 / 3)),
        lambda h: (lambda x: pow(x + 2 * h, 1 / 3) - 2 * pow(x + h, 1 / 3) + pow(x, 1 / 3)),
        lambda h: (lambda x: pow(x, 1 / 3) - pow(x - h, 1 / 3)),
        lambda h: (lambda x: pow(x, 1 / 3) - 2 * pow(x - h, 1 / 3) + pow(x - 2 * h, 1 / 3)),
        lambda h: (lambda x: pow(x + h / 2, 1 / 3) - pow(x - h / 2, 1 / 3)),
        lambda h: (lambda x: pow(x + h, 1 / 3) - 2 * pow(x, 1 / 3) + pow(x - h, 1 / 3)),
        lambda h: (lambda x: (pow(x + h / 2, 1 / 3) - pow(x - h / 2, 1 / 3)) / h),
        lambda h: (
            lambda x: (pow(x + h, 1 / 3) - 2 * pow(x, 1 / 3) + pow(x - h, 1 / 3)) / pow(h, 2)
        ), np.linspace(1, 10)
    ), RealFunctionCase(
        lambda x: pow(-x, 1 / 3),
        lambda h: (lambda x: pow(-x - h, 1 / 3) - pow(-x, 1 / 3)),
        lambda h: (lambda x: pow(-x - 2 * h, 1 / 3) - 2 * pow(-x - h, 1 / 3) + pow(-x, 1 / 3)),
        lambda h: (lambda x: pow(-x, 1 / 3) - pow(-x + h, 1 / 3)),
        lambda h: (lambda x: pow(-x, 1 / 3) - 2 * pow(-x + h, 1 / 3) + pow(-x + 2 * h, 1 / 3)),
        lambda h: (lambda x: pow(-x - h / 2, 1 / 3) - pow(-x + h / 2, 1 / 3)),
        lambda h: (lambda x: pow(-x - h, 1 / 3) - 2 * pow(-x, 1 / 3) + pow(-x + h, 1 / 3)),
        lambda h: (lambda x: (pow(-x - h / 2, 1 / 3) - pow(-x + h / 2, 1 / 3)) / h),
        lambda h: (
            lambda x: (pow(-x - h, 1 / 3) - 2 * pow(-x, 1 / 3) + pow(-x + h, 1 / 3)) / pow(h, 2)
        ), np.linspace(-10, -1)
    ), RealFunctionCase(
        lambda x: -pow(x, 1 / 3),
        lambda h: (lambda x: -pow(x + h, 1 / 3) + pow(x, 1 / 3)),
        lambda h: (lambda x: -pow(x + 2 * h, 1 / 3) + 2 * pow(x + h, 1 / 3) - pow(x, 1 / 3)),
        lambda h: (lambda x: -pow(x, 1 / 3) + pow(x - h, 1 / 3)),
        lambda h: (lambda x: -pow(x, 1 / 3) + 2 * pow(x - h, 1 / 3) - pow(x - 2 * h, 1 / 3)),
        lambda h: (lambda x: -pow(x + h / 2, 1 / 3) + pow(x - h / 2, 1 / 3)),
        lambda h: (lambda x: -pow(x + h, 1 / 3) + 2 * pow(x, 1 / 3) - pow(x - h, 1 / 3)),
        lambda h: (lambda x: (-pow(x + h / 2, 1 / 3) + pow(x - h / 2, 1 / 3)) / h),
        lambda h: (
            lambda x: (-pow(x + h, 1 / 3) + 2 * pow(x, 1 / 3) - pow(x - h, 1 / 3)) / pow(h, 2)
        ), np.linspace(1, 10)
    ), RealFunctionCase(
        lambda x: -pow(-x, 1 / 3),
        lambda h: (lambda x: -pow(-x - h, 1 / 3) + pow(-x, 1 / 3)),
        lambda h: (lambda x: -pow(-x - 2 * h, 1 / 3) + 2 * pow(-x - h, 1 / 3) - pow(-x, 1 / 3)),
        lambda h: (lambda x: -pow(-x, 1 / 3) + pow(-x + h, 1 / 3)),
        lambda h: (lambda x: -pow(-x, 1 / 3) + 2 * pow(-x + h, 1 / 3) - pow(-x + 2 * h, 1 / 3)),
        lambda h: (lambda x: -pow(-x - h / 2, 1 / 3) + pow(-x + h / 2, 1 / 3)),
        lambda h: (lambda x: -pow(-x - h, 1 / 3) + 2 * pow(-x, 1 / 3) - pow(-x + h, 1 / 3)),
        lambda h: (lambda x: (-pow(-x - h / 2, 1 / 3) + pow(-x + h / 2, 1 / 3)) / h),
        lambda h: (
            lambda x: (-pow(-x - h, 1 / 3) + 2 * pow(-x, 1 / 3) - pow(-x + h, 1 / 3)) / pow(h, 2)
        ), np.linspace(-10, -1)
    ), RealFunctionCase(
        lambda x: 1 / x,
        lambda h: (lambda x: -h / (x * (x + h))),
        lambda h: (lambda x: 2 * h ** 2 / (x * (x + h) * (x + 2 * h))),
        lambda h: (lambda x: -h / (x * (x - h))),
        lambda h: (lambda x: 2 * h ** 2 / (x * (x - h) * (x - 2 * h))),
        lambda h: (lambda x: -4 * h / ((2 * x + h) * (2 * x - h))),
        lambda h: (lambda x: 2 * h ** 2 / (x * (x + h) * (x - h))),
        lambda h: (lambda x: -4 / ((2 * x + h) * (2 * x - h))),
        lambda h: (lambda x: 2 / (x * (x + h) * (x - h)))
    ), RealFunctionCase(
        lambda x: -1 / x,
        lambda h: (lambda x: h / (x * (x + h))),
        lambda h: (lambda x: -2 * h ** 2 / (x * (x + h) * (x + 2 * h))),
        lambda h: (lambda x: h / (x * (x - h))),
        lambda h: (lambda x: -2 * h ** 2 / (x * (x - h) * (x - 2 * h))),
        lambda h: (lambda x: 4 * h / ((2 * x + h) * (2 * x - h))),
        lambda h: (lambda x: -2 * h ** 2 / (x * (x + h) * (x - h))),
        lambda h: (lambda x: 4 / ((2 * x + h) * (2 * x - h))),
        lambda h: (lambda x: -2 / (x * (x + h) * (x - h)))
    ), RealFunctionCase(
        lambda x: np.log(np.abs(x)),
        lambda h: (lambda x: np.log((x + h) / x)),
        lambda h: (lambda x: np.log((x * (x + 2 * h)) / pow(x + h, 2))),
        lambda h: (lambda x: np.log(x / (x - h))),
        lambda h: (lambda x: np.log((x * (x - 2 * h)) / pow(x - h, 2))),
        lambda h: (lambda x: np.log((2 * x + h) / (2 * x - h))),
        lambda h: (lambda x: np.log((x + h) * (x - h) / pow(x, 2))),
        lambda h: (lambda x: np.log((2 * x + h) / (2 * x - h)) / h),
        lambda h: (lambda x: np.log((x + h) * (x - h) / pow(x, 2)) / pow(h, 2)),
        np.concatenate([np.linspace(-10, -1, endpoint=False), np.linspace(10, 1, endpoint=False)])
    ), RealFunctionCase(
        lambda x: -np.log(np.abs(x)),
        lambda h: (lambda x: -np.log((x + h) / x)),
        lambda h: (lambda x: -np.log((x * (x + 2 * h)) / pow(x + h, 2))),
        lambda h: (lambda x: -np.log(x / (x - h))),
        lambda h: (lambda x: -np.log((x * (x - 2 * h)) / pow(x - h, 2))),
        lambda h: (lambda x: -np.log((2 * x + h) / (2 * x - h))),
        lambda h: (lambda x: -np.log((x + h) * (x - h) / pow(x, 2))),
        lambda h: (lambda x: -np.log((2 * x + h) / (2 * x - h)) / h),
        lambda h: (lambda x: -np.log((x + h) * (x - h) / pow(x, 2)) / pow(h, 2)),
        np.concatenate([np.linspace(-10, -1, endpoint=False), np.linspace(10, 1, endpoint=False)])
    )
]
