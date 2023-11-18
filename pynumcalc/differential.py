"""
"""

import typing

import numpy as np
import scipy.special

from . import typedef


class FiniteDifference:
    r"""
    Computes finite differences of one-dimensional real-valued functions.

    - *Forward Finite Difference*: :math:`{\Delta}_{h}[f](x)`
    - *Backward Finite Difference*: :math:`{\nabla}_{h}[f](x)`
    - *Central Finite Difference*: :math:`{\delta}_{h}[f](x)`
    """
    @staticmethod
    @typedef.FiniteDifferenceC
    def forward(f: typedef.RealFunction, h: float, x: float) -> typedef.RealFunction:
        r"""
        Computes the first-order forward finite difference of a one-dimensional real-valued
        function ``f`` (:math:`f: \mathbb{R} \mapsto \mathbb{R}`) at ``x`` using step size ``h``.

        .. math::

            {\Delta}_{h} [f](x) = f(x + h) - f(x)
        """
        return f(x + h) - f(x)

    @staticmethod
    @typedef.FiniteDifferenceC
    def forward2(f: typedef.RealFunction, h: float, x: float) -> typedef.RealFunction:
        r"""
        Computes the second-order forward finite difference of a one-dimensional real-valued
        function ``f`` (:math:`f: \mathbb{R} \mapsto \mathbb{R}`) at ``x`` using step size ``h``.

        .. math::

            {\Delta}_{h}^{2} [f](x) = f(x + 2h) - 2f(x + h) + f(x)
        """
        return f(x + 2 * h) - 2 * f(x + h) + f(x)

    @staticmethod
    @typedef.FiniteDifferenceC
    def forwardn(f: typedef.RealFunction, h: float, x: float, n: int) -> typedef.RealFunction:
        r"""
        Computes the ``n``\th-order forward finite difference of a one-dimensional real-valued
        function ``f`` (:math:`f: \mathbb{R} \mapsto \mathbb{R}`) at ``x`` using step size ``h``.

        .. math::

            {\Delta}_{h}^{n} [f](x)
            = \sum_{i = 0}^{n} {(-1)}^{n - i} {{n}\choose{i}} f(x + ih)
        """
        array = np.arange(0, n + 1)
        return ((-1) ** (n - array) * scipy.special.comb(n, array) * f(x + array * h)).sum()

    @staticmethod
    @typedef.FiniteDifferenceC
    def backward(f: typedef.RealFunction, h: float, x: float) -> typedef.RealFunction:
        r"""
        Computes the first-order backward finite difference of a one-dimensional real-valued
        function ``f`` (:math:`f: \mathbb{R} \mapsto \mathbb{R}`) at ``x`` using step size ``h``.

        .. math::

            {\nabla}_{h} [f](x) = f(x) - f(x - h)
        """
        return f(x) - f(x - h)

    @staticmethod
    @typedef.FiniteDifferenceC
    def backward2(f: typedef.RealFunction, h: float, x: float) -> typedef.RealFunction:
        r"""
        Computes the second-order backward finite difference of a one-dimensional real-valued
        function ``f`` (:math:`f: \mathbb{R} \mapsto \mathbb{R}`) at ``x`` using step size ``h``.

        .. math::

            {\nabla}_{h}^{2} [f](x) = f(x) - 2f(x - h) + f(x - 2h)
        """
        return f(x) - 2 * f(x - h) + f(x - 2 * h)

    @staticmethod
    @typedef.FiniteDifferenceC
    def backwardn(f: typedef.RealFunction, h: float, x: float, n: int) -> typedef.RealFunction:
        r"""
        Computes the ``n``\th-order backward finite difference of a one-dimensional real-valued
        function ``f`` (:math:`f: \mathbb{R} \mapsto \mathbb{R}`) at ``x`` using step size ``h``.

        .. math::

            {\nabla}_{h}^{n} [f](x)
            = \sum_{i = 0}^{n} {(-1)}^{i} {{n}\choose{i}} f(x - ih)
        """
        array = np.arange(0, n + 1)
        return ((-1) ** array * scipy.special.comb(n, array) * f(x - array * h)).sum()

    @staticmethod
    @typedef.FiniteDifferenceC
    def central(f: typedef.RealFunction, h: float, x: float) -> typedef.RealFunction:
        r"""
        Computes the first-order central finite difference of a one-dimensional real-valued
        function ``f`` (:math:`f: \mathbb{R} \mapsto \mathbb{R}`) at ``x`` using step size ``h``.

        .. math::

            {\delta}_{h} [f](x) = f(x + \frac{h}{2}) - f(x - \frac{h}{2})
        """
        return f(x + h / 2) - f(x - h / 2)

    @staticmethod
    @typedef.FiniteDifferenceC
    def central2(f: typedef.RealFunction, h: float, x: float) -> typedef.RealFunction:
        r"""
        Computes the second-order central finite difference of a one-dimensional real-valued
        function ``f`` (:math:`f: \mathbb{R} \mapsto \mathbb{R}`) at ``x`` using step size ``h``.

        .. math::

            {\delta}_{h}^{2} [f](x) = f(x + h) - 2f(x) + f(x - h)
        """
        return f(x + h) - 2 * f(x) + f(x - h)

    @staticmethod
    @typedef.FiniteDifferenceC
    def centraln(f: typedef.RealFunction, h: float, x: float, n: int) -> typedef.RealFunction:
        r"""
        Computes the ``n``\th-order central finite difference of a one-dimensional real-valued
        function ``f`` (:math:`f: \mathbb{R} \mapsto \mathbb{R}`) at ``x`` using step size ``h``.

        .. math::

            {\delta}_{h}^{n} [f](x)
            = \sum_{i = 0}^{n} {(-1)}^{i} {{n}\choose{i}} f(x + (\frac{n}{2} - i)h)
        """
        array = np.arange(0, n + 1)
        return ((-1) ** array * scipy.special.comb(n, array) * f(x + (n / 2 - array) * h)).sum()


class PFiniteDifference(FiniteDifference):
    r"""
    Computes partial finite differences of :math:`n`-dimensional real-valued functions.

    - *Forward Finite Difference*:

        .. math::

            {\Delta}_{h}{[f]}(\vec{x})
            = \begin{bmatrix}
                {\Delta}_{h}{[f]}_{{x}_{1}}(\vec{x}) \\
                \vdots \\
                {\Delta}_{h}{[f]}_{{x}_{\dim{\vec{x}}}}(\vec{x}) \\
            \end{bmatrix}

    - *Backward Finite Difference*:

        .. math::

            {\nabla}_{h}{[f]}(\vec{x})
            = \begin{bmatrix}
                {\nabla}_{h}{[f]}_{{x}_{1}}(\vec{x}) \\
                \vdots \\
                {\nabla}_{h}{[f]}_{{x}_{\dim{\vec{x}}}}(\vec{x}) \\
            \end{bmatrix}

    - *Central Finite Difference*:

        .. math::

            {\delta}_{h}{[f]}(\vec{x})
            = \begin{bmatrix}
                {\delta}_{h}{[f]}_{{x}_{1}}(\vec{x}) \\
                \vdots \\
                {\delta}_{h}{[f]}_{{x}_{\dim{\vec{x}}}}(\vec{x}) \\
            \end{bmatrix}
    """
    @staticmethod
    @typedef.PFiniteDifferenceC
    def pforward(f: typedef.RealFunction, h: float, ndim: int, x: np.ndarray) -> typedef.RealFunction:
        r"""
        Computes the first-order partial forward finite differences of a ``dim``-dimensional
        real-valued function ``f`` (:math:`f: \mathbb{R}^{n} \mapsto \mathbb{R}`) at ``x`` using
        step size ``h``.

        .. math::

            {\Delta}_{h} {[f]}_{{x}_{i}}(\vec{x})
            = f(
                \langle {x}_{1}, \dots, {x}_{i} + h, \dots, {x}_{\dim{\vec{x}}} \rangle
            ) - f(\vec{x})
        """
        return f([*x[:ndim], x[ndim] + h, *x[(ndim + 1):]]) - f(x)

    @staticmethod
    @typedef.PFiniteDifferenceC
    def pforward2(f: typedef.RealFunction, h: float, ndim: int, x: np.ndarray) -> typedef.RealFunction:
        r"""
        Computes the second-order partial forward finite differences of a ``dim``-dimensional
        real-valued function ``f`` (:math:`f: \mathbb{R}^{n} \mapsto \mathbb{R}`) at ``x`` using
        step size ``h``.

        .. math::

            {\Delta}_{h}^{2} {[f]}_{{x}_{i}}(\vec{x})
            = f(
                \langle {x}_{1}, \dots, {x}_{i} + 2h, \dots, {x}_{\dim{\vec{x}}} \rangle
            ) - 2f(
                \langle {x}_{1}, \dots, {x}_{i} + h, \dots, {x}_{\dim{\vec{x}}} \rangle
            ) + f(\vec{x})
        """
        return f(
            [*x[:ndim], x[ndim] + 2 * h, *x[(ndim + 1):]]
        ) - 2 * f(
            [*x[:ndim], x[ndim] + h, *x[(ndim + 1):]]
        ) + f(x)

    @staticmethod
    @typedef.PFiniteDifferenceC
    def pforwardn(f: typedef.RealFunction, h: float, ndim: int, x: np.ndarray, n: int) -> typedef.RealFunction:
        r"""
        Computes the ``n``\th-order partial forward finite differences of a ``dim``-dimensional
        real-valued function ``f`` (:math:`f: \mathbb{R}^{n} \mapsto \mathbb{R}`) at ``x`` using
        step size ``h``.

        .. math::

            {\Delta}_{h}^{n} {[f]}_{{x}_{i}}(\vec{x})
            = \sum_{k = 0}^{n} {(-1)}^{n - k} {{n}\choose{k}} f(
                \langle {x}_{1}, \dots, {x}_{i} + kh, \dots, {x}_{\dim{\vec{x}}} \rangle
            )
        """
        array = np.arange(0, n + 1)
        return (
            (-1) ** (n - array) * scipy.special.comb(n, array) * f(
                [*x[:ndim], x[ndim] + array * h, *x[(ndim + 1):]]
            )
        ).sum()

    @staticmethod
    @typedef.PFiniteDifferenceC
    def pbackward(f: typedef.RealFunction, h: float, ndim: int, x: np.ndarray) -> typedef.RealFunction:
        r"""
        Computes the first-order partial backward finite differences of a ``dim``-dimensional
        real-valued function ``f`` (:math:`f: \mathbb{R}^{n} \mapsto \mathbb{R}`) at ``x`` using
        step size ``h``.

        .. math::

            {\nabla}_{h} {[f]}_{{x}_{i}}(\vec{x})
            = f(\vec{x}) - f(
                \langle {x}_{1}, \dots, {x}_{i} - h, \dots, {x}_{\dim{\vec{x}}} \rangle
            )
        """
        return f(x) - f([*x[:ndim], x[ndim] - h, *x[(ndim + 1):]])

    @staticmethod
    @typedef.PFiniteDifferenceC
    def pbackward2(f: typedef.RealFunction, h: float, ndim: int, x: np.ndarray) -> typedef.RealFunction:
        r"""
        Computes the second-order partial backward finite differences of a ``dim``-dimensional
        real-valued function ``f`` (:math:`f: \mathbb{R}^{n} \mapsto \mathbb{R}`) at ``x`` using
        step size ``h``.

        .. math::

            {\nabla}_{h}^{2} {[f]}_{{x}_{i}}(\vec{x})
            = f(\vec{x}) - 2f(
                \langle {x}_{1}, \dots, {x}_{i} - h, \dots, {x}_{\dim{\vec{x}}} \rangle
            ) + f(
                \langle {x}_{1}, \dots, {x}_{i} - 2h, \dots, {x}_{\dim{\vec{x}}} \rangle
            )
        """
        return f(x) - 2 * f(
            [*x[:ndim], x[ndim] - h, *x[(ndim + 1):]]
        ) + f(
            [*x[:ndim], x[ndim] - 2 * h, *x[(ndim + 1):]]
        )

    @staticmethod
    @typedef.PFiniteDifferenceC
    def pbackwardn(f: typedef.RealFunction, h: float, ndim: int, x: np.ndarray, n: int) -> typedef.RealFunction:
        r"""
        Computes the ``n``\th-order partial backward finite differences of a ``dim``-dimensional
        real-valued function ``f`` (:math:`f: \mathbb{R}^{n} \mapsto \mathbb{R}`) at ``x`` using
        step size ``h``.

        .. math::

            {\nabla}_{h}^{n} {[f]}_{{x}_{i}}(\vec{x})
            = \sum_{k = 0}^{n} {(-1)}^{k} {{n}\choose{k}} f(
                \langle {x}_{1}, \dots, {x}_{i} - kh, \dots, {x}_{\dim{\vec{x}}} \rangle
            )
        """
        array = np.arange(0, n + 1)
        return (
            (-1) ** array * scipy.special.comb(n, array) * f(
                [*x[:ndim], x[ndim] - array * h, *x[(ndim + 1):]]
            )
        ).sum()
    
    @staticmethod
    @typedef.PFiniteDifferenceC
    def pcentral(f: typedef.RealFunction, h: float, ndim: int, x: np.ndarray) -> typedef.RealFunction:
        r"""
        Computes the first-order partial central finite differences of a ``dim``-dimensional
        real-valued function ``f`` (:math:`f: \mathbb{R}^{n} \mapsto \mathbb{R}`) at ``x`` using
        step size ``h``.

        .. math::

            {\nabla}_{h} {[f]}_{{x}_{i}}(\vec{x})
            = f(
                \langle {x}_{1}, \dots, {x}_{i} + \frac{h}{2}, \dots, {x}_{\dim{\vec{x}}} \rangle
            ) - f(
                \langle {x}_{1}, \dots, {x}_{i} - \frac{h}{2}, \dots, {x}_{\dim{\vec{x}}} \rangle
            )
        """
        return f(
            [*x[:ndim], x[ndim] + h / 2, *x[(ndim + 1):]]
        ) - f(
            [*x[:ndim], x[ndim] - h / 2, *x[(ndim + 1):]]
        )

    @staticmethod
    @typedef.PFiniteDifferenceC
    def pcentral2(f: typedef.RealFunction, h: float, ndim: int, x: np.ndarray) -> typedef.RealFunction:
        r"""
        Computes the second-order partial central finite differences of a ``dim``-dimensional
        real-valued function ``f`` (:math:`f: \mathbb{R}^{n} \mapsto \mathbb{R}`) at ``x`` using
        step size ``h``.

        .. math::

            {\nabla}_{h}^{2} {[f]}_{{x}_{i}}(\vec{x})
            = f(
                \langle {x}_{1}, \dots, {x}_{i} + h, \dots, {x}_{\dim{\vec{x}}} \rangle
            ) - 2f(\vec{x}) + f(
                \langle {x}_{1}, \dots, {x}_{i} - h, \dots, {x}_{\dim{\vec{x}}} \rangle
            )
        """
        return f(
            [*x[:ndim], x[ndim] + h, *x[(ndim + 1):]]
        ) - 2 * f(x) + f(
            [*x[:ndim], x[ndim] - h, *x[(ndim + 1):]]
        )

    @staticmethod
    @typedef.PFiniteDifferenceC
    def pcentraln(f: typedef.RealFunction, h: float, ndim: int, x: np.ndarray, n: int) -> typedef.RealFunction:
        r"""
        Computes the ``n``\th-order partial central finite differences of a ``dim``-dimensional
        real-valued function ``f`` (:math:`f: \mathbb{R}^{n} \mapsto \mathbb{R}`) at ``x`` using
        step size ``h``.

        .. math::

            {\nabla}_{h}^{n} {[f]}_{{x}_{i}}(\vec{x})
            = \sum_{k = 0}^{n} {(-1)}^{k} {{n}\choose{k}} f(
                \langle
                {x}_{1}, \dots, {x}_{i} + (\frac{n}{2} - k)h, \dots, {x}_{\dim{\vec{x}}}
                \rangle
            )
        """
        array = np.arange(0, n + 1)
        return (
            (-1) ** array * scipy.special.comb(n, array) * f(
                [*x[:ndim], x[ndim] + (n / 2 - array) * h, *x[(ndim + 1):]]
            )
        ).sum()


class DifferenceQuotient:
    """
    """
    @classmethod
    def quotient(
        cls, f: typing.Callable[[float], float], h: float
    ) -> typing.Callable[[float], float]:
        r"""
        """
        try:
            fdiff = FiniteDifference.central(f, h)
        except ValueError:
            try:
                fdiff = FiniteDifference.forward(f, h)
            except ValueError:
                fdiff = FiniteDifference.backward(f, h)

        return lambda x: fdiff(x) / h

    @classmethod
    def quotient2(
        cls, f: typing.Callable[[float], float], h: float
    ) -> typing.Callable[[float], float]:
        r"""
        """
        try:
            fdiff = FiniteDifference.central2(f, h)
        except ValueError:
            try:
                fdiff = FiniteDifference.forward2(f, h)
            except ValueError:
                fdiff = FiniteDifference.backward2(f, h)

        return lambda x: fdiff(x) / (h ** 2)

    @classmethod
    def quotientn(
        cls, f: typing.Callable[[float], float], h: float, n: int
    ) -> typing.Callable[[float], float]:
        r"""
        """
        try:
            fdiff = FiniteDifference.centraln(f, h, n)
        except ValueError:
            try:
                fdiff = FiniteDifference.forwardn(f, h, n)
            except ValueError:
                fdiff = FiniteDifference.backwardn(f, h, n)

        return lambda x: fdiff(x) / (h ** n)


class PDifferenceQuotient:
    """
    """
    @classmethod
    def pquotient(
        cls, f: typing.Callable[[typing.Sequence[float]], float], h: float, dim: int,
        *, ndim: int = None
    ) -> typing.Sequence[typing.Callable[[float], float]]:
        r"""
        """
        def partial(
            fdiff_: typing.Callable[[typing.Sequence[float]], float], h_: float
        ) -> typing.Callable[[typing.Sequence[float]], float]:
            """
            """
            return lambda x: fdiff_(x) / h_
        
        try:
            fdiff = PFiniteDifference.pcentral(f, h, dim, ndim=ndim)
        except ValueError:
            try:
                fdiff = PFiniteDifference.pforward(f, h, dim, ndim=ndim)
            except ValueError:
                fdiff = PFiniteDifference.pbackward(f, h, dim, ndim=ndim)

        if ndim is not None:
            return partial(fdiff, h)
        return [partial(diff, h) for diff in fdiff]
    
    @classmethod
    def pquotient2(
        cls, f: typing.Callable[[typing.Sequence[float]], float], h: float, dim: int
    ) -> typing.Sequence[typing.Callable[[float], float]]:
        r"""
        """
        def partial(
            fdiff_: typing.Callable[[typing.Sequence[float]], float], h_: float
        ) -> typing.Callable[[typing.Sequence[float]], float]:
            """
            """
            return lambda x: fdiff_(x) / (h_ ** 2)
        
        try:
            fdiff = PFiniteDifference.pcentral2(f, h, dim)
        except ValueError:
            try:
                fdiff = PFiniteDifference.pforward2(f, h, dim)
            except ValueError:
                fdiff = PFiniteDifference.pbackward2(f, h, dim)

        return [partial(diff, h) for diff in fdiff]
    
    @classmethod
    def pquotientn(
        cls, f: typing.Callable[[typing.Sequence[float]], float], h: float, n: int, dim: int
    ) -> typing.Sequence[typing.Callable[[float], float]]:
        r"""
        """
        def partial(
            fdiff_: typing.Callable[[typing.Sequence[float]], float], h_: float, n_: int
        ) -> typing.Callable[[typing.Sequence[float]], float]:
            """
            """
            return lambda x: fdiff_(x) / (h_ ** n_)
        
        try:
            fdiff = PFiniteDifference.pcentraln(f, h, n, dim)
        except ValueError:
            try:
                fdiff = PFiniteDifference.pforwardn(f, h, n, dim)
            except ValueError:
                fdiff = PFiniteDifference.pbackwardn(f, h, n, dim)

        return [partial(diff, h, n) for diff in fdiff]
