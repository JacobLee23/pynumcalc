"""
"""

import typing

import numpy as np
import scipy.special


class FiniteDifference:
    """
    Computes finite differences of one-dimensional real-valued functions and partial finite
    differences of :math:`n`-dimensional real-valued functions.
    """
    @classmethod
    def forward(
        cls, f: typing.Callable[[float], float], h: float
    ) -> typing.Callable[[float], float]:
        r"""
        Computes the first-order forward finite difference of a one-dimensional real-valued
        function ``f`` (:math:`f: \mathbb{R} \mapsto \mathbb{R}`) using step size ``h``.

        .. math::

            {\Delta}_{h}[f](x) = f(x + h) - f(x)
        """
        return lambda x: f(x + h) - f(x)

    @classmethod
    def forward2(
        cls, f: typing.Callable[[float], float], h: float
    ) -> typing.Callable[[float], float]:
        r"""
        Computes the second-order forward finite difference of a one-dimensional real-valued
        function ``f`` (:math:`f: \mathbb{R} \mapsto \mathbb{R}`) using step size ``h``.

        .. math::

            {\Delta}_{h}^{2}[f](x) = f(x + 2h) - 2f(x + h) + f(x)
        """
        return lambda x: f(x + 2 * h) - 2 * f(x + h) + f(x)

    @classmethod
    def forwardn(
        cls, f: typing.Callable[[float], float], h: float, n: int
    ) -> typing.Callable[[float], float]:
        r"""
        Computes the ``n``\th-order forward finite difference of a one-dimensional real-valued
        function ``f`` (:math:`f: \mathbb{R} \mapsto \mathbb{R}`) using step size ``h``.

        .. math::

            {\Delta}_{h}^{n}[f](x) = \sum_{i = 0}^{n} {(-1)}^{n - i} {{n}\choose{i}} f(x + ih)
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
        Computes the first-order backward finite difference of a one-dimensional real-valued
        function ``f`` (:math:`f: \mathbb{R} \mapsto \mathbb{R}`) using step size ``h``.

        .. math::

            {\nabla}_{h}[f](x) = f(x) - f(x - h)
        """
        return lambda x: f(x) - f(x - h)

    @classmethod
    def backward2(
        cls, f: typing.Callable[[float], float], h: float
    ) -> typing.Callable[[float], float]:
        r"""
        Computes the second-order backward finite difference of a one-dimensional real-valued
        function ``f`` (:math:`f: \mathbb{R} \mapsto \mathbb{R}`) using step size ``h``.

        .. math::

            {\nabla}_{h}^{2}[f](x) = f(x) - 2f(x - h) + f(x - 2h)
        """
        return lambda x: f(x) - 2 * f(x - h) + f(x - 2 * h)

    @classmethod
    def backwardn(
        cls, f: typing.Callable[[float], float], h: float, n: int
    ) -> typing.Callable[[float], float]:
        r"""
        Computes the ``n``\th-order backward finite difference of a one-dimensional real-valued
        function ``f`` (:math:`f: \mathbb{R} \mapsto \mathbb{R}`) using step size ``h``.

        .. math::

            {\nabla}_{h}^{n}[f](x) = \sum_{i = 0}^{n} {(-1)}^{i} {{n}\choose{i}} f(x - ih)
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
        Computes the first-order central finite difference of a one-dimensional real-valued
        function ``f`` (:math:`f: \mathbb{R} \mapsto \mathbb{R}`) using step size ``h``.

        .. math::

            {\delta}_{h}[f](x) = f(x + \frac{h}{2}) - f(x - \frac{h}{2})
        """
        return lambda x: f(x + h / 2) - f(x - h / 2)

    @classmethod
    def central2(
        cls, f: typing.Callable[[float], float], h: float
    ) -> typing.Callable[[float], float]:
        r"""
        Computes the second-order central finite difference of a one-dimensional real-valued
        function ``f`` (:math:`f: \mathbb{R} \mapsto \mathbb{R}`) using step size ``h``.

        .. math::

            {\delta}_{h}^{2}[f](x) = f(x + h) - 2f(x) + f(x - h)
        """
        return lambda x: f(x + h) - 2 * f(x) + f(x - h)

    @classmethod
    def centraln(
        cls, f: typing.Callable[[float], float], h: float, n: int
    ) -> typing.Callable[[float], float]:
        r"""
        Computes the ``n``\th-order central finite difference of a one-dimensional real-valued
        function ``f`` (:math:`f: \mathbb{R} \mapsto \mathbb{R}`) using step size ``h``.

        .. math::

            {\delta}_{h}^{n}[f](x) = \sum_{i = 0}^{n} {(-1)}^{i} {{n}\choose{i}} f(x + (\frac{n}{2} - i)h)
        """
        array = np.arange(0, n + 1)
        return lambda x: (
            (-1) ** array * scipy.special.comb(n, array) * f(x + (n / 2 - array) * h)
        ).sum()

    @classmethod
    def pforward(
        cls, f: typing.Callable[[typing.Sequence[float]], float], h: float, dim: int,
        *, ndim: typing.Optional[int] = None
    ) -> typing.Sequence[typing.Callable[[typing.Sequence[float]], float]]:
        r"""
        Computes the first-order partial forward finite differences of a ``dim``-dimensional
        real-valued function ``f`` (:math:`f: \mathbb{R}^{n} \mapsto \mathbb{R}`) using step size
        ``h``.

        .. math::

            {\Delta}_{h}{[f]}(\vec{x}) = \begin{bmatrix}
                {\Delta}_{h}{[f]}_{{x}_{1}}(\vec{x}) \\
                \vdots \\
                {\Delta}_{h}{[f]}_{{x}_{\dim{\vec{x}}}}(\vec{x}) \\
            \end{bmatrix}

        .. math::

            {\Delta}_{h}{[f]}_{{x}_{i}}(\vec{x}) = f(
                {x}_{1}, \dots, {x}_{i} + h, \dots, {x}_{\dim{\vec{x}}}
            ) - f(\vec{x})

        If ``ndim`` is not specified, returns a list of ``dim`` callable objects representing each of
        the ``dim``-dimensional real-valued functions obtained when computing the finite difference
        with respect to each of the ``dim`` dimensions in the domain of ``f``. If ``ndim`` is
        specified, returns a single callable object representing the ``dim``-dimensional real-valued
        function obtained when computing the finite difference with respect to the ``ndim``\th
        dimension of the domain of ``f``.
        """
        def partial(
            f_: typing.Callable[[typing.Sequence[float]], float], h_: float, ndim_: int
        ) -> typing.Callable[[typing.Sequence[float]], float]:
            r"""
            Computes the first-order partial forward finite difference of an :math:`n`-dimensional
            real-valued function (:math:`f: \mathbb{R}^{n} \mapsto \mathbb{R}`), ``f_``, using step
            size ``h_`` with respect to the ``ndim_``th dimension of the domain of :math:`f`.
            """
            return lambda x: f_([*x[:ndim_], x[ndim_] + h_, *x[(ndim_ + 1):]]) - f_(x)

        if ndim is not None:
            return partial(f, h, ndim)
        return [(partial(f, h, i)) for i in range(dim)]

    @classmethod
    def pforward2(
        cls, f: typing.Callable[[typing.Sequence[float]], float], h: float, dim: int,
        *, ndim: int = None
    ) -> typing.Sequence[typing.Callable[[typing.Sequence[float]], float]]:
        r"""
        Computes the second-order partial forward finite differences of a ``dim``-dimensional
        real-valued function ``f`` (:math:`f: \mathbb{R}^{n} \mapsto \mathbb{R}`) using step size
        ``h``.

        .. math::

            {\Delta}_{h}{[f]}(\vec{x}) = \begin{bmatrix}
                {\Delta}_{h}{[f]}_{{x}_{1}}(\vec{x}) \\
                \vdots \\
                {\Delta}_{h}{[f]}_{{x}_{\dim{\vec{x}}}}(\vec{x}) \\
            \end{bmatrix}

        .. math::

            {\Delta}_{h}{[f]}_{{x}_{i}}(\vec{x}) = f(
                {x}_{1}, \dots, {x}_{i} + 2h, \dots, {x}_{\dim{\vec{x}}}
            ) - 2f(
                {x}_{1}, \dots, {x}_{i} + h, \dots, {x}_{\dim{\vec{x}}}
            ) + f(\vec{x})

        If ``ndim`` is not specified, returns a list of ``dim`` callable objects representing each of
        the ``dim``-dimensional real-valued functions obtained when computing the finite difference
        with respect to each of the ``dim`` dimensions in the domain of ``f``. If ``ndim`` is
        specified, returns a single callable object representing the ``dim``-dimensional real-valued
        function obtained when computing the finite difference with respect to the ``ndim``\th
        dimension of the domain of ``f``.
        """
        def partial(
            f_: typing.Callable[[typing.Sequence[float]], float], h_: float, dim_: int
        ) -> typing.Callable[[typing.Sequence[float]], float]:
            r"""
            Computes the second-order partial forward finite difference of an :math:`n`-dimensional
            real-valued function (:math:`f: \mathbb{R}^{n} \mapsto \mathbb{R}`), ``f_``, using step
            size ``h_`` with respect to the ``ndim_``th dimension of the domain of :math:`f`.
            """
            return lambda x: f_(
                [*x[:dim_], x[dim_] + 2 * h_, *x[(dim_ + 1):]]
            ) - 2 * f_(
                [*x[:dim_], x[dim_] + h_, *x[(dim_ + 1):]]
            ) + f_(x)

        if ndim is not None:
            return partial(f, h, ndim)
        return [partial(f, h, i) for i in range(dim)]

    @classmethod
    def pforwardn(
        cls, f: typing.Callable[[float], float], h: float, n: int, dimensions: int,
        *, dim: int = None
    ) -> typing.Sequence[typing.Callable[[typing.Sequence[float]], float]]:
        r"""
        Computes the ``n``\th-order partial forward finite differences of a ``dim``-dimensional
        real-valued function ``f`` (:math:`f: \mathbb{R}^{n} \mapsto \mathbb{R}`) using step size
        ``h``.

        .. math::

            {\Delta}_{h}{[f]}(\vec{x}) = \begin{bmatrix}
                {\Delta}_{h}{[f]}_{{x}_{1}}(\vec{x}) \\
                \vdots \\
                {\Delta}_{h}{[f]}_{{x}_{\dim{\vec{x}}}}(\vec{x}) \\
            \end{bmatrix}

        .. math::

            {\Delta}_{h}{[f]}_{{x}_{i}}(\vec{x}) = \sum_{i = 0}^{n} {(-1)}^{n - i} {{n}\choose{i}} f(
                {x}_{1}, \dots, {x}_{i} + ih, \dots, {x}_{\dim{\vec{x}}}
            )

        If ``ndim`` is not specified, returns a list of ``dim`` callable objects representing each of
        the ``dim``-dimensional real-valued functions obtained when computing the finite difference
        with respect to each of the ``dim`` dimensions in the domain of ``f``. If ``ndim`` is
        specified, returns a single callable object representing the ``dim``-dimensional real-valued
        function obtained when computing the finite difference with respect to the ``ndim``\th
        dimension of the domain of ``f``.
        """
        def partial(
            f_: typing.Callable[[typing.Sequence[float]], float], h_: float, n_: int, dim_: int
        ) -> typing.Callable[[typing.Sequence[float]], float]:
            r"""
            Computes the ``n_``\th-order partial forward finite difference of an
            :math:`n`-dimensional real-valued function
            (:math:`f: \mathbb{R}^{n} \mapsto \mathbb{R}`), ``f_``, using step size ``h_`` with
            respect to the ``ndim_``th dimension of the domain of :math:`f`.
            """
            array = np.arange(0, n + 1)
            return lambda x: (
                (-1) ** (n_ - array) * scipy.special.comb(n_, array) * f_(
                    [*x[:dim_], x[dim_] + array * h_, *x[(dim_ + 1):]]
                )
            ).sum()

        if dim is not None:
            return partial(f, h, n, dim)
        return [partial(f, h, n, i) for i in range(dimensions)]

    @classmethod
    def pbackward(
        cls, f: typing.Callable[[typing.Sequence[float]], float], h: float, dim: int,
        *, ndim: typing.Optional[int] = None
    ) -> typing.Sequence[typing.Callable[[typing.Sequence[float]], float]]:
        r"""
        Computes the first-order partial backward finite differences of a ``dim``-dimensional
        real-valued function ``f`` (:math:`f: \mathbb{R}^{n} \mapsto \mathbb{R}`) using step size
        ``h``.

        .. math::

            {\nabla}_{h}{[f]}(\vec{x}) = \begin{bmatrix}
                {\nabla}_{h}{[f]}_{{x}_{1}}(\vec{x}) \\
                \vdots \\
                {\nabla}_{h}{[f]}_{{x}_{\dim{\vec{x}}}}(\vec{x}) \\
            \end{bmatrix}

        .. math::

            {\nabla}_{h}{[f]}_{{x}_{i}}(\vec{x}) = f(\vec{x}) - f(
                {x}_{1}, \dots, {x}_{i} - h, \dots, {x}_{\dim{\vec{x}}}
            )

        If ``ndim`` is not specified, returns a list of ``dim`` callable objects representing each of
        the ``dim``-dimensional real-valued functions obtained when computing the finite difference
        with respect to each of the ``dim`` dimensions in the domain of ``f``. If ``ndim`` is
        specified, returns a single callable object representing the ``dim``-dimensional real-valued
        function obtained when computing the finite difference with respect to the ``ndim``\th
        dimension of the domain of ``f``.
        """
        def partial(
            f_: typing.Callable[[typing.Sequence[float]], float], h_: float, ndim_: int
        ) -> typing.Callable[[typing.Sequence[float]], float]:
            r"""
            Computes the first-order partial forward finite difference of an :math:`n`-dimensional
            real-valued function (:math:`f: \mathbb{R}^{n} \mapsto \mathbb{R}`), ``f_``, using step
            size ``h_`` with respect to the ``ndim_``th dimension of the domain of :math:`f`.
            """
            return lambda x: f_(x) - f([*x[:ndim_], x[ndim_] - h_, *x[(ndim_ + 1):]])

        if ndim is not None:
            return partial(f, h, ndim)
        return [(partial(f, h, i)) for i in range(dim)]

    @classmethod
    def pbackward2(
        cls, f: typing.Callable[[typing.Sequence[float]], float], h: float, dim: int,
        *, ndim: int = None
    ) -> typing.Sequence[typing.Callable[[typing.Sequence[float]], float]]:
        r"""
        Computes the second-order partial backward finite differences of a ``dim``-dimensional
        real-valued function ``f`` (:math:`f: \mathbb{R}^{n} \mapsto \mathbb{R}`) using step size
        ``h``.

        .. math::

            {\nabla}_{h}{[f]}(\vec{x}) = \begin{bmatrix}
                {\nabla}_{h}{[f]}_{{x}_{1}}(\vec{x}) \\
                \vdots \\
                {\nabla}_{h}{[f]}_{{x}_{\dim{\vec{x}}}}(\vec{x}) \\
            \end{bmatrix}

        .. math::

            {\nabla}_{h}{[f]}_{{x}_{i}}(\vec{x}) = f(\vec{x}) - 2f(
                {x}_{1}, \dots, {x}_{i} - h, \dots, {x}_{\dim{\vec{x}}}
            ) + f(
                {x}_{1}, \dots, {x}_{i} - 2h, \dots, {x}_{\dim{\vec{x}}}
            )

        If ``ndim`` is not specified, returns a list of ``dim`` callable objects representing each of
        the ``dim``-dimensional real-valued functions obtained when computing the finite difference
        with respect to each of the ``dim`` dimensions in the domain of ``f``. If ``ndim`` is
        specified, returns a single callable object representing the ``dim``-dimensional real-valued
        function obtained when computing the finite difference with respect to the ``ndim``\th
        dimension of the domain of ``f``.
        """
        def partial(
            f_: typing.Callable[[typing.Sequence[float]], float], h_: float, dim_: int
        ) -> typing.Callable[[typing.Sequence[float]], float]:
            r"""
            Computes the second-order partial backward finite difference of an :math:`n`-dimensional
            real-valued function (:math:`f: \mathbb{R}^{n} \mapsto \mathbb{R}`), ``f_``, using step
            size ``h_`` with respect to the ``ndim_``th dimension of the domain of :math:`f`.
            """
            return lambda x: f_(x) - 2 * f_(
                [*x[:dim_], x[dim_] - h_, *x[(dim_ + 1):]]
            ) + f_(
                [*x[:dim_], x[dim_] - 2 * h_, *x[(dim_ + 1):]]
            )

        if ndim is not None:
            return partial(f, h, ndim)
        return [partial(f, h, i) for i in range(dim)]

    @classmethod
    def pbackwardn(
        cls, f: typing.Callable[[float], float], h: float, n: int, dimensions: int,
        *, dim: int = None
    ) -> typing.Sequence[typing.Callable[[typing.Sequence[float]], float]]:
        r"""
        Computes the ``n``\th-order partial forward finite differences of a ``dim``-dimensional
        real-valued function ``f`` (:math:`f: \mathbb{R}^{n} \mapsto \mathbb{R}`) using step size
        ``h``.

        .. math::

            {\nabla}_{h}{[f]}(\vec{x}) = \begin{bmatrix}
                {\nabla}_{h}{[f]}_{{x}_{1}}(\vec{x}) \\
                \vdots \\
                {\nabla}_{h}{[f]}_{{x}_{\dim{\vec{x}}}}(\vec{x}) \\
            \end{bmatrix}

        .. math::

            {\nabla}_{h}{[f]}_{{x}_{i}}(\vec{x}) = \sum_{i = 0}^{n} {(-1)}^{i} {{n}\choose{i}} f(
                {x}_{1}, \dots, {x}_{i} - ih, \dots, {x}_{\dim{\vec{x}}}
            )

        If ``ndim`` is not specified, returns a list of ``dim`` callable objects representing each of
        the ``dim``-dimensional real-valued functions obtained when computing the finite difference
        with respect to each of the ``dim`` dimensions in the domain of ``f``. If ``ndim`` is
        specified, returns a single callable object representing the ``dim``-dimensional real-valued
        function obtained when computing the finite difference with respect to the ``ndim``\th
        dimension of the domain of ``f``.
        """
        def partial(
            f_: typing.Callable[[typing.Sequence[float]], float], h_: float, n_: int, dim_: int
        ) -> typing.Callable[[typing.Sequence[float]], float]:
            """
            :param f_:
            :param h_:
            :param h_:
            :param dim_:
            :return:
            """
            array = np.arange(0, n + 1)
            return lambda x: (
                (-1) ** (n_ - array) * scipy.special.comb(n_, array) * f_(
                    [*x[:dim_], x[dim_] - array * h_, *x[(dim_ + 1):]]
                )
            ).sum()

        if dim is not None:
            return partial(f, h, n, dim)
        return [partial(f, h, n, i) for i in range(dimensions)]


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
                fdiff = FiniteDifference.forward(f, h)
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
                fdiff = FiniteDifference.forward2(f, h)
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
                fdiff = FiniteDifference.forwardn(f, h, n)
            except ValueError:
                fdiff = FiniteDifference.backwardn(f, h, n)

        return lambda x: fdiff(x) / (h ** n)
