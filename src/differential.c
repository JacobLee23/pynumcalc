/**
 * Source file for "../include/differential.h"
 */

#include <stdlib.h>

#include "../include/differential.h"
#include "../include/numbers.h"
#include "../include/types.h"


/**
 * Duplicates a domain element to a dynamically allocated array.
 * 
 * @param x The domain element to duplicate
 * @param dim The dimension of `x`
 * @return A pointer to the duplicated `double` array
 */
double *duplicatex_(double *x, unsigned int dim) {

    double *x_ = (double *)calloc(dim, sizeof(double));
    for (int i = 0; i < dim; ++i) {
        *(x_ + i) = *(x + i);
    }

    return x_;

}


/**
 * Computes the first-order forward finite differences of `f` at `x` using step size `h`. `f` is a
 * real-valued function of `dim` dimensions defined as $f: {\mathbb{R}}^{dim} \to \mathbb{R}$.
 * 
 * For a one-dimensional function, the first-order forward finite difference is defined as:
 * 
 * \f[
 *      {\Delta}_{h} [f](x) = f(x + h) - f(x)
 * \f]
 * 
 * Generalizing this to all real-valued functions, given
 * 
 * \f[
 *      \vec{x} = \langle {x}_{1}, \dots, {x}_{i}, \dots, {x}_{dim} \rangle
 * \f]
 * 
 * the first-order forward finite difference with respect to \f${x}_{i}\f$ is defined as:
 * 
 * \f[
 *      {\Delta}_{h} {[f]}_{{x}_{i}}(\vec{x}) = f(
 *          \langle {x}_{i}, \dots, {x}_{i} + h, \dots, {x}_{dim} \rangle
 *      ) - f(\vec{x})
 * \f]
 * 
 * The computed finite differences are stored in a dynamically allocated `double` array of size
 * `dim`. Index `i` for $0 <= i < dim$ corresponds to the finite difference of `f` computed with
 * respect to ${x}_{i + 1}$.
 * 
 * @param f A real-valued function
 * @param x The domain element of `f` at which to compuate the finite difference
 * @param h The step size of the finite difference
 * @param dim The dimension of the domain of `f`
 * @return A pointer to the computed finite differences stored as a `double` array of size `dim`
 */
double *forward_first(RealFunction f, double *x, double h, unsigned int dim) {

    double *x1 = duplicatex_(x, dim);
    double *finite_differences = (double *)calloc(dim, sizeof(double));

    double a, b;
    for (int i = 0; i < dim; ++i) {

        *(x1 + i) += h;
        *(finite_differences + i) = f(x1, dim) - f(x, dim);
        *(x1 + i) -= h;

    }

    free(x1);

    return finite_differences;

}


/**
 * Computes the second-order forward finite differences of `f` at `x` using step size `h`. `f` is a
 * real-valued function of `dim` dimensions defined as $f: {\mathbb{R}}^{dim} \to \mathbb{R}$.
 * 
 * For a one-dimensional function, the second-order forward finite difference is defined as:
 * 
 * \f[
 *      {\Delta}_{h}^{2} [f](x) = f(x + 2h) - 2f(x + h) + f(x)
 * \f]
 * 
 * Generalizing this to all real-valued functions, given
 * 
 * \f[
 *      \vec{x} = \langle {x}_{1}, \dots, {x}_{i}, \dots, {x}_{dim} \rangle
 * \f]
 * 
 * the second-order forward finite difference with respect to \f${x}_{i}\f$ is defined as:
 * 
 * \f[
 *      {\Delta}_{h}^{2} {[f]}_{{x}_{i}}(\vec{x}) = f(
 *          \langle {x}_{i}, \dots, {x}_{i} + 2h, \dots, {x}_{dim} \rangle
 *      ) - 2f(
 *          \langle {x}_{i}, \dots, {x}_{i} + h, \dots, {x}_{dim} \rangle
 *      ) + f(\vec{x})
 * \f]
 * 
 * The computed finite differences are stored in a dynamically allocated `double` array of size
 * `dim`. Index `i` for $0 <= i < dim$ corresponds to the finite difference of `f` computed with
 * respect to ${x}_{i + 1}$.
 * 
 * @param f A real-valued function
 * @param x The domain element of `f` at which to compuate the finite difference
 * @param h The step size of the finite difference
 * @param dim The dimension of the domain of `f`
 * @return A pointer to the computed finite differences stored as a `double` array of size `dim`
 */
double *forward_second(RealFunction f, double *x, double h, unsigned int dim) {

    double *x1 = duplicatex_(x, dim);
    double *x2 = duplicatex_(x, dim);
    double *finite_differences = (double *)calloc(dim, sizeof(double));

    for (int i = 0; i < dim; ++i) {

        *(x1 + i) += 2 * h, *(x2 + i) += h;
        *(finite_differences + i) = f(x1, dim) - 2 * f(x2, dim) + f(x, dim);
        *(x1 + i) -= 2 * h, *(x2 + i) -= h;

    }

    free(x1);
    free(x2);

    return finite_differences;

}


/**
 * Computes the `n`th-order forward finite differences of `f` at `x` using step size `h`. `f` is a
 * real-valued function of `dim` dimensions defined as $f: {\mathbb{R}}^{dim} \to \mathbb{R}$.
 * 
 * For a one-dimensional function, the `n`th-order forward finite difference is defined as:
 * 
 * \f[
 *      {\Delta}_{h}^{n} [f](x) = \sum_{k=0}^{n} {(-1)}^{n - k} {{n}\choose{k}} f(x + kh)
 * \f]
 * 
 * Generalizing this to all real-valued functions, given
 * 
 * \f[
 *      \vec{x} = \langle {x}_{1}, \dots, {x}_{i}, \dots, {x}_{dim} \rangle
 * \f]
 * 
 * the `n`th-order forward finite difference with respect to \f${x}_{i}\f$ is defined as:
 * 
 * \f[
 *      {\Delta}_{h}^{n} {[f]}_{{x}_{i}}(\vec{x}) = \sum_{k=0}^{n} {(-1)}^{n - k} {{n}\choose{k}} f(
 *          \langle {x}_{1}, \dots, {x}_{i} + kh, \dots, {x}_{dim} \rangle
 *      )
 * \f]
 * 
 * The computed finite differences are stored in a dynamically allocated `double` array of size
 * `dim`. Index `i` for $0 <= i < dim$ corresponds to the finite difference of `f` computed with
 * respect to ${x}_{i + 1}$.
 * 
 * @param f A real-valued function
 * @param x The domain element of `f` at which to compuate the finite difference
 * @param h The step size of the finite difference
 * @param n The order of the finite difference
 * @param dim The dimension of the domain of `f`
 * @return A pointer to the computed finite differences stored as a `double` array of size `dim`
 */
double *forward_nth(RealFunction f, double *x, double h, unsigned int n, unsigned int dim) {

    double *x1 = duplicatex_(x, dim);
    double *finite_differences = (double *)calloc(dim, sizeof(double));

    for (int i = 0; i < dim; ++i) {

        *(finite_differences + i) = 0.;

        for (int k = 0; k <= n; ++k) {

            *(x1 + i) += k * h;
            *(finite_differences + i) += (n - k % 2 == 0 ? 1 : -1) * comb(n, k) * f(x1, dim);
            *(x1 + i) -= k * h;

        }

    }

    free(x1);

    return finite_differences;

}


/**
 * Computes the first-order backward finite differences of `f` at `x` using step size `h`. `f` is a
 * real-valued function of `dim` dimensions defined as $f: {\mathbb{R}}^{dim} \to \mathbb{R}$.
 * 
 * For a one-dimensional function, the first-order backward finite difference is defined as:
 * 
 * \f[
 *      {\nabla}_{h} [f](x) = f(x) - f(x - h)
 * \f]
 * 
 * Generalizing this to all real-valued functions, given
 * 
 * \f[
 *      \vec{x} = \langle {x}_{1}, \dots, {x}_{i}, \dots, {x}_{dim} \rangle
 * \f]
 * 
 * the first-order backward finite difference with respect to \f${x}_{i}\f$ is defined as:
 * 
 * \f[
 *      {\nabla}_{h} {[f]}_{{x}_{i}}(\vec{x}) = f(\vec{x}) - f(
 *          \langle {x}_{i}, \dots, {x}_{i} - h, \dots, {x}_{dim} \rangle
 *      )
 * \f]
 * 
 * The computed finite differences are stored in a dynamically allocated `double` array of size
 * `dim`. Index `i` for $0 <= i < dim$ corresponds to the finite difference of `f` computed with
 * respect to ${x}_{i + 1}$.
 * 
 * @param f A real-valued function
 * @param x The domain element of `f` at which to compuate the finite difference
 * @param h The step size of the finite difference
 * @param dim The dimension of the domain of `f`
 * @return A pointer to the computed finite differences stored as a `double` array of size `dim`
 */
double *backward_first(RealFunction f, double *x, double h, unsigned int dim) {

    double *x1 = duplicatex_(x, dim);
    double *finite_differences = (double *)calloc(dim, sizeof(double));

    for (int i = 0; i < dim; ++i) {

        *(x1 + i) -= h;
        *(finite_differences + i) = f(x, dim) - f(x1, dim);
        *(x1 + i) += h;

    }

    free(x1);

    return finite_differences;

}


/**
 * Computes the second-order backward finite differences of `f` at `x` using step size `h`. `f` is a
 * real-valued function of `dim` dimensions defined as $f: {\mathbb{R}}^{dim} \to \mathbb{R}$.
 * 
 * For a one-dimensional function, the second-order backward finite difference is defined as:
 * 
 * \f[
 *      {\nabla}_{h}^{2} [f](x) = f(x) - 2f(x - h) + f(x - 2h)
 * \f]
 * 
 * Generalizing this to all real-valued functions, given
 * 
 * \f[
 *      \vec{x} = \langle {x}_{1}, \dots, {x}_{i}, \dots, {x}_{dim} \rangle
 * \f]
 * 
 * the second-order backward finite difference with respect to \f${x}_{i}\f$ is defined as:
 * 
 * \f[
 *      {\nabla}_{h}^{2} {[f]}_{{x}_{i}}(\vec{x}) = f(\vec{x}) - 2f(
 *          \langle {x}_{i}, \dots, {x}_{i} - h, \dots, {x}_{dim} \rangle
 *      ) + f(
 *          \langle {x}_{i}, \dots, {x}_{i} - 2h, \dots, {x}_{dim} \rangle
 *      ) + f(\vec{x})
 * \f]
 * 
 * The computed finite differences are stored in a dynamically allocated `double` array of size
 * `dim`. Index `i` for $0 <= i < dim$ corresponds to the finite difference of `f` computed with
 * respect to ${x}_{i + 1}$.
 * 
 * @param f A real-valued function
 * @param x The domain element of `f` at which to compuate the finite difference
 * @param h The step size of the finite difference
 * @param dim The dimension of the domain of `f`
 * @return A pointer to the computed finite differences stored as a `double` array of size `dim`
 */
double *backward_second(RealFunction f, double *x, double h, unsigned int dim) {

    double *x1 = duplicatex_(x, dim);
    double *x2 = duplicatex_(x, dim);
    double *finite_differences = (double *)calloc(dim, sizeof(double));

    for (int i = 0; i < dim; ++i) {

        *(x1 + i) -= h, *(x2 + i) -= 2 * h;
        *(finite_differences + i) = f(x, dim) - 2 * f(x1, dim) + f(x2, dim);
        *(x1 + i) += h, *(x2 + i) += 2 * h;

    }

    free(x1);
    free(x2);

    return finite_differences;

}


/**
 * Computes the `n`th-order backward finite differences of `f` at `x` using step size `h`. `f` is a
 * real-valued function of `dim` dimensions defined as $f: {\mathbb{R}}^{dim} \to \mathbb{R}$.
 * 
 * For a one-dimensional function, the `n`th-order backward finite difference is defined as:
 * 
 * \f[
 *      {\nabla}_{h}^{n} [f](x) = \sum_{k=0}^{n} {(-1)}^{k} {{n}\choose{k}} f(x - kh)
 * \f]
 * 
 * Generalizing this to all real-valued functions, given
 * 
 * \f[
 *      \vec{x} = \langle {x}_{1}, \dots, {x}_{i}, \dots, {x}_{dim} \rangle
 * \f]
 * 
 * the `n`th-order backward finite difference with respect to \f${x}_{i}\f$ is defined as:
 * 
 * \f[
 *      {\nabla}_{h}^{n} {[f]}_{{x}_{i}}(\vec{x}) = \sum_{k=0}^{n} {(-1)}^{k} {{n}\choose{k}} f(
 *          \langle {x}_{1}, \dots, {x}_{i} - kh, \dots, {x}_{dim} \rangle
 *      )
 * \f]
 * 
 * The computed finite differences are stored in a dynamically allocated `double` array of size
 * `dim`. Index `i` for $0 <= i < dim$ corresponds to the finite difference of `f` computed with
 * respect to ${x}_{i + 1}$.
 * 
 * @param f A real-valued function
 * @param x The domain element of `f` at which to compuate the finite difference
 * @param h The step size of the finite difference
 * @param n The order of the finite difference
 * @param dim The dimension of the domain of `f`
 * @return A pointer to the computed finite differences stored as a `double` array of size `dim`
 */
double *backward_nth(RealFunction f, double *x, double h, unsigned int n, unsigned int dim) {

    double *x1 = duplicatex_(x, dim);
    double *finite_differences = (double *)calloc(dim, sizeof(double));

    for (int i = 0; i < dim; ++i) {

        *(finite_differences + i) = 0.;

        for (int i = 0; i <= n; ++i) {

            *(x1 + i) -= i * h;
            *(finite_differences + i) += (i % 2 == 0 ? 1 : -1) * comb(n, i) * f(x1, dim);
            *(x1 + i) += i * h;

        }

    }

    free(x1);

    return finite_differences;

}


/**
 * Computes the first-order central finite differences of `f` at `x` using step size `h`. `f` is a
 * real-valued function of `dim` dimensions defined as $f: {\mathbb{R}}^{dim} \to \mathbb{R}$.
 * 
 * For a one-dimensional function, the first-order central finite difference is defined as:
 * 
 * \f[
 *      {\delta}_{h} [f](x) = f(x + \frac{h}{2}) - f(x - \frac{h}{2})
 * \f]
 * 
 * Generalizing this to all real-valued functions, given
 * 
 * \f[
 *      \vec{x} = \langle {x}_{1}, \dots, {x}_{i}, \dots, {x}_{dim} \rangle
 * \f]
 * 
 * the first-order central finite difference with respect to \f${x}_{i}\f$ is defined as:
 * 
 * \f[
 *      {\delta}_{h} {[f]}_{{x}_{i}}(\vec{x}) = f(
 *          \langle {x}_{i}, \dots, {x}_{i} + \frac{h}{2}, \dots, {x}_{dim} \rangle
 *      ) - f(
 *          \langle {x}_{i}, \dots, {x}_{i} - \frac{h}{2}, \dots, {x}_{dim} \rangle
 *      )
 * \f]
 * 
 * The computed finite differences are stored in a dynamically allocated `double` array of size
 * `dim`. Index `i` for $0 <= i < dim$ corresponds to the finite difference of `f` computed with
 * respect to ${x}_{i + 1}$.
 * 
 * @param f A real-valued function
 * @param x The domain element of `f` at which to compuate the finite difference
 * @param h The step size of the finite difference
 * @param dim The dimension of the domain of `f`
 * @return A pointer to the computed finite differences stored as a `double` array of size `dim`
 */
double *central_first(RealFunction f, double *x, double h, unsigned int dim) {

    double *x1 = duplicatex_(x, dim);
    double *x2 = duplicatex_(x, dim);
    double *finite_differences = (double *)calloc(dim, sizeof(double));

    for (int i = 0; i < dim; ++i) {

        *(x1 + i) += h / 2, *(x2 + i) -= h / 2;
        *(finite_differences + i) = f(x1, dim) - f(x2, dim);
        *(x1 + i) -= h / 2, *(x2 + i) += h / 2;

    }

    free(x1);
    free(x2);

    return finite_differences;

}


/**
 * Computes the second-order central finite differences of `f` at `x` using step size `h`. `f` is a
 * real-valued function of `dim` dimensions defined as $f: {\mathbb{R}}^{dim} \to \mathbb{R}$.
 * 
 * For a one-dimensional function, the second-order central finite difference is defined as:
 * 
 * \f[
 *      {\delta}_{h}^{2} [f](x) = f(x + h) - 2f(x) + f(x - h)
 * \f]
 * 
 * Generalizing this to all real-valued functions, given
 * 
 * \f[
 *      \vec{x} = \langle {x}_{1}, \dots, {x}_{i}, \dots, {x}_{dim} \rangle
 * \f]
 * 
 * the second-order central finite difference with respect to \f${x}_{i}\f$ is defined as:
 * 
 * \f[
 *      {\delta}_{h}^{2} {[f]}_{{x}_{i}}(\vec{x}) = f(
 *          \langle {x}_{i}, \dots, {x}_{i} + h, \dots, {x}_{dim} \rangle
 *      ) - 2f(\vec{x}) + f(
 *          \langle {x}_{i}, \dots, {x}_{i} - h, \dots, {x}_{dim} \rangle
 *      )
 * \f]
 * 
 * The computed finite differences are stored in a dynamically allocated `double` array of size
 * `dim`. Index `i` for $0 <= i < dim$ corresponds to the finite difference of `f` computed with
 * respect to ${x}_{i + 1}$.
 * 
 * @param f A real-valued function
 * @param x The domain element of `f` at which to compuate the finite difference
 * @param h The step size of the finite difference
 * @param dim The dimension of the domain of `f`
 * @return A pointer to the computed finite differences stored as a `double` array of size `dim`
 */
double *central_second(RealFunction f, double *x, double h, unsigned int dim) {

    double *x1 = duplicatex_(x, dim);
    double *x2 = duplicatex_(x, dim);
    double *finite_differences = (double *)calloc(dim, sizeof(double));

    for (int i = 0; i < dim; ++i) {

        *(x1 + i) += h, *(x2 + i) -= h;
        *(finite_differences + i) = f(x1, dim) - 2 * f(x, dim) + f(x2, dim);
        *(x1 + i) -= h, *(x2 + i) += h;

    }

    free(x1);
    free(x2);

    return finite_differences;

}


/**
 * Computes the `n`th-order central finite differences of `f` at `x` using step size `h`. `f` is a
 * real-valued function of `dim` dimensions defined as $f: {\mathbb{R}}^{dim} \to \mathbb{R}$.
 * 
 * For a one-dimensional function, the `n`th-order central finite difference is defined as:
 * 
 * \f[
 *      {\delta}_{h}^{n} [f](x) = \sum_{k=0}^{n} {(-1)}^{k} {{n}\choose{k}} f(x + (\frac{n}{2} - k)h)
 * \f]
 * 
 * Generalizing this to all real-valued functions, given
 * 
 * \f[
 *      \vec{x} = \langle {x}_{1}, \dots, {x}_{i}, \dots, {x}_{dim} \rangle
 * \f]
 * 
 * the `n`th-order central finite difference with respect to \f${x}_{i}\f$ is defined as:
 * 
 * \f[
 *      {\delta}_{h}^{n} {[f]}_{{x}_{i}}(\vec{x}) = \sum_{k=0}^{n} {(-1)}^{k} {{n}\choose{k}} f(
 *          \langle {x}_{1}, \dots, {x}_{i} + (\frac{n}{2} - k)h, \dots, {x}_{dim} \rangle
 *      )
 * \f]
 * 
 * The computed finite differences are stored in a dynamically allocated `double` array of size
 * `dim`. Index `i` for $0 <= i < dim$ corresponds to the finite difference of `f` computed with
 * respect to ${x}_{i + 1}$.
 * 
 * @param f A real-valued function
 * @param x The domain element of `f` at which to compuate the finite difference
 * @param h The step size of the finite difference
 * @param n The order of the finite difference
 * @param dim The dimension of the domain of `f`
 * @return A pointer to the computed finite differences stored as a `double` array of size `dim`
 */
double *central_nth(RealFunction f, double *x, double h, unsigned int n, unsigned int dim) {

    double *x1 = duplicatex_(x, dim);
    double *finite_differences = (double *)calloc(dim, sizeof(double));

    for (int i = 0; i < dim; ++i) {

        *(finite_differences + i) = 0.;

        for (int i = 0; i <= n; ++i) {

            *(x1 + i) += (n / 2 - i) * h;
            *(finite_differences + i) += (i % 2 == 0 ? 1 : -1) * comb(n, i) * f(x1, dim);
            *(x1 + i) -= (n / 2 - i) * h;

        }

    }

    free(x1);

    return finite_differences;

}


/**
 * Computes the `n`th-order difference quotients for a real-valued function `f` at `x` using step
 * size `h`. `f` is a real-valued function function of `dim` dimensions defined as
 * \f$f: {\mathbb{R}}^{dim} \to \mathbb{R}\f$. `findiff` contains functions for computing first-,
 * second-, and `n`th-order finite differences.
 * 
 * For a one-dimensional function, the first-order difference quotient is defined as:
 * 
 * \f[
 * 
 *      \frac{F(x, h)}{h}
 * 
 * \f]
 * 
 * where \f$F: \mathbb{R} \times \mathbb{R} \to \mathbb{R}\f$ defines an appropriate finite
 * difference function.
 * 
 * Generalizing this to all real-valued functions, given
 * 
 * \f[
 *      \vec{x} = \langle {x}_{1}, \dots, {x}_{i}, \dots, {x}_{dim} \rangle
 * \f]
 * 
 * the `n`th-order central finite difference with respect to \f${x}_{i}\f$ is defined as:
 * 
 * \f[
 * 
 *      \frac{F(\vec{x}, h)}{{h}^{n}}
 * 
 * \f]
 * 
 * where \f$F: {\mathbb{R}}^{dim} \times {\mathbb{R}} \to \mathbb{R}\f$ defines an appropriate
 * finite difference function.
 * 
 * The computed difference quotients are stored in a dynamically allocated `double` array of size
 * `dim`. Index `i` for $0 <= i < dim$ corresponds to the difference quotients of `f` computed with
 * respect to ${x}_{i + 1}$.
 * 
 * @param f A real-valued function
 * @param x The domain element of `f` at which to compute the difference quotient
 * @param h The step size of the difference quotient
 * @param n The order of the difference quotient
 * @param dim The dimension of the domain of `f`
 * @param findiff The finite difference method to use
 * @return A pointer to the computed difference quotients stored as a `double` array of size `dim`
 */
double *difference_quotient(
    RealFunction f, double *x, double h, unsigned int n, unsigned int dim,
    struct FiniteDifference *findiff
) {

    double *finite_differences;
    switch (n) {
        case 1:
            finite_differences = findiff->first(f, x, h, dim);
            break;
        case 2:
            finite_differences = findiff->second(f, x, h, dim);
            break;
        default:
            finite_differences = findiff->nth(f, x, h, dim, n);
            break;
    }

    double step = power(h, n);
    for (int i = 0; i < dim; ++i) {
        *(finite_differences + i) /= step;
    }

    return finite_differences;

}
