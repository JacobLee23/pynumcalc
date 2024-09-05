/**
 * Type definitions for mathematical entities.
 */


/**
 * Represents a real-valued function of `dim` dimensions denoted by
 * \f$f: {\mathbb{R}}^{dim} \to \mathbb{R}\f$.
 */
typedef double (*RealFunction)(double *x, float dim);
