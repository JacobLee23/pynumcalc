/**
 * Auxiliary mathematical computation.
 */


/**
 * Computes \f${b}^{p}\f$.
 */
double power(double b, unsigned int p);

/**
 * Computes \f$n!\f$.
 */
unsigned long factorial(unsigned int n);

/**
 * Computes \f$C(n, r)\f$.
 */
unsigned int comb(unsigned int n, unsigned int r);

/**
 * Computes \f$P(n, r)\f$.
 */
unsigned int perm(unsigned int n, unsigned int r);

/**
 * Computes \f$\binom{n}{k}\f$ for \f$0 \le k \le n\f$.
 */
unsigned int *binom(unsigned int n);
