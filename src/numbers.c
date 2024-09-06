/**
 * Source file for "../includes/numbers.h".
 */

#include <stdlib.h>

#include "../include/numbers.h"


/**
 * Computes an exponential expression.
 * 
 * @param b The base of the exponent
 * @param p The power of the exponent
 * @return \f${b}^{p}\f$
 */
double power(double b, unsigned int p) {

    double res = 1.;
    for (int i = 0; i < p; ++i) {
        res *= b;
    }
    return res;

}


/**
 * Tail-recursive algorithm for factorial computation.
 * 
 * @param n A non-negative integer
 * @param res The cumulative product of the factorial computation
 * @return \f$res * (n - 1)!\f$
 */
unsigned long factorial_(unsigned int n, unsigned long res) {

    if (n == 0 || n == 1) {
        return res;
    }

    return factorial_(n - 1, n * res);

}


/**
 * Returns the factorial of a non-negative integer.
 * 
 * @param n A non-negative integer
 * @return \f$n!\f$
 */
unsigned long factorial(unsigned int n) { return factorial_(n, 1); }


/**
 * Computes the number of unique, non-repeating selections of size `r` from a set of items of size
 * `n`, in which the order of the selected items is insignificant.
 * 
 * @param n The number of items available to select
 * @param r The number of items to select
 * @return \f$C(n, r)\f$
 */
unsigned int comb(unsigned int n, unsigned int r) {
    
    if (r > n) {
        return 0;
    } else if (r == 0 || r == n) {
        return 1;
    } else if (r == 1 || r == n - 1) {
        return n;
    }

    return (unsigned int)(factorial(n) / (factorial(r) * factorial(n - r)));
    
}


/**
 * Computes the number of unique, non-repeating selections of size `r` from a set of items of size
 * `n`, in which the order of the selected items is significant.
 * 
 * @param n The number of items available to select
 * @param r The number of items to select
 * @return \f$P(n, r)\f$
 */
unsigned int perm(unsigned int n, unsigned int r) {

    if (r > n) {
        return 0;
    } else if (r == 0) {
        return 1;
    } else if (r == 1) {
        return n;
    }
    
    return (unsigned int)(factorial(n) / factorial(n - r));
    
}


/**
 * Computes coefficients in the polynomial expansion of the binomial power \f${(1 + x)}^{n}\f$.
 * 
 * @param n The power of the polynomial expansion.
 * @return \f$\binom{n}{k}\f$ for \f$0 \le k \le n\f$
 */
unsigned int *binom(unsigned int n) {

    if (n == 0) { return 0; }

    int *res = (int *)calloc(n + 1, sizeof(int));
    for (int k = 0; k < n / 2 + 1; ++k) {
        *(res + k) = *(res + (n - k)) = comb(n, k);
    }

    return res;

}
