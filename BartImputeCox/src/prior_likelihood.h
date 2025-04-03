#ifndef PRIOR_LIKELIHOOD_H
#define PRIOR_LIKELIHOOD_H

#include <Rcpp.h>

// Constant coefficient are removed for the loglikelihood.


//  Function to compute the log-likelihood of a Gamma distribution
/*  Input:  x: The random variable (must be positive).
            shape: The shape parameter of the Gamma distribution (must be positive).
            rate: The rate parameter of the Gamma distribution (must be positive).
    Output: The log-likelihood of the Gamma distribution for the given x.
*/
// [[Rcpp::export]]
double GammaLikelihood(double x, double shape, double rate);

// Function to compute the log-likelihood of a normal distribution
/*  Input:  x: The random variable.
            mean: The mean of the log-normal distribution.
            stddev: The standard deviation of the normal distribution (must be positive).
    Output: The log-likelihood of the normal distribution for the given x.
*/
// [[Rcpp::export]]
double NormalLikelihood(double x, double mean, double stddev);

#endif // PRIOR_LIKELIHOOD_H
