#include <cmath>
#include <iostream>
#include "prior_likelihood.h" // See documation for details on the functions in this file.


double  GammaLikelihood(double x, double shape, double rate) {
    if (x <= 0) {
        throw std::invalid_argument("Random variable x must be positive.");
    }
    if (shape <= 0 || rate <= 0) {
        throw std::invalid_argument("Shape and rate parameters must be positive.");
    }

    // Compute the log-likelihood for the Gamma distribution
    double logLikelihood = (shape - 1) * std::log(x) - x * rate;

    return logLikelihood;
}

double NormalLikelihood(double x, double mean, double stddev) {
    if (stddev <= 0) {
        throw std::invalid_argument("Standard deviation must be positive.");
    }
    double variance = stddev * stddev;
    double logLikelihood = - ((x - mean) * (x - mean)) / (2 * variance);
    return logLikelihood;
}




