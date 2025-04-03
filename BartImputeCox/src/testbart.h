#ifndef TESTBART_H
#define TESTBART_H

#include <vector>
#include <Rcpp.h>

// This file is a test file to test "fitbatr.cpp" by model/update the BART, do the prediction and return the result.


// Function to initialize BART and predict
// Output: A random sample from the normal distribution with mean = predicted value and sd = sigma by BART
// [[Rcpp::export]]
double initialize_and_predict(
    int type,                     // type of the response covariate. 1: continuous, 2: binary
    const std::vector<std::vector<double>>& x_train, // A matrix of traning covariates X (nxp)
    const std::vector<double>& H0,      // cumulated hazard for each individual (nx1)
    const std::vector<double>& y_train, // response variable (nx1)
    const std::vector<double>& x_test, // A vector of test covariates (1xp)
    double H0_test, // cumulated hazard for the individual (1x1)
    size_t m, // number of trees
    int numcut, // number of cut points for split the tree
    double mybeta, // power = 2 for the probability of split a node
    double alpha, // base = 0.95 for the probability of split a node. (p = alpha * (1 + d)^mybeta)
    double tau, // standard deviation for sample value in the terminal nodes.
    double nu, // Degrees of freedom for sigma prior (continuous case)
    double lambda // Scale for sigma prior (continuous case)
);

// Function to initialize, update BART, and predict
// Output: A random sample from the normal distribution with mean = predicted value and sd = sigma by BART
// [[Rcpp::export]]
double initialize_update_and_predict(
    int type,                    // type of the response covariate. 1: continuous, 2: binary
    const std::vector<std::vector<double>>& x_train,  // A matrix of traning covariates X (nxp)
    const std::vector<double>& H0,     // cumulated hazard for each individual (nx1)
    const std::vector<double>& y_train, // response variable (nx1)
    const std::vector<double>& x_test, // A vector of test covariates (1xp)
    double H0_test, // cumulated hazard for the individual (1x1)
    size_t m, // number of trees
    int numcut, // number of cut points for split the tree
    double mybeta, // power = 2 for the probability of split a node
    double alpha,  // base = 0.95 for the probability of split a node. (p = alpha * (1 + d)^mybeta)
    double tau, // standard deviation for sample value in the terminal nodes.
    double nu, // Degrees of freedom for sigma prior (continuous case)
    double lambda // Scale for sigma prior (continuous case)
);

#endif // IMPUTATION_H
