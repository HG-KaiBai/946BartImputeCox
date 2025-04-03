#ifndef FITBART_H
#define FITBART_H

#include <vector>
#include <utility>
#include "bart.h"

// Function to initialize a BART model
// Output: A pair containing the BART model and the standard deviation(sigma) value
std::pair<bart, double> initialize_bart(
    int type,                          // type of the response covariate. 1: continuous, 2: binary
    const std::vector<std::vector<double>>& x_train, // A matrix of traning covariates X (nxp)
    const std::vector<double>& H0,     // cumulated hazard for each individual (nx1)
    const std::vector<double>& y_train,// response variable (nx1)
    size_t m,                          // number of trees
    int numcut,                        // number of cut points for split the tree
    double mybeta,                     // power = 2 for the probability of split a node
    double alpha,                      // base = 0.95 for the probability of split a node. (p = alpha * (1 + d)^mybeta)
    double tau,                        // standard deviation for sample value in the terminal nodes.
    // tau = (max - min) / (2 * k * sqrt(m)) as default
    double nu,                         // Degrees of freedom for sigma prior (continuous case)
    double lambda                      // Scale for sigma prior (continuous case)
);

// Function to update a BART model
// Output: A pair containing the updated BART model and the updated standard deviation(sigma) value
std::pair<bart, double> update_bart(
    const bart& bm,                    // Current BART model to be updated
    double& sigma,                     // Current sigma value to be updated
    const std::vector<std::vector<double>>& x_train, // A matrix of updated traning covariates X (nxp)
    const std::vector<double>& H0,     // Updated cumulated hazard for each individual (nx1)
    const std::vector<double>& y_train,// Updated response variable (nx1)
    int numcut,                        // number of cut points
    int type,                          // type of the response covariate. 1: continuous, 2: binary
    double nu,                         // Degrees of freedom for sigma prior (continuous case)
    double lambda                      // Scale for sigma prior (continuous case)
);

// Function to predict using a BART model
// Output: The predicted value for the given input x_test and H0
double predict_bart(
    bart& bm,                          // BART model for prediction
    const std::vector<double>& x_test, // A vector of test covariates (1xp)
    double H0                          // cumulated hazard for the individual (1x1)     
);



#endif // FITBART_H
