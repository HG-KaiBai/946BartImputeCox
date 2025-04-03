#ifndef UPDATE_BH_H
#define UPDATE_BH_H

#include <vector>
#include <Rcpp.h>
#include "bart.h"

// Function to perform Metropolis-Hastings update for beta coefficients
// Output: A vector of updated beta coefficients
// [[Rcpp::export]]
std::vector<double> mh_beta(
    const std::vector<double>& current_beta, // Current beta coefficients
    const std::vector<double>& mean,         // Mean of the prior for beta
    const std::vector<double>& variances,    // Variances of the prior for beta
    const std::vector<double>& T,            // Observed time (event or censoring time)
    const std::vector<int>& C,               // Censoring indicator (1 for event, 0 for censored)
    const std::vector<std::vector<double>>& X, // Covariates (features) matrix
    const std::vector<double>& h0,          // Baseline hazard values
     const std::vector<double>& time,       // Piecewise time points for the baseline hazard
    double proposal_stddev                  // Standard deviation for the proposal distribution of beta
);

// Function to perform Metropolis-Hastings update for baseline hazard h0
// Output: A vector of updated h0 values
std::vector<double> mh_h0(
    const std::vector<double>& current_h0,  // Current baseline hazard values
    const std::vector<double>& h0_prior_shape, // Shape parameters for the prior on h0
    const std::vector<double>& h0_prior_rate,  // Rate parameters for the prior on h0
    std::vector<double>& T,                 // Observed time (event or censoring time)
    std::vector<int>& C,                    // Censoring indicator (1 for event, 0 for censored)
    const std::vector<std::vector<double>>& X, // Covariates (features) matrix (nxp)
    const std::vector<double>& beta,        // Coefficients for the Cox model
    const std::vector<double>& time,        // Piecewise time points for the baseline hazard
    double proposal_stddev_h0,              // Standard deviation for the proposal distribution of h0
    const std::vector<std::pair<bart, double>>& bart_models, // BART models for each covariate
    size_t missing_covariate_start,         // Index of the first missing covariate
    const std::vector<int>& covariate_types // Covariate types (1 for continuous, 2 for binary)
);

#endif // UPDATE_BH_H
