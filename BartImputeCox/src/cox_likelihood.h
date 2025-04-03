#ifndef COX_LIKELIHOOD_H
#define COX_LIKELIHOOD_H

#include <vector>
#include <Rcpp.h>

// Function to compute the cumulative baseline hazard
/*  Input:   h0: The baseline hazard values for each time interval.
            T: The event times for each individual.
            time: The time points at which the baseline hazard is defined.
    Output: H0: The cumulative baseline hazard for each individual.
            h0_for_T: The baseline hazard value for the event time of each individual.
*/
std::pair<std::vector<double>, std::vector<double>> compute_cumulative_baseline_hazard(
    const std::vector<double>& h0, const std::vector<double>& T,
    const std::vector<double>& time);

// Function to compute the individual cox likelihood
/*  Input:  C: The censoring indicator (1 for event, 0 for censored).
            X: The covariate values for the individual.
            beta: The coefficients for the Cox model.
            h0: The baseline hazard value for the event time of the individual.
            H0: The cumulative baseline hazard for the individual.
    Output: The log-likelihood for the individual.
*/
double compute_individual_likelihood(
    int C, const std::vector<double>& X,
    const std::vector<double>& beta, double h0, double H0);

// Function to compute the Cox likelihood
/*  Input:  T: The vector of event times.
            C: The vector of censoring indicators.
            X: The vector of covariate matrix.
            beta: The coefficients for the Cox model.
            h0: The baseline hazard values for each time interval.
            time: The time points at which the baseline hazard is defined.
    Output: The total log-likelihood for the Cox model.
*/
// [[Rcpp::export]]
double compute_cox_likelihood(
    const std::vector<double>& T, const std::vector<int>& C,
    const std::vector<std::vector<double>>& X, const std::vector<double>& beta,
    const std::vector<double>& h0, const std::vector<double>& time);

#endif // COX_LIKELIHOOD_H
