#include <vector>
#include <random>
#include <iostream>
#include <cmath>
#include <Rcpp.h>
#include "cox_likelihood.h"
#include "update_bh.h"
#include "prior_likelihood.h"
#include "fitbart.h"
#include "bart.h"
#include "info.h"
#include "rtnorm.h"

// MCMC algorithm for Cox regression with BART imputation
// Output: List containing posterior samples of beta and the last imputed dataset.
// [[Rcpp::export]]
SEXP cCBI(
    std::vector<double>& T, // Observed time (event or censoring time)
    std::vector<int>& C, // Censoring indicator (1 for event, 0 for censored)
    std::vector<std::vector<double>>& X, // Covariates (features) matrix
    const std::vector<int>& covariate_types, // Covariate types (1 for continuous, 2 for binary)
    const std::vector<std::vector<int>>& missing_matrix, // Missing data indicator matrix (1 for missing, 0 for observed)
    const std::vector<double>& time, // Piecewise time points for the baseline hazard
    size_t num_iterations,  // Number of MCMC iterations
    size_t burn_in, // Number of burn-in iterations
    std::vector<double>& beta, // Coefficients for the Cox model
    std::vector<double>& h0, // Baseline hazard values
    const std::vector<double>& beta_mean, // Mean of the prior for beta
    const std::vector<double>& beta_variances, // Variances of the prior for beta
    const std::vector<double>& h0_prior_shape, // Shape parameters for the prior on h0
    const std::vector<double>& h0_prior_rate, // Rate parameters for the prior on h0
    size_t m, // Number of trees in BART
    int numcut, // Number of cut points for covariate value selection in BART
    double alpha, // parameter for the depth of the trees in BART (p = alpha * (1 + d)^mybeta)
    double mybeta, // parameter for the number of terminal nodes in BART  (p = alpha * (1 + d)^mybeta)
    double nu, // Degrees of freedom for sigma prior
    double lambda, // Scale for sigma prior
    double proposal_stddev_beta, // Standard deviation for the proposal distribution of beta
    double proposal_stddev_h0,  // Standard deviation for the proposal distribution of h0
    double proposal_stddev_X,  // Standard deviation for the proposal distribution of X
    size_t missing_covariate_start // Index of the first missing covariate. Assume all covariates before this index are observed.
) {
    // Check dimensions of the input
    if (T.size() != C.size()) {
        throw std::invalid_argument("T and C must have the same size.");
    }
    if (T.size() != X.size()) {
        throw std::invalid_argument("T and X must have the same number of rows.");
    }
    if (!X.empty() && X[0].size() != covariate_types.size()) {
        throw std::invalid_argument("Each row of X must have the same number of columns as covariate_types.");
    }
    if (missing_matrix.size() != X.size()) {
        throw std::invalid_argument("missing_matrix must have the same number of rows as X.");
    }
    if (!missing_matrix.empty() && missing_matrix[0].size() != X[0].size()) {
        throw std::invalid_argument("Each row of missing_matrix must have the same number of columns as X.");
    }
    if (beta.size() != beta_mean.size() || beta.size() != beta_variances.size()) {
        throw std::invalid_argument("beta, beta_mean, and beta_variances must have the same size.");
    }
    if (h0.size() > time.size()) {
        throw std::invalid_argument("h0[0] size must not exceed time size.");
    }
    if (h0_prior_shape.size() != h0.size() || h0_prior_rate.size() != h0.size()) {
        throw std::invalid_argument("h0_prior_shape and h0_prior_rate must match the size of individual h0.");
    }
    if (missing_covariate_start >= X[0].size()) {
        throw std::invalid_argument("missing_covariate_start must be less than the number of columns in X.");
    }
    // Initialize BART models for imputation
    std::vector<std::pair<bart, double>> bart_models; // Store BART models and their sigmas
    // Compute cumulative baseline hazard for each individual
    auto [H0, h0_for_T] = compute_cumulative_baseline_hazard(h0, T, time);
    for (size_t j = missing_covariate_start; j < X[0].size(); ++j) {
        // Prepare training data for the current covariate
        std::vector<std::vector<double>> x_train;
        std::vector<double> y_train;
        for (size_t i = 0; i < X.size(); ++i) {
            // Select the training data by excluding the jth covariate
            std::vector<double> x_row;
            for (size_t k = 0; k < X[i].size(); ++k) {
                if (k != j) { // Exclude the jth covariate
                    x_row.push_back(X[i][k]);
                }
            }
            x_train.push_back(x_row);
            if (covariate_types[j] == 1) { // Continuous covariate
                y_train.push_back(X[i][j]); // The current covariate as the target
            } else if (covariate_types[j] == 2) { // Binary covariate
                if (X[i][j] == 1.0) {   // Initialize the latent Z to be 1 or -1
                    y_train.push_back(1.0);
                } else {
                    y_train.push_back(-1.0);
                }
            }
        }
        double tau; // parameter for the terminal nodes values in BART (make the crediable interval suitable for the response values)
        if (covariate_types[j] == 1) { // For continuous
            tau = (*std::max_element(y_train.begin(), y_train.end()) - *std::min_element(y_train.begin(), y_train.end())) / (2.0 * 2.0 * std::sqrt(m)); // For continuous covariate
        } else if (covariate_types[j] == 2) {
            tau = 3.0 / (2.0 * std::sqrt(m)); // For binary covariate
        }
        // Initialize a BART model for the current covariate using cumulative hazard
        auto [bart_model, sigma] = initialize_bart(covariate_types[j], x_train, H0, y_train, m, numcut, mybeta, alpha, tau, nu, lambda);
        bart_models.emplace_back(bart_model, sigma);
    }

    // Storage for posterior samples
    std::vector<std::vector<double>> beta_samples;

    // Random number generator
    std::mt19937 gen(1234);
    arn gen_arn;
    for (size_t iter = 0; iter < num_iterations; ++iter) {
        // Step 1: Update H0 and h0 using MH
        h0 = mh_h0(h0, h0_prior_shape, h0_prior_rate, T, C, X, beta, time, proposal_stddev_h0, bart_models, missing_covariate_start, covariate_types);

        // Step 2: Update the BART models using the current X and H0
        // Recompute cumulative baseline hazard with the current h0
        auto [current_H0, h0_for_T] = compute_cumulative_baseline_hazard(h0, T, time);

        for (size_t j = 0; j < bart_models.size(); ++j) {
            std::vector<std::vector<double>> x_train_updated;
            std::vector<double> y_train_updated;
            // Load the current training data for the jth covariate
            for (size_t i = 0; i < X.size(); ++i) {
                std::vector<double> x_row;
                for (size_t k = 0; k < X[i].size(); ++k) {
                    if (k != j + missing_covariate_start) { // Exclude the jth covariate
                        x_row.push_back(X[i][k]);
                    }
                }
                x_train_updated.push_back(x_row);
                if (covariate_types[j + missing_covariate_start] == 1) { // Continuous covariate
                    y_train_updated.push_back(X[i][j + missing_covariate_start]);
                } else if (covariate_types[j + missing_covariate_start] == 2) {
                    // For binary covariate, we need to sample from the truncated normal distribution.
                    double z_mean = predict_bart(bart_models[j].first, x_row, current_H0[i]);
                    double z_sigma = bart_models[j].second;
                    int sign = (X[i][j + missing_covariate_start] == 1.0) ? 1 : -1;
                    double Z = sign * rtnorm(sign * z_mean, 0.0, z_sigma, gen_arn);
                    y_train_updated.push_back(Z);   // Trunctated normal distribution at 0 based on X
                }
            }
            // Update the BART
            auto [updated_bart_model, updated_sigma] = update_bart(
                bart_models[j].first, bart_models[j].second, x_train_updated, current_H0,
                y_train_updated, numcut, covariate_types[j + missing_covariate_start], nu, lambda);

            bart_models[j].first = updated_bart_model;
            bart_models[j].second = updated_sigma;
        }
        // Step 3: Update beta using MH
        beta = mh_beta(beta, beta_mean, beta_variances, T, C, X, h0, time, proposal_stddev_beta);

        // Step 4: Impute missing covariates
        for (size_t i = 0; i < X.size(); ++i) { // Loop by each individual first
            for (size_t j = missing_covariate_start; j < X[0].size(); ++j) {
                if (missing_matrix[i][j] == 1) { // Check if the covariate is missing
                    // Propose a new value for the missing covariate
                    // Prepare the input for prediction: covariates before the current one and cumulative hazard
                    std::vector<double> xj_input;
                    for (size_t k = 0; k < X[i].size(); ++k) {
                        if (k != j) { // Exclude the jth covariate
                            xj_input.push_back(X[i][k]);
                        }
                    }

                    // estimate the mean and standard deviation using BART
                    double bart_predicted_mean = predict_bart(bart_models[j - missing_covariate_start].first, xj_input, current_H0[i]);
                    double bart_sigma = bart_models[j - missing_covariate_start].second;

                    double proposed_value;
                    double current_log_likelihood;
                    double proposed_log_likelihood;
                    if (covariate_types[j] == 1) { // Continuous covariate
                        // Do a random walk proposal by sd = proposal_stddev_X
                        std::normal_distribution<double> proposal_dist(X[i][j], proposal_stddev_X);
                        proposed_value = proposal_dist(gen);

                        // Compute acceptance probability using Normal log-likelihood
                        current_log_likelihood = NormalLikelihood(X[i][j], bart_predicted_mean, bart_sigma);
                        proposed_log_likelihood = NormalLikelihood(proposed_value, bart_predicted_mean, bart_sigma);
                    } else if (covariate_types[j] == 2) { // Binary covariate
                        // Determine the proposed value based on the current value
                        proposed_value = fabs(X[i][j] - 1.0); // Flip the value
                        // log(prob) where p is probability for proposed value. (The probability of Z >= or < 0)
                        double prob = fabs(proposed_value - 0.5 * (1.0 + std::erf((- bart_predicted_mean) / (bart_sigma * std::sqrt(2.0)))));
                        current_log_likelihood = std::log(1 - prob);
                        proposed_log_likelihood = std::log(prob);
                    }
                    // Compute log-likelihood from other covariates
                    double current_covariate_log_likelihood = 0.0;
                    double proposed_covariate_log_likelihood = 0.0;
                    for (size_t k = missing_covariate_start; k < X[0].size(); ++k) {
                        if (k == j) continue;   // For other covariates
                        std::vector<double> xk_input;
                        for (size_t l = 0; l < X[i].size(); ++l) {
                            if (l != k) { // Exclude the kth covariate
                                xk_input.push_back(X[i][l]);
                            }
                        }

                        // For current value of X[i][j]
                        double current_predicted_mean = predict_bart(bart_models[k - missing_covariate_start].first, xk_input, current_H0[i]);
                        double current_sigma = bart_models[k - missing_covariate_start].second;

                        // For proposed value of X[i][j]
                        // Update the proposed covariate in the input vector
                        if (j > k) {xk_input[j-1] = proposed_value;} else {xk_input[j] = proposed_value;}
                        double proposed_predicted_mean = predict_bart(bart_models[k - missing_covariate_start].first, xk_input, current_H0[i]);
                        double proposed_sigma = bart_models[k - missing_covariate_start].second;

                        if (covariate_types[k] == 1) { // Continuous covariate

                            current_covariate_log_likelihood += NormalLikelihood(X[i][k], current_predicted_mean, current_sigma);
                            proposed_covariate_log_likelihood += NormalLikelihood(X[i][k], proposed_predicted_mean, proposed_sigma);
                        } else if (covariate_types[k] == 2) { // Binary covariate
                            // P(X[i][k] = 1 | *) = P(Z >= 0 | *), * is the rest of the current/proposed covariates
                            double current_prob = fabs(X[i][k] - 0.5 * (1.0 + std::erf((- current_predicted_mean) / (current_sigma * std::sqrt(2.0)))));
                            double proposed_prob = fabs(X[i][k] - 0.5 * (1.0 + std::erf((- proposed_predicted_mean) / (proposed_sigma * std::sqrt(2.0)))));
                            current_covariate_log_likelihood += std::log(current_prob);
                            proposed_covariate_log_likelihood += std::log(proposed_prob);
                        }
                    }
                    // Compute Cox likelihood for the current and proposed values
                    std::vector<double> proposed_X = X[i];
                    proposed_X[j] = proposed_value;
                    // Since the covariates are only for individual, only need the individual likelihood
                    double current_cox_likelihood = compute_individual_likelihood(
                        C[i], X[i], beta, h0_for_T[i], current_H0[i]);
                    double proposed_cox_likelihood = compute_individual_likelihood(
                        C[i], proposed_X, beta, h0_for_T[i], current_H0[i]);

                    // Combine all log-likelihoods
                    current_log_likelihood += current_covariate_log_likelihood + current_cox_likelihood;
                    proposed_log_likelihood += proposed_covariate_log_likelihood + proposed_cox_likelihood;

                    double acceptance_ratio;
                    // The probability of accepting the proposed value.
                    if (covariate_types[j] == 1) {
                        acceptance_ratio = std::exp(proposed_log_likelihood - current_log_likelihood);
                    } else if (covariate_types[j] == 2) {
                        // For bianry covariate, it is the probability of change from 1 to 0 or 0 to 1
                        acceptance_ratio = std::exp(proposed_log_likelihood) / (std::exp(current_log_likelihood) + std::exp(proposed_log_likelihood));
                    }
                    if (std::uniform_real_distribution<>(0.0, 1.0)(gen) < acceptance_ratio) {
                        // Accept the proposed value
                        X[i][j] = proposed_value;
                    }
                }
            }
        }

        // Store samples after burn-in
        if (iter >= burn_in) {
            beta_samples.push_back(beta);
        }

        // Print progress
        if ((iter+1) % 100 == 0) {
        Rcpp::Rcout << "Iteration: " << iter+1 << " / " << num_iterations << std::endl;
        }
    }
    // Convert beta_samples to Rcpp::NumericMatrix
    // Output the posterior samples of beta
    size_t num_samples = beta_samples.size();
    size_t num_betas = beta.size();
    Rcpp::NumericMatrix beta_result(num_samples, num_betas);
    for (size_t i = 0; i < num_samples; ++i) {
        for (size_t j = 0; j < num_betas; ++j) {
            beta_result(i, j) = beta_samples[i][j];
        }
    }

    // Convert X to Rcpp::NumericMatrix
    // Output the last imputed dataset
    size_t num_rows = X.size();
    size_t num_cols = X[0].size();
    Rcpp::NumericMatrix X_result(num_rows, num_cols);
    for (size_t i = 0; i < num_rows; ++i) {
        for (size_t j = 0; j < num_cols; ++j) {
            X_result(i, j) = X[i][j];
        }
    }

    Rcpp::List bart_models_result;
    bart_models_result["beta_samples"] = beta_result;
    bart_models_result["Imputed_data"] = X_result;
    return bart_models_result;
}
