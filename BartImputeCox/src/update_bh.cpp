#include <random>
#include <cmath>
#include "update_bh.h" // See documation for details on the functions in this file.
#include "prior_likelihood.h"
#include "cox_likelihood.h"
#include "fitbart.h"
#include "bart.h"


std::vector<double> mh_beta(
    const std::vector<double>& current_beta, const std::vector<double>& mean,
    const std::vector<double>& variances, const std::vector<double>& T,
    const std::vector<int>& C, const std::vector<std::vector<double>>& X,
    const std::vector<double>& h0, const std::vector<double>& time,
    double proposal_stddev) {

    // Check dimensions of the input
    if (current_beta.size() != mean.size() || current_beta.size() != variances.size()) {
        throw std::invalid_argument("Dimension mismatch: current_beta, mean, and variances must have the same size.");
    }
    if (T.size() != C.size() || T.size() != X.size()) {
        throw std::invalid_argument("Dimension mismatch: (T, C, X) must have the correct sizes.");
    }
    if (h0.size()  > time.size()) {
        throw std::invalid_argument("Dimension mismatch: h0[0] size must not exceed time size.");
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> proposal_dist(0.0, proposal_stddev);
    std::vector<double> proposal_beta = current_beta;

    // Propose and update each beta_i
    for (size_t i = 0; i < current_beta.size(); ++i) {
        proposal_beta[i] += proposal_dist(gen); // Generate proposal for beta_i

        // Do the MH algorithm
        double current_prior = NormalLikelihood(current_beta[i], mean[i], std::sqrt(variances[i]));
        double proposal_prior = NormalLikelihood(proposal_beta[i], mean[i], std::sqrt(variances[i]));

        double current_likelihood = compute_cox_likelihood(T, C, X, current_beta, h0, time);
        double proposal_likelihood = compute_cox_likelihood(T, C, X, proposal_beta, h0, time);

        double acceptance_ratio = std::exp((proposal_prior + proposal_likelihood) -
                                           (current_prior + current_likelihood));

        std::uniform_real_distribution<> uniform_dist(0.0, 1.0);
        if (uniform_dist(gen) > acceptance_ratio) {
            proposal_beta[i] = current_beta[i]; // Not accept the proposal for beta_i
        }
    }
    return proposal_beta;
}


std::vector<double> mh_h0(
    const std::vector<double>& current_h0, const std::vector<double>& h0_prior_shape,
    const std::vector<double>& h0_prior_rate, std::vector<double>& T, std::vector<int>& C,
    const std::vector<std::vector<double>>& X, const std::vector<double>& beta,
    const std::vector<double>& time, double proposal_stddev_h0,
    const std::vector<std::pair<bart, double>>& bart_models, size_t missing_covariate_start, const std::vector<int>& covariate_types) {

    if (current_h0.size() != h0_prior_shape.size() || current_h0.size() != h0_prior_rate.size()) {
        throw std::invalid_argument("Dimension mismatch: current_h0, h0_prior_shape, and h0_prior_rate must have the same size.");
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::lognormal_distribution<> proposal_dist(0.0, proposal_stddev_h0);
    std::vector<double> proposal_h0 = current_h0;
    
    // Propose and update each h0_j
    for (size_t j = 0; j < current_h0.size(); ++j) {
        //  Generate proposal for h0_j using log-normal distribution
        proposal_h0[j] *= proposal_dist(gen); // Generate proposal for h0_j using log-normal distribution
        auto [current_H0, current_h0_for_T] = compute_cumulative_baseline_hazard(current_h0, T, time);
        auto [proposal_H0, proposal_h0_for_T] = compute_cumulative_baseline_hazard(proposal_h0, T, time);

        
        double current_prior = GammaLikelihood(current_h0[j], h0_prior_shape[j], h0_prior_rate[j]);
        double proposal_prior = GammaLikelihood(proposal_h0[j], h0_prior_shape[j], h0_prior_rate[j]);
        
        // Compute log-likelihood for the covariates with current and proposed hazard
        double current_covariate_log_likelihood = 0.0;
        double proposal_covariate_log_likelihood = 0.0;
        for (size_t k = missing_covariate_start; k < X[0].size(); ++k) {
            bart current_bart = bart_models[k - missing_covariate_start].first;
            for (size_t i = 0; i < X.size(); ++i) {
                std::vector<double> x_input;
                for (size_t l = 0; l < X[i].size(); ++l) {
                    if (l != k) { // Exclude the kth covariate
                        x_input.push_back(X[i][l]);
                    }
                }
                
                // For current value of h0
                double current_predicted_mean = predict_bart(current_bart, x_input, current_H0[i]);
                double current_sigma = bart_models[k - missing_covariate_start].second;

                // For proposed value of h0
                double proposed_predicted_mean = predict_bart(current_bart, x_input, proposal_H0[i]);
                double proposed_sigma = bart_models[k - missing_covariate_start].second;
                if (covariate_types[k] == 1) { // Continuous covariate
                    current_covariate_log_likelihood += NormalLikelihood(X[i][k], current_predicted_mean, current_sigma);
                    proposal_covariate_log_likelihood += NormalLikelihood(X[i][k], proposed_predicted_mean, proposed_sigma);
                } else if (covariate_types[k] == 2) { // Binary covariate
                    double current_prob = fabs(X[i][k] - 0.5 * (1.0 + std::erf((- current_predicted_mean) / (current_sigma * std::sqrt(2.0)))));
                    double proposed_prob = fabs(X[i][k] - 0.5 * (1.0 + std::erf((- proposed_predicted_mean) / (proposed_sigma * std::sqrt(2.0)))));
                    current_covariate_log_likelihood += std::log(current_prob);
                    proposal_covariate_log_likelihood += std::log(proposed_prob);
                }
            }
        }
        double current_likelihood = compute_cox_likelihood(T, C, X, beta, current_h0, time);
        double proposal_likelihood = compute_cox_likelihood(T, C, X, beta, proposal_h0, time);

        current_likelihood += current_covariate_log_likelihood;
        proposal_likelihood += proposal_covariate_log_likelihood;

        double acceptance_ratio = std::exp((proposal_prior + proposal_likelihood) -(current_prior + current_likelihood)) * proposal_h0[j] / current_h0[j];

        std::uniform_real_distribution<> uniform_dist(0.0, 1.0);
        if (uniform_dist(gen) > acceptance_ratio) {
            proposal_h0[j] = current_h0[j]; // Not accept the proposal for h0_j
        }
    }
    return proposal_h0;
}
