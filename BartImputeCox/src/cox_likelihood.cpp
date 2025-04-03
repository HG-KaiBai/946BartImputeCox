#include "cox_likelihood.h" // See documation for details on the functions in this file.
#include <cmath>
#include <numeric>


std::pair<std::vector<double>, std::vector<double>> compute_cumulative_baseline_hazard(
  const std::vector<double>& h0, const std::vector<double>& T,
  const std::vector<double>& time) {
  std::vector<double> H0(T.size(), 0.0);
  std::vector<double> h0_for_T(T.size(), 0.0);

  for (size_t i = 0; i < T.size(); ++i) {
    double sum_hazard = 0.0;
    for (size_t j = 0; j < time.size() - 1; ++j) {
      if (T[i] > time[j] && T[i] <= time[j + 1]) { // Loop the finde the interval piece
        sum_hazard += h0[j] * (T[i] - time[j]); // store the cumulative hazard
        h0_for_T[i] = h0[j];
        break;
      } else if (T[i] > time[j + 1]) {
        sum_hazard += h0[j] * (time[j + 1] - time[j]); // store the cumulative hazard
      }
    }
    H0[i] = sum_hazard;
  }
  return {H0, h0_for_T};
}

double compute_individual_likelihood(
    int C, const std::vector<double>& X,
    const std::vector<double>& beta, double h0, double H0) {
  double linear_pred = 0.0;
  for (size_t j = 0; j < X.size(); ++j) {
    linear_pred += X[j] * beta[j];  // Compute linear predictor
  }

  double likelihood;
  likelihood = C * (std::log(h0) + linear_pred) - H0 * exp(linear_pred); // individual cox likelihood
  return likelihood;
}

double compute_cox_likelihood(
  const std::vector<double>& T, const std::vector<int>& C,
  const std::vector<std::vector<double>>& X, const std::vector<double>& beta,
  const std::vector<double>& h0, const std::vector<double>& time) {

  double log_likelihood = 0.0;

  // Loop through all individuals to compute the total likelihood
  for (size_t i = 0; i < T.size(); ++i) {
    size_t h0_index = 0;
    double sum_hazard = 0.0;
    for (size_t j = 0; j < time.size() - 1; ++j) { // find the hazard first
      if (T[i] > time[j] && T[i] <= time[j + 1]) {
        h0_index = j;
        sum_hazard += h0[j] * (T[i] - time[j]);
        break;
      } else if (T[i] > time[j + 1]) {
        sum_hazard += h0[j] * (time[j + 1] - time[j]);
      }
    }
    double individual_likelihood = compute_individual_likelihood(C[i], X[i], beta, h0[h0_index], sum_hazard);
    log_likelihood += individual_likelihood;
  }
  return log_likelihood;
}


