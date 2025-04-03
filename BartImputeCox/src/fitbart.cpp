#include <ctime>
#include "common.h"
#include "tree.h"
#include "treefuns.h"
#include "info.h"
#include "bartfuns.h"
#include "bd.h"
#include "bart.h"
#include "rtnorm.h"
#include "lambda.h"
#include "fitbart.h" // See documation for details on the functions in this file.

std::pair<bart, double> initialize_bart(
    int type,                          // 1: continuous BART, 2: binary BART
    const std::vector<std::vector<double>>& x_train, // x, train, nxp
    const std::vector<double>& H0,             // cumulated hazard
    const std::vector<double>& y_train,             // y, train, nx1
    size_t m,                          // number of trees
    int numcut,                        // number of cut points (as int)
    double mybeta,                     // power = 2
    double alpha,                      // base = 0.95
    double tau,
    double nu,                   // Degrees of freedom for sigma prior (continuous BART) 3.0
    double lambda                // Scale for sigma prior (continuous BART) 1.0
) {

    // Add H0 as a new column to x_train
    std::vector<std::vector<double>> x_train_with_H0 = x_train;
    for (size_t i = 0; i < x_train_with_H0.size(); ++i) {
        x_train_with_H0[i].push_back(H0[i]);
    }
    // Determine the sample size n and number of covariates p
    size_t n = x_train_with_H0.size();
    size_t p = x_train_with_H0[0].size();

    // Initialize BART model
    bart bm(m);

    // Flatten x_train_with_H0 for compatibility
    std::vector<double> ix;
    for (const auto& row : x_train_with_H0) {
        ix.insert(ix.end(), row.begin(), row.end());
    }
    std::vector<double> y_train_copy = y_train; // Ensure y_train is contiguous

    // Set the prior parameters for BART
    bm.setprior(alpha, mybeta, tau);

    // Create a vector of numcut values for each predictor
    std::vector<int> numcut_vec(p, numcut);
    bm.setdata(p, n, ix.data(), y_train_copy.data(), numcut_vec.data());

    // Initialize sigma for continuous BART
    double sigma = 1.0; // Default value as 1 for binary BART.
    if (type == 1) { // Initialize as the sample sd for continuous BART
        double y_mean = std::accumulate(y_train.begin(), y_train.end(), 0.0) / n;
        double y_var = std::accumulate(y_train.begin(), y_train.end(), 0.0,
            [y_mean](double acc, double val) { return acc + (val - y_mean) * (val - y_mean); }) / n;
        sigma = sqrt(y_var);
    }

    // Perform the first training iteration
    arn gen;
    bm.draw(sigma, gen);
    if (type == 1) { // Continuous BART
        // Draw sigma from the inverse chi-squared distribution
        double rss = 0.0;
        for (size_t i = 0; i < n; i++) {
            rss += pow(y_train[i] - bm.f(i), 2); // Residual sum of squares
        }
        double df = n + nu;
        sigma = sqrt((nu * lambda + rss) / gen.chi_square(df)); // Update sigma
    }
    return {bm, sigma};
}

std::pair<bart, double> update_bart(
    const bart& bm,
    double& sigma,
    const std::vector<std::vector<double>>& x_train,
    const std::vector<double>& H0,
    const std::vector<double>& y_train,
    int numcut,    // number of cut points
    int type,                          // 1: continuous BART, 2: binary BART
    double nu,                   // Degrees of freedom for sigma prior (continuous BART) 3.0
    double lambda                // Scale for sigma prior (continuous BART) 1.0
) {
    // Reshape the current training data to include H0
    std::vector<std::vector<double>> x_train_with_H0 = x_train;
    for (size_t i = 0; i < x_train_with_H0.size(); ++i) {
        x_train_with_H0[i].push_back(H0[i]);
    }
    size_t n = x_train_with_H0.size();
    size_t p = x_train_with_H0[0].size();

    arn gen;
    // Flatten x_train for compatibility
    std::vector<double> ix;
    for (const auto& row : x_train_with_H0) {
        ix.insert(ix.end(), row.begin(), row.end());
    }
    std::vector<double> y_train_copy = y_train; // Ensure y_train is contiguous
    // Update BART data with new x_train, y_train, and numcut
    std::vector<int> numcut_vec(p, numcut);
    // Initialize a new BART model with same parameters and tree structures as the previous bm
    bart new_bm(bm);
    // Update the new training data
    new_bm.setdata(p, n, ix.data(), y_train_copy.data(), numcut_vec.data());

    // Perform one draw with the new training data
    new_bm.draw(sigma, gen);

    if (type == 1) { // Continuous BART
        // Draw sigma from the inverse chi-squared distribution
        double rss = 0.0;
        for (size_t i = 0; i < n; i++) {
            rss += pow(y_train[i] - new_bm.f(i), 2);
        }
        double df = n + nu;
        sigma = sqrt((nu * lambda + rss) / gen.chi_square(df));
    }
    // Node need to draw for binary BART as it is initialized as 1 before.

    return {new_bm, sigma};
}

double predict_bart(
    bart& bm,
    const std::vector<double>& x_test,
    double H0
) {
    size_t n_test = 1; // Single test sample

    // Allocate memory for prediction
    std::vector<double> predictions(n_test);

    // Perform prediction
    std::vector<double> x_test_copy = x_test; // Create a non-const copy
    x_test_copy.push_back(H0); // Add H0 to the test sample

    size_t p = x_test_copy.size(); 

    bm.predict(p, n_test, x_test_copy.data(), predictions.data());
    // Return the single prediction value
    return predictions[0];
}
