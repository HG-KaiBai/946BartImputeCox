#include "fitbart.h"
#include "testbart.h" // See documation for details on the functions in this file.
#include "bart.h"
#include <random>

// Function to take all inputs and return the result of initialize_bart
double initialize_and_predict(
    int type,
    const std::vector<std::vector<double>>& x_train,
    const std::vector<double>& H0,
    const std::vector<double>& y_train,
    const std::vector<double>& x_test,
    double H0_test,
    size_t m,
    int numcut,
    double mybeta,  // 2
    double alpha,   // 0.95
    double tau,    // variance for terminal nodes. default as (max(y) - min(y)) / (2*k*sqrt(m)) k = 2 for 95% interval
    double nu,   // 3.0
    double lambda) {

    // Check the shape of x_train, H0, and y_train
    if (x_train.empty() || x_train[0].empty()) {
        throw std::invalid_argument("x_train is empty or improperly formatted.");
    }
    if (H0.size() != x_train.size()) {
        throw std::invalid_argument("H0 size does not match the number of rows in x_train.");
    }
    if (y_train.size() != x_train.size()) {
        throw std::invalid_argument("y_train size does not match the number of rows in x_train.");
    }
    // Initialize BART
    auto init_result = initialize_bart(type, x_train, H0, y_train, m, numcut, mybeta, alpha, tau, nu, lambda);
    bart bm = init_result.first;
    double sigma = init_result.second;

    // Predict using initialized BART
    double predicted_value = predict_bart(bm, x_test, H0_test);

    // Sample from normal distribution
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(predicted_value, sigma);

    return dist(gen);
}

// Function to initialize and update BART
double initialize_update_and_predict(
    int type,
    const std::vector<std::vector<double>>& x_train,
    const std::vector<double>& H0,
    const std::vector<double>& y_train,
    const std::vector<double>& x_test,
    double H0_test,
    size_t m,
    int numcut,
    double mybeta,
    double alpha,
    double tau,
    double nu,
    double lambda) {
    // Check the shape of x_train, H0, and y_train
    if (x_train.empty() || x_train[0].empty()) {
        throw std::invalid_argument("x_train is empty or improperly formatted.");
    }
    if (H0.size() != x_train.size()) {
        throw std::invalid_argument("H0 size does not match the number of rows in x_train.");
    }
    if (y_train.size() != x_train.size()) {
        throw std::invalid_argument("y_train size does not match the number of rows in x_train.");
    }
    // Initialize BART
    auto init_result = initialize_bart(type, x_train, H0, y_train, m, numcut, mybeta, alpha, tau, nu, lambda);
    bart bm = init_result.first;
    double sigma = init_result.second;

    // Update BART
    auto update_result = update_bart(bm, sigma, x_train, H0, y_train, numcut, type, nu, lambda);
    bm = update_result.first;
    sigma = update_result.second;


    printf("sigma: %f\n", sigma);
    // Update BART twice
    update_result = update_bart(bm, sigma, x_train, H0, y_train, numcut, type, nu, lambda);
    bm = update_result.first;
    sigma = update_result.second;

    // Predict using updated BART
    double predicted_value = predict_bart(bm, x_test, H0_test);

    // Sample from normal distribution
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(predicted_value, sigma);

    return dist(gen);
}
