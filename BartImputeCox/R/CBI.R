#' Cox regression with BART imputation
#'
#' @param C Censoring indicators
#' @param X Covariates
#' @param covariate_types Covariate types (1: continuous, 2: binary)
#' @param missing_covariate_start Index of the first covariate with missing values
#' @param time Piecewise time points for the baseline hazard
#' @param num_iterations Number of iterations for MCMC
#' @param burn_in Number of burn-in iterations
#' @param beta Initial beta values
#' @param h0 Initial baseline hazard
#' @param beta_mean Mean of the beta prior
#' @param beta_variances Variances of the beta prior
#' @param h0_prior_shape Shape of the baseline hazard prior
#' @param h0_prior_rate Rate of the baseline hazard prior
#' @param m Number of trees in the BART model
#' @param numcut Number of cutpoints for splitting in the BART model
#' @param alpha Parameter for the depth of the trees in BART (p = alpha * (1 + d)^mybeta)
#' @param mybeta Parameter for the number of terminal nodes in BART  (p = alpha * (1 + d)^mybeta)
#' @param nu Degrees of freedom for sigma prior
#' @param lambda Scale for sigma prior
#' @param proposal_stddev_beta  Standard deviation for the proposal distribution of beta
#' @param proposal_stddev_h0 Standard deviation for the proposal distribution of h0
#' @param proposal_stddev_X Standard deviation for the proposal distribution of X
#'
#' @returns A list with the following components: A matrix of beta samples, last imputed X
#' @export
#'
#' @examples
#' X = test_data(1, n = 500)
#' T = X$observe
#' C = X$status
#' X = X[, -c(1, 2)]
#' covariate_types = c(1, 2, 1)
#' missing_covariate_start = 2
#' X = as.matrix(X)
#' result = CBI(T, C, X, covariate_types, missing_covariate_start, num_iterations = 10000,burn_in = 5000)

CBI <- function(T,  # Survival times
                C, # Censoring indicators
                X,  # Covariates
                covariate_types,  # Covariate types (e.g., continuous)
                missing_covariate_start = 0,
                time = NULL,  # Time grid
                num_iterations = 1000,
                burn_in = 200,
                beta = NULL,  # Initial beta values
                h0 = NULL,  # Initial baseline hazard
                beta_mean = NULL,
                beta_variances = NULL,
                h0_prior_shape = NULL,
                h0_prior_rate = NULL,
                m = 200,
                numcut = 100,
                alpha = 0.95,
                mybeta = 2.0,
                nu = 3.0,
                lambda = 1.0,
                proposal_stddev_beta = 1,
                proposal_stddev_h0 = 1,
                proposal_stddev_X = 1){
  # Initial all null values. The time intervals default divide the time from 0 to max(T) into 10 intervals.
  if (is.null(time)) {
    time = quantile(T, probs = seq(0, 1, 0.1))
  }
  time = as.numeric(time)
  time = c(0, time)
  if(is.null(beta_mean)){
    beta_mean = rep(0, ncol(X))
  }
  if(is.null(beta_variances)){
    beta_variances = rep(1, ncol(X))
  }
  if(is.null(beta)){
    beta = rep(0, ncol(X))
  }
  if(is.null(h0_prior_shape)){
    h0_prior_shape = rep(1, length(time))
  }
  if(is.null(h0_prior_rate)){
    h0_prior_rate = rep(1, length(time))
  }
  if(is.null(h0)){
    h0 = rep(0.1, length(time))
  }
  # Generate the missing matrix to indicate the missing values in X
  missing_matrix = matrix(0, nrow = nrow(X), ncol = ncol(X))
  missing_matrix[is.na(X)] = 1
  # Single imputation for the X
  for (i in 1:ncol(X)){
    if (covariate_types[i] == 1){
      # Continuous covariate
      # Selcet a random value from normal distribution with mean of the observed value
      X[is.na(X[,i]),i] = rnorm(sum(is.na(X[,i])), mean(X[,i], na.rm = TRUE), proposal_stddev_X)
    } else if(covariate_types[i] == 2){
      # Binary covariate
      # Selcet a random value from binomial distribution with probability with mean of the observed value
      X[is.na(X[,i]),i] = rbinom(sum(is.na(X[,i])), 1, mean(X[,i], na.rm = TRUE))
    }
  }
  # Split the matrix into a list of rows to fit the c++ input type
  X = split(X, 1:nrow(X))
  missing_matrix = split(missing_matrix, 1:nrow(missing_matrix))
  # Call the cCBI from src file.
  result <- cCBI(T, C, X, covariate_types, missing_matrix, time,
                  num_iterations, burn_in,
                  beta, h0, beta_mean, beta_variances,
                  h0_prior_shape, h0_prior_rate,
                  m, numcut, alpha, mybeta, nu, lambda,
                  proposal_stddev_beta, proposal_stddev_h0, proposal_stddev_X,
                  missing_covariate_start)
  return(result)
}
