#' Generate a Weibull survival dataset with missing covariates by MAR / MCAR
#'
#' @param seed Set seed
#' @param n Number of observations
#' @param lambda Scale parameter of the Weibull distribution
#' @param K Shape parameter of the Weibull distribution
#' @param q Censoring proportion
#' @param mis Missing proportion
#' @param method Missing data mechanism: "MCAR" or "MAR"
#'
#' @returns A data frame with the following columns: observe, status, x1, x2, x3
#' @export
#' @import rje
#'
#' @examples
#' data = test_data(1, n = 500)
#' data = test_data(1, n = 500, lambda = 0.1, K = 1.5, q = 0.2)
test_data <- function(seed, n = 500, lambda = 0.2, K = 1.5,
                      q = 0.2, mis = 0.2, method = "MCAR"){
  set.seed(seed)
  x1 = rnorm(n, mean = 0, sd = 1)  # Fully observed

  # Generate dependent covariates
  x2 = rbinom(n, 1, prob = plogis(0.5 * x1))  # Dependence on x1
  x3 = rnorm(n, mean = sin(x1)+0.3*x2, sd = 1)  # Non-linear dependence on x1

  # True regression function (non-linear)
  lp = 0.3 * x1 - 0.2 * x2 + 0.5 * x3

  # Generate survival times (Weibull)
  scal = lambda^(-1/K) * exp(-lp/K)
  T = rweibull(n, shape = K, scale = scal)
  scal_c = scal / (q / (1 - q))^(1/K)

  # Generate censoring times (independent exponential)
  C = rweibull(n, shape = K, scale = scal_c)

  # Observed time and event indicator
  observe = pmin(T, C)
  status = as.numeric(T <= C)

  # Introduce missing values
  if (method == "MCAR") {
    id2 = rbinom(n, 1, prob = mis)
    x2[id2 == 1] <- NA  # missing in x2
    id3 = rbinom(n, 1, prob = mis)
    x3[id3 == 1] <- NA  # missing in x3
  } else if (method == "MAR") {
    # Missing at random: introduce missingness based on x1
    id2 = rbinom(n, 1, prob = expit(mis + 0.5 * x1))
    x2[id2 == 1] <- NA  # missing in x2
    id3 = rbinom(n, 1, prob = expit(mis + 0.5 * x1))
    x3[id3 == 1] <- NA  # missing in x3
  }
  # Combine into a data frame
  data = data.frame(observe, status, x1, x2, x3)
  return(data)
}
