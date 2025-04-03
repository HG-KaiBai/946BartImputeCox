install.packages("./BartImputeCox_0.1.0.tar.gz")
library(BartImputeCox)
library(mice)
library(survival)
library(parallel)
library(rje)
options(warn = -1)
# One simulation loop
# Input:
#    seed: Data generation seed
#    missing_rate: For MCAR, the probability of missing
#    method: Type of missing, MCAR or MAR
# Output:
#    A list contains the squared error of estimation, 95% CI cover rate and length for BART, MICE and COX regression
simu <- function(seed, missing_rate, method){
  true_beta = c(0.3, -0.2, 0.5) # The tree hazard ratio for covariates in test_data
  res = c()
  X = test_data(seed, n = 500, mis = missing_rate, method = method)
  T = X$observe  # Time
  C = X$status   # Censoring indicator
  X = X[, -c(1, 2)]  # Only keep covariates

  # BART method
  covariate_types = c(1, 2, 1)  # 1: continuous, 2: binary
  missing_covariate_start = 1  # Start missing 
  dataset = as.matrix(X)
  result = CBI(T, C, dataset, covariate_types, missing_covariate_start, num_iterations = 1000,burn_in = 200)
  beta = result$beta
  estimate = colMeans(beta)  # Estimation
  CI = apply(beta, 2, quantile, probs = c(0.025,0.975))
  square_error = (true_beta - estimate)^2
  cover = (CI[1,] < true_beta) & (CI[2,] > true_beta)
  length = CI[2,] - CI[1,]  # Check 95% CI cover rate and length
  res = c(res, square_error, cover, length)

  # MICE method do the mice and combine with the rubin's rule
  imp = mice(X, m = 5, method = c("pmm", "logreg", "pmm"), seed = seed)
  fit = with(imp, coxph(Surv(T, C) ~ x1 + x2 + x3))
  mod = summary(pool(fit))
  estimate = mod$estimate
  CI = cbind(estimate - 1.96 * mod$std.error, estimate + 1.96 * mod$std.error)
  square_error = (true_beta - estimate)^2
  cover = (CI[,1] < true_beta) & (CI[,2] > true_beta)
  length = CI[,2] - CI[,1]
  res = c(res, square_error, cover, length)

  # Remove all the data, do the cox regression by removing all data with NA values
  mod = coxph(Surv(T, C) ~ X[,1] + X[,2] + X[,3])
  estimate = coef(mod)
  CI = confint(mod)
  square_error = (true_beta - estimate)^2
  cover = (CI[,1] < true_beta) & (CI[,2] > true_beta)
  length = CI[,2] - CI[,1]
  res = c(res, square_error, cover, length)
  return(res)
}

# R times simulation function
main <- function(R, missing_rate, method){
  num_cores <- detectCores() - 1
  result = mclapply(1:R, function(t) simu(t, missing_rate, method), mc.cores = num_cores)
  result = do.call(rbind, result)
  colnames(result) =c("BART_square_error1", "BART_square_error2","BART_square_error3",
                      "BART_cover1", "BART_cover2","BART_cover3",
                      "BART_length1", "BART_length2","BART_length3",
                      "MICE_square_error1","MICE_square_error2","MICE_square_error3",
                      "MICE_cover1","MICE_cover2","MICE_cover3",
                      "MICE_length1","MICE_length2","MICE_length3",
                      "Cox_square_error1","Cox_square_error2","Cox_square_error3",
                      "Cox_cover1", "Cox_cover2", "Cox_cover3",
                      "Cox_length1", "Cox_length2", "Cox_length3")
  result = rbind(result, colMeans(result))
  rownames(result) = c(seq(1:R), "Ave")
  filename = paste0(method,"_mis", missing_rate, ".csv")
  write.csv(result, filename)
}


#start.time <- Sys.time()
#main(500, 0.2, "MCAR")
#print(Sys.time() - start.time)
#start.time <- Sys.time()
#main(500, 0.5, "MCAR")
#print(Sys.time() - start.time)
#start.time <- Sys.time()
#main(500, 0.2, "MAR")
#print(Sys.time() - start.time)
#start.time <- Sys.time()
#main(500, 0.5, "MAR")
#print(Sys.time() - start.time)
