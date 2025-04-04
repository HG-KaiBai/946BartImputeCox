# BartImputeCox: Bayesian Cox Regression with BART-Based Imputation for Missing Covariates
Bayesian Cox regression under piecewise constant hazard assumption with BART based imputation for missing covariates. The package is based on the BART package by  Sparapani et al., which can be found at (doi:10.18637/jss.v097.i01).

## Installation

```r
 install.packages("BartImputeCox_0.1.0.tar.gz")
```

## Usage 

File `testcase.Rmd` contains a simple example of how to use the package. The example is generated from the `test_data` function in the `BartImputeCox` package. The example is a simulated dataset with 500 observations and 3 covariates. The dataset contains missing values in the covariates under MAR or MCAR scheme.

Folder `Simulation results` contains the simulation results of the package. The simulation is based on the `test_data` function in the `BartImputeCox` package. The simulation is done under different missing data mechanisms (MCAR, MAR) with 500 sample size.