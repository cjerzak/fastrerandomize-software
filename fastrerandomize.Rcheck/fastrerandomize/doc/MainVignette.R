## ----setup, eval=FALSE--------------------------------------------------------
# # If you haven't installed or set up the package:
# # devtools::install_github("cjerzak/fastrerandomize-software/fastrerandomize")
# 
# # (Done once) Optionally build the JAX backend if needed
# # fastrerandomize::build_backend()
# 
# # note that this vignette can be found like so:
# # vignette(package = "fastrerandomize")

## ----generate_covariates------------------------------------------------------
library(fastrerandomize)

# Obtain pre-treatment covariates 
data(QJEData, package = "fastrerandomize")
myCovariates <- c("children","married","hh_size","hh_sexrat")
QJEData <- QJEData[!is.na(rowSums(QJEData[,myCovariates])),]
X <- QJEData[,myCovariates]
  
# Analysis parameters
n_units   <- nrow(X)
n_treated <- round(nrow(X)/2)

## ----generate_randomizations--------------------------------------------------
RunMainAnalysis <- (!is.null(check_jax_availability()) )
# if FALSE, consider calling build_backend()

if(RunMainAnalysis){
CandRandomizations <- generate_randomizations(
  n_units = n_units,
  n_treated = n_treated,
  X = X,
  randomization_type = "monte_carlo", 
  max_draws = 100000L,
  batch_size = 1000L,
  randomization_accept_prob = 0.0001
)
}

## ----s3_methods---------------------------------------------------------------
if(RunMainAnalysis){
# 4a. Print the object
print(CandRandomizations)

# 4b. Summary
summary(CandRandomizations)

# 4c. Plot the balance distribution
plot(CandRandomizations)
}

## ----setup_outcomes-----------------------------------------------------------
if(RunMainAnalysis){
CoefY <- rnorm(ncol(X))          # Coefficients for outcome model
Wobs  <- CandRandomizations$randomizations[1, ]  # use the 1st acceptable randomization
tau_true <- 1                     # true treatment effect
# Generate Y = X * Coef + W*tau + noise
Yobs <- as.numeric( as.matrix(X) %*% as.matrix(CoefY) + Wobs * tau_true + rnorm(n_units, sd = 0.1))
}

## ----randomization_test-------------------------------------------------------
if(RunMainAnalysis){
randomization_test_results <- randomization_test(
  obsW = Wobs,
  obsY = Yobs,
  candidate_randomizations = CandRandomizations$randomizations,
  findFI = TRUE
)

# Inspect results
print(randomization_test_results)
summary(randomization_test_results)
plot(randomization_test_results)
}

