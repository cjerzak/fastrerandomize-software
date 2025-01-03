% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/FRR_GenerateRandomizations_MonteCarlo.R
\name{generate_randomizations_mc}
\alias{generate_randomizations_mc}
\title{Draws a random sample of acceptable randomizations from all possible complete randomizations using Monte Carlo sampling}
\usage{
generate_randomizations_mc(
  n_units,
  n_treated,
  X,
  randomization_accept_prob = 1,
  threshold_func = VectorizedFastHotel2T2,
  max_draws = 1e+05,
  batch_size = 1000,
  seed = NULL,
  approximate_inv = TRUE,
  verbose = FALSE,
  file = NULL,
  conda_env = "fastrerandomize",
  conda_env_required = T
)
}
\arguments{
\item{n_units}{An integer specifying the total number of experimental units}

\item{n_treated}{An integer specifying the number of units to be assigned to treatment}

\item{X}{A numeric matrix of covariates used for balance checking. Cannot be NULL.}

\item{randomization_accept_prob}{A numeric value between 0 and 1 specifying the probability threshold for accepting randomizations based on balance. Default is 1}

\item{threshold_func}{A JAX function that computes a balance measure for each randomization. Must be vectorized using jax$vmap with in_axes = list(NULL, 0L, NULL, NULL), and inputs covariates (matrix of X), treatment_assignment (vector of 0s and 1s), n0 (scalar), n1 (scalar). Default is VectorizedFastHotel2T2 which uses Hotelling's T^2 statistic}

\item{max_draws}{An integer specifying the maximum number of randomizations to draw. Default is 100000}

\item{batch_size}{An integer specifying how many randomizations to process at once. Default is 10000. Lower values use less memory but may be slower}

\item{seed}{An integer seed for random number generation. Default is 42}

\item{verbose}{A logical value indicating whether to print detailed information about batch processing progress, and GPU memory usage. Default is FALSE}
}
\value{
A JAX array containing the accepted randomizations, where each row represents one possible treatment assignment vector
}
\description{
This function performs sampling with replacement to generate randomizations in a memory-efficient way.
It processes randomizations in batches to avoid memory issues and filters them based on covariate balance.
The function uses JAX for fast computation and memory management.
}
\details{
The function works by:
\enumerate{
\item Generating batches of random permutations using JAX's random permutation functionality
\item Computing balance measures for each permutation using the provided threshold function
\item Keeping only the top permutations that meet the acceptance probability threshold
\item Managing memory by clearing unused objects and JAX caches between batches
}

The function uses smaller data types (int8, float16) where possible to reduce memory usage.
It also includes assertions to verify array shapes and dimensions throughout.
}
\examples{
# Generate 1000 randomizations for 100 units with 50 treated
X <- matrix(rnorm(100*5), 100, 5) # 5 covariates
rand <- GenerateRandomizations_MonteCarlo(100, 50, X, max_draws=1000)

# Use a stricter balance criterion
rand_strict <- GenerateRandomizations_MonteCarlo(
               n_units = 100, 
               n_treated = 50, 
               X = X, 
               randomization_accept_prob=0.1, 
               max_draws=1000)

}
\seealso{
\code{\link{GenerateRandomizations}} for the non-Monte Carlo version
\code{\link{VectorizedFastHotel2T2}} for the default threshold function
}
