# Tutorial with synthetic data

# Install devtools if needed
# install.packages("devtools")

# Install fastrerandomize if you haven't already
# devtools::install_github(repo = "cjerzak/fastrerandomize-software/fastrerandomize")


# Load the package
library(  fastrerandomize  )

# Before running any code, you'll need to initialize the JAX environment
fastrerandomize::InitializeJAX(conda_env = "tensorflow_m1", conda_env_required = T)

# If you didn't use a conda environment in which to install JAX, try:
# fastrerandomize::InitializeJAX(  conda_env = NULL )

# Note: If you leave `conda_env = NULL`, we will search in the default Python environment for JAX.

# First, specify some analysis parameters
n_units <- 20; n_treated <- 10

# Generate covariate data
X <- matrix(rnorm(n_units*5),nrow = n_units)

# Generate set of acceptable randomizations based randomization_accept_prob.
# When randomization_accept_prob = 1, all randomizations are accepted.
# When randomization_accept_prob < 1, only well-balanced randomizations are accepted.
# When randomization_accept_prob = 1/|Size of cand. randomization set|, 1 randomization is accepted.
candidate_randomizations_array <- fastrerandomize::GenerateRandomizations(
  n_units = n_units,
  n_treated = n_treated,
  X = X,
  randomization_accept_prob = 0.001)
candidate_randomizations_array$shape

# You can coerce candidate_randomizations_array into R like this:
candidate_randomizations <- np$array( candidate_randomizations_array )
dim( candidate_randomizations )

# We can also use `fastrerandomize` to perform a randomization test using those acceptable randomizations.
# Setup simulated outcome data
CoefY <- rnorm(ncol(X))
Wobs <- candidate_randomizations[1,]
tau_true <- 1
Yobs <- c(X %*% as.matrix(CoefY) + Wobs*tau_true + rnorm(n_units, sd = 0.1))

# Perform exact randomization set based on accepted randomizations
ExactRandomizationTestResults <- fastrerandomize::RandomizationTest(
  obsW = Wobs,
  obsY = Yobs,
  candidate_randomizations = candidate_randomizations,
  findFI = F # set to T if an exact fiducial interval needed
)
ExactRandomizationTestResults$p_value # p-value
ExactRandomizationTestResults$tau_obs # difference-in-means ATE estimate

#You can also use `fastrerandomize` to perform a pre-analysis evaluation for what would be an _a priori_ optimal rerandomization acceptance threshold in terms of minimizing the expected exact _p_-value.
# setup acceptable randomization sequence
n_randomizations <- choose(n_units, n_treated)
starting_value = abs( min(log( 2/n_randomizations, base = 10 ), log(0.05,base=10)))
prob_accept_randomization_seq <- 10^(- seq(from = starting_value, to = 1/3, length.out = 32L ) )

# perform pre-design analysis (runtime: ~20 seconds)
PreAnalysisEvaluation <- fastrerandomize::RandomizationTest(
  X = X,
  randomization_accept_prob = prob_accept_randomization_seq,
  prior_treatment_effect_mean = 0.1,
  prior_treatment_effect_SD = 1,
  coef_prior = function(){rnorm(ncol(X), sd = 1)},
  simulate = T
)

# suggested acceptance threshold minimizing a priori expected p-value
PreAnalysisEvaluation$suggested_randomization_accept_prob

# expected p-values for all values of randomization_accept_prob input
PreAnalysisEvaluation$p_value

