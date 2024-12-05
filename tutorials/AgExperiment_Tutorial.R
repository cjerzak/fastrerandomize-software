{
options(error = NULL)
#######################################
# AgExperiment_Tutorial.R - An agricutlure experiment tutorial
#######################################  

# Install devtools if needed
# install.packages("devtools")

# Install fastrerandomize if you haven't already
# devtools::install_github(repo = "cjerzak/fastrerandomize-software/fastrerandomize")

# local install for development team
# install.packages("~/Documents/fastrerandomize-software/fastrerandomize",repos = NULL, type = "source",force = F)
  
# build backend if needed
# fastrerandomize::build_backend()

# start timer
t0 <- Sys.time()

# First, specify some analysis parameters
n_units <- 22L; n_treated <- 12L

# Generate covariate data
X <- matrix(rnorm(n_units*5),nrow = n_units)

# Generate set of acceptable randomizations based randomization_accept_prob.
# When randomization_accept_prob = 1, all randomizations are accepted.
# When randomization_accept_prob < 1, only well-balanced randomizations are accepted.
# When randomization_accept_prob = 1/|Size of cand. randomization set|, 1 randomization is accepted.
CandRandomizationsPackage <- fastrerandomize::generate_randomizations(
  n_units = n_units,
  n_treated = n_treated,
  X = X,
  randomization_type = "exact", max_draws = 10000L,  # exact sampling 
  #randomization_type = "monte_carlo", max_draws = 50000L, batch_size = 1000L, # monte carlo sampling 
  randomization_accept_prob = 0.0001)
CandRandomizationsPackage$candidate_randomizations$shape
CandRandomizationsPackage$M_candidate_randomizations$shape

# You can coerce candidate_randomizations_array into R like this:
candidate_randomizations <- np$array( CandRandomizationsPackage$candidate_randomizations )
dim( candidate_randomizations )

# We can also use `fastrerandomize` to perform a randomization test using those acceptable randomizations.
# Setup simulated outcome data
CoefY <- rnorm(ncol(X))
Wobs <- candidate_randomizations[1,]
tau_true <- 1
Yobs <- c(X %*% as.matrix(CoefY) + Wobs*tau_true + rnorm(n_units, sd = 0.1))

# Perform exact randomization set based on accepted randomizations
ExactRandomizationTestResults <- fastrerandomize::randomization_test(
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
starting_value = abs( min(log( 100/n_randomizations, base = 10 ), log(0.05,base=10)))
prob_accept_randomization_seq <- 10^(- seq(from = starting_value, to = 1/9, length.out = 32L ) )

# perform pre-design analysis 
PreAnalysisEvaluation <- fastrerandomize::randomization_test(
  X = X,
  randomization_accept_prob = prob_accept_randomization_seq,
  prior_treatment_effect_mean = 0.1,
  prior_treatment_effect_SD = 1,
  randomization_type = "exact", 
  coef_prior = function(){rnorm(ncol(X), sd = 1)},
  simulate = T
)

# suggested acceptance threshold minimizing a priori expected p-value
PreAnalysisEvaluation$suggested_randomization_accept_prob

# expected p-values for all values of randomization_accept_prob input
PreAnalysisEvaluation$p_value

# final print message
print(  "Agricultural experiment tutorial complete! Timing details:"  )
print(  Sys.time() - t0  )
}
