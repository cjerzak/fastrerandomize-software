{
options(error = NULL)
#######################################
# AgExperiment_Tutorial.R - an agriculture experiment tutorial
#######################################  
  
# Install fastrerandomize if you haven't already (install devtools if needed via install.packages("devtools"))
# devtools::install_github(repo = "cjerzak/fastrerandomize-software/fastrerandomize")

# local install for development team
# install.packages("~/Documents/fastrerandomize-software/fastrerandomize",repos = NULL, type = "source",force = F)
  
# build backend if needed
# fastrerandomize::build_backend()

# First, specify some analysis parameters
n_units <- 22L; n_treated <- 12L

# Generate covariate data
X <- matrix(rnorm(n_units*5),nrow = n_units)

fastrerandomize::print2("Generating a set of acceptable randomizations based randomization_accept_prob...") 
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

fastrerandomize::print2("Starting randomization test...") 
ExactRandomizationTestResults <- fastrerandomize::randomization_test(
  obsW = Wobs,
  obsY = Yobs,
  candidate_randomizations = candidate_randomizations,
  findFI = F # set to T if an exact fiducial interval needed
)
ExactRandomizationTestResults$p_value # p-value
ExactRandomizationTestResults$tau_obs # difference-in-means ATE estimate

print(  "Agricultural experiment tutorial complete!"  )
}
