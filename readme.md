# `fastrerandomize`: An R Package for Ultra-fast Rerandomization Using a JAX Backend

[**What is `fastrerandomize`?**](#description)
| [**Installation**](#installation)
| [**Tutorial**](#tutorial)
| [**Data**](#data)
| [**References**](#references)
| [**Documentation**](https://github.com/cjerzak/fastrerandomize-software/blob/main/fastrerandomize.pdf)

*Note:* `fastrerandomize` has been successfully tested on CPU, CUDA, and METAL frameworks. 

# What is `fastrerandomize`?<a id="description"></a>
The `fastrerandomize` contains functions such as `RandomizationTest`, which offers a streamlined approach for performing randomization tests after using rerandomization in the research design. 

We employ a [JAX backend](https://en.wikipedia.org/wiki/Google_JAX) to make exact rerandomization inference possible even for larger experiments where the number of randomizations is in the hundreds of millions. In future releases, we will employ improved memory footprint handling to handle even larger cases where the candidate randomization set ranges in the billions. 

# Package Installation and Loading <a id="installation"></a>
```
# Install devtools if needed 
# install.packages("devtools")

# Install fastrerandomize if you haven't already
# devtools::install_github(repo = "cjerzak/fastrerandomize-software/fastrerandomize")

# Load the package
library(  fastrerandomize  )  

# Before running any code, you'll need to initialize the JAX environment 
fastrerandomize::InitializeJAX(conda_env = "tensorflow_m1", conda_env_required = T)

# If JAX is not in a conda environment, try: 
# fastrerandomize::InitializeJAX(conda_env = NULL)

# Note: If you leave `conda_env = NULL`, we will search in the default Python environment for JAX.
```

## Installing JAX 
In order to download the JAX backend in a conda environment, see [\[this link\]](https://jax.readthedocs.io/en/latest/installation.html) from the JAX developers.

You can also try downloading `jax` and `jaxlib` via `reticulate` package from within `R`:
```
library(reticulate)
py_install("jax") # install JAX in the default Python environment 
py_install("jaxlib") # install jaxlib in the default Python environment 

# now, try restarting your R session and running: 
fastrerandomize::InitializeJAX(conda_env = NULL)
```
As noted, it often helps to re-start your `R` session when debugging package installs. Note that this package has been tested successfully on Windows and Apple machines and should run well on Linux. If you're having trouble installing `reticulate`, try installing the latest version of `R`. 

# Tutorial<a id="tutorial"></a>
Let's get started with a tutorial. We're first going to use the package for generate a pool of acceptable rerandomizations. 
```
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
```
We can also use `fastrerandomize` to perform a randomization test using those acceptable randomizations. 
```
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
```
You can also use `fastrerandomize` to perform a pre-analysis evaluation for what would be an _a priori_ optimal rerandomization acceptance threshold in terms of minimizing the expected exact _p_-value.
```
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
```
Currently, we support non-approximate tests and randomization generations for $n \leq 30$ (where the total number of available complete randomizations is about 155 million). In the future, we plan to increase this by both approximations and smart memory handling of rejected randomizations. 

# Replication Data<a id="data"></a>
Replication data for the package is available using the `data` command. 
```
data( fastrerandomize )
```

# Development 
We welcome new features or bug fixes (you can raise an issue or submit a pull request in the repository). We will keep the package up-to-date with the latest version of the [JAX backend](https://en.wikipedia.org/wiki/Google_JAX). 

# References<a id="references"></a>
Connor T. Jerzak and Rebecca Goldstein. "Degrees of Randomness in Rerandomization Procedures." *ArXiv Preprint*, 2023. [\[PDF\]](https://arxiv.org/pdf/2310.00861.pdf)
```
@article{JerGol2023,
         title={Degrees of Randomness in Rerandomization Procedures},
         author={Jerzak, Connor T. and Rebecca Goldstein},
         journal={ArXiv Preprint},
         year={2023}}
```
