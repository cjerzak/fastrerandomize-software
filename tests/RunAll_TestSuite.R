#!/usr/bin/env Rscript
{
  ##########################################
  # code for testing functionalities of fastrerandomize on your hardware
  ##########################################
  tryTests <- try({
    # remote install latest version of the package
    # devtools::install_github(repo = "cjerzak/fastrerandomize-software/fastrerandomize")

    # local install for development team
    # install.packages("~/Documents/fastrerandomize-software/fastrerandomize",repos = NULL, type = "source",force = F)
    
    options(error = NULL); t_Initialize <- try({
      fastrerandomize::InitializeJAX(conda_env = "jax_cpu", conda_env_required = T)
    },T)
    if("try-error" %in% class(t_Initialize)){ stop("Failed at t_Initialize...") }
    
    t_GenData <- try({
      set.seed(999L, kind = "Wichmann-Hill")
      X <- matrix(rnorm(20*5), 20, 5)
    },T)
    if("try-error" %in% class(t_GenData)){ stop("Failed at t_GenData...") }
    
    RandomizationSet_ <- fastrerandomize::GenerateRandomizations(
      n_units = 20,
      n_treated = 10,
      X = X,
      randomization_accept_prob=0.1,
      randomization_type="exact",
      max_draws=1000)
    
    
    for(type_ in c("exact","monte_carlo")){ 
      print(sprintf("On type: %s", type_))
      t_GetSet <- try({
        RandomizationSet_ <- fastrerandomize::GenerateRandomizations(
          n_units = 20,
          n_treated = 10,
          X = X,
          randomization_accept_prob=0.1,
          randomization_type=type_,
          max_draws=1000)
      },T)
      if("try-error" %in% class(t_GetSet)){ stop(sprintf("Failed at t_GetSet: %s...",type_)) }
      
      t_RRTest <- try({
        RandomizationSet_ <- fastrerandomize::RandomizationTest(
          obsW = RandomizationSet_[1,],
          obsY = rnorm(nrow(RandomizationSet_)),
          candidate_randomizations = RandomizationSet_,
          X = X,
          max_draws=1000)
      },T)
      if("try-error" %in% class(t_RRTest)){ stop(sprintf("Failed at t_RRTest: %s...",type_)) }
    }
  }, T)
    
  if('try-error' %in% class(tryTests)){ print("At least one test failed..."); print( tryTests ) }
  if(!'try-error' %in% class(tryTests)){ print("All tests succeeded!") }
}
