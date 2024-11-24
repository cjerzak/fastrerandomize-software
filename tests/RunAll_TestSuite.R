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
    
    for(findFI in c(FALSE, TRUE)){ 
    for(type_ in c("exact","monte_carlo")){ 
      fastrerandomize::print2(sprintf("On type: %s", type_))
      t_GetSet <- try({
        RandomizationSet_ <- fastrerandomize::generate_randomizations(
          n_units = 20,
          n_treated = 10,
          X = X,
          randomization_accept_prob=0.1,
          randomization_type=type_,
          max_draws=1000)
      },T)
      if("try-error" %in% class(t_GetSet)){ stop(sprintf("Failed at t_GetSet: %s...",type_)) }
      
      t_RRTest <- try({
        RRTest_ <- fastrerandomize::randomization_test(
          obsW = (W_<-as.integer(np$array(RandomizationSet_[1,]))),
          obsY = rnorm(RandomizationSet_$shape[[2]])+2*W_,
          candidate_randomizations_array = RandomizationSet_,
          findFI = findFI, 
          X = X)
        if(!is.null(RRTest_$FI)){ 
          print(sprintf("FI: {%s}", paste(round(RRTest_$FI,2),collapse = ", " )))
        }
      },T)
      if("try-error" %in% class(t_RRTest)){ stop(sprintf("Failed at t_RRTest: [%s, %s]...",type_, findFI)) }
    }
    }
  }, T)
    
  if('try-error' %in% class(tryTests)){ fastrerandomize::print2("At least one test failed..."); print( tryTests ) }
  if(!'try-error' %in% class(tryTests)){ fastrerandomize::print2("All tests succeeded!") }
}
