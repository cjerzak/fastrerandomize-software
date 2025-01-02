{
  #######################################
  # AgExperiment_Tutorial.R - an agriculture experiment tutorial
  #######################################
  
  # NOTE: If you haven't installed or set up the package:
  #  devtools::install_github("cjerzak/fastrerandomize-software/fastrerandomize")
  #  If needed (done once), install the JAX backend:
  #       fastrerandomize::build_backend()
  
  options(error = NULL)
  
  # 1. Analysis parameters
  n_units   <- 22L
  n_treated <- 12L
  
  # 2. Generate covariate data
  X <- matrix(rnorm(n_units * 5), nrow = n_units)
  
  fastrerandomize::print2("Generating a set of acceptable randomizations based on randomization_accept_prob...")
  
  # 3. Generate randomizations
  #    - 'generate_randomizations' now returns an S3 object of class 'fastrerandomize_randomization'
  #    - randomization_type can be "exact" or "monte_carlo"
  #    - Adjust 'max_draws' or 'batch_size' as needed
  CandRandomizations <- fastrerandomize::generate_randomizations(
    n_units = n_units,
    n_treated = n_treated,
    X = X,
    randomization_type = "exact", 
    max_draws = 10000L,
    # randomization_type = "monte_carlo", max_draws = 50000L, batch_size = 1000L,
    randomization_accept_prob = 0.0001
  )
  
  # --- Demonstrate S3 usage ---
  cat("\n--- S3 method usage demo ---\n\n")
  
  # 4a. Print the object (will call print.fastrerandomize_instance)
  print(CandRandomizations)
  
  # 4b. Show a summary (will call summary.fastrerandomize_instance)
  summary(CandRandomizations)
  
  # 4c. Plot the balance distribution if available (plot.fastrerandomize_instance)
  plot(CandRandomizations)
  
  # -------------------------------------------------------------------
  # 5. Randomization Test
  fastrerandomize::print2("Starting randomization test...")
  
  # 5a. Setup simulated outcome data
  CoefY <- rnorm(ncol(X))
  Wobs <- CandRandomizations$randomizations[1, ]  # pick the first acceptable randomization
  tau_true <- 1
  Yobs <- c(X %*% as.matrix(CoefY) + Wobs * tau_true + rnorm(n_units, sd = 0.1))
  
  # 5b. Perform randomization test 
  randomization_test_results <- fastrerandomize::randomization_test(
    obsW = Wobs,
    obsY = Yobs,
    candidate_randomizations = CandRandomizations$randomizations,  # pure R matrix is fine
    findFI = TRUE
  )
  
  cat("\n--- Randomization test results ---\n")
  print( randomization_test_results )       
  summary( randomization_test_results )     
  plot( randomization_test_results )     
  
  cat("\nAgricultural experiment tutorial complete!\n")
}
