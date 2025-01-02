{
  #######################################
  # AgExperiment_Tutorial.R - an agriculture experiment tutorial
  #######################################
  
  # NOTE: If you haven't installed or set up the package:
  #  devtools::install_github("cjerzak/fastrerandomize-software/fastrerandomize")
  #  Then the Python backend (if needed, done once):
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
  
  # 4a. Print the object (will call print.fastrerandomize_randomization)
  print(CandRandomizations)
  
  # 4b. Show a summary (will call summary.fastrerandomize_randomization)
  summary(CandRandomizations)
  
  # 4c. Plot the balance distribution if available (plot.fastrerandomize_randomization)
  plot(CandRandomizations)
  
  # 5. Because it's an S3 object, the randomizations themselves are stored in:
  #      CandRandomizations$randomizations
  #    If you want the underlying jax/numpy shape (instead of R's dim()):
  if (!is.null(CandRandomizations$randomizations$shape)) {
    cat("\nShape of randomizations in Python/jax sense:\n")
    print(CandRandomizations$randomizations$shape)
  }
  
  # 6. If a balance vector was stored, its shape (or length) can be shown similarly:
  if (!is.null(CandRandomizations$balance)) {
    cat("\nLength of balance vector:\n")
    print(length(CandRandomizations$balance))
  }
  
  # 7. Convert to base R (matrix) if you need to manipulate randomizations in pure R
  candidate_randomizations <- NULL
  if (!is.null(CandRandomizations$randomizations)) {
    candidate_randomizations <- as.matrix(fastrerandomize::np$array(CandRandomizations$randomizations))
    cat("\nDimensions of randomizations in R:\n")
    print(dim(candidate_randomizations))
  }
  
  # -------------------------------------------------------------------
  # 8. Randomization Test (optionally using these randomizations)
  fastrerandomize::print2("Starting randomization test...")
  
  #    Setup simulated outcome data
  CoefY <- rnorm(ncol(X))
  if (is.null(candidate_randomizations)) {
    # fallback if something didn't generate
    # otherwise just use candidate_randomizations from above
    cat("Warning: candidate_randomizations is NULL, generating a simple Wobs\n")
    Wobs <- c(rep(1, n_treated), rep(0, n_units - n_treated))
  } else {
    Wobs <- candidate_randomizations[1, ]  # pick the first acceptable randomization
  }
  tau_true <- 1
  Yobs <- c(X %*% as.matrix(CoefY) + Wobs * tau_true + rnorm(n_units, sd = 0.1))
  
  ExactRandomizationTestResults <- fastrerandomize::randomization_test(
    obsW = Wobs,
    obsY = Yobs,
    candidate_randomizations = candidate_randomizations,  # pure R matrix is fine
    findFI = FALSE
  )
  cat("\n--- Randomization test results ---\n")
  cat("P-value:     ", ExactRandomizationTestResults$p_value, "\n")
  cat("Tau (D-in-M):", ExactRandomizationTestResults$tau_obs, "\n")
  
  cat("\nAgricultural experiment tutorial complete!\n")

}
