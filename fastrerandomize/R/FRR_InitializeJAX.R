initialize_jax <- function(conda_env = "fastrerandomize", 
                           conda_env_required = TRUE) {
  print2("Loading fastrerandomize environment...")
  
  # We assume `print2` is already in your package with no global assignment, 
  # or is just standard `message()` if you prefer.
  
  # Load reticulate (Declared in Imports: in DESCRIPTION)
  reticulate::use_condaenv(condaenv = conda_env, required = conda_env_required)
  
  # Import Python packages once, storing them in the env
  if (!exists("jax", envir = env, inherits = FALSE)) {
    fastrr_env$jax <- reticulate::import("jax", convert = FALSE)
  }
  if (!exists("jnp", envir = env, inherits = FALSE)) {
    fastrr_env$jnp <- reticulate::import("jax.numpy", convert = FALSE)
  }
  if (!exists("np", envir = env, inherits = FALSE)) {
    fastrr_env$np  <- reticulate::import("numpy", convert = FALSE)
  }
  if (!exists("py_gc", envir = env, inherits = FALSE)) {
    fastrr_env$py_gc <- reticulate::import("gc", convert = FALSE)
  }
  
  # Disable 64-bit computations
  fastrr_env$jax$config$update("jax_enable_x64", FALSE)
  fastrr_env$jaxFloatType <- fastrr_env$jnp$float32
  
  print2("Success loading JAX!")
  
  # 4. Setup your core JAX functions and store them in env
  print2("Setup of core JAX functions...")
  
  # For example:
  fastrr_env$expand_grid_JAX <- function(n_treated, n_control) {
    # Code that uses fastrr_env$jnp instead of a global jnp
    # ...
    # e.g.  something like:
    # grid <- fastrr_env$jnp$meshgrid(...)
    # return(grid)
  }
  
  # A JIT-compiled function
  fastrr_env$InsertOnes <- fastrr_env$jax$jit(
    function(treat_indices_, zeros_) {
      zeros_ <- zeros_$at[treat_indices_]$add(1L)
      zeros_
    }
  )
  
  fastrr_env$InsertOnesVectorized <- fastrr_env$jax$jit(
    fastrr_env$jax$vmap(
      function(treat_indices_, zeros_) {
        fastrr_env$InsertOnes(treat_indices_, zeros_)
      },
      in_axes = list(1L, NULL)
    )
  )
  
  # Example for FastDiffInMeans
  fastrr_env$FastDiffInMeans_R <- function(y_, w_, n0, n1) {
    my1 <- fastrr_env$jnp$divide(fastrr_env$jnp$sum(fastrr_env$jnp$multiply(y_, w_)), n1)
    my0 <- fastrr_env$jnp$divide(fastrr_env$jnp$sum(fastrr_env$jnp$multiply(y_, fastrr_env$jnp$subtract(1, w_))), n0)
    fastrr_env$jnp$subtract(my1, my0)
  }
  fastrr_env$FastDiffInMeans <- fastrr_env$jax$jit(fastrr_env$FastDiffInMeans_R)
  
  # ... Continue defining the rest, each stored in env
  # e.g. fastrr_env$VectorizedFastHotel2T2 <- fastrr_env$jax$jit(...)
  
  print2("Success setting up core JAX functions!")
  
  # 5. Return invisible so user doesn’t see repeated messages, if desired
  invisible(TRUE)
}
