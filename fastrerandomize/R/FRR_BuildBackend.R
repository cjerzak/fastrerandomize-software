#' A function to build the environment for fastrerandomize. Builds a conda environment in which 'JAX' and 'np' are installed. Users can also create a conda environment where 'JAX' and 'np' are installed themselves. 
#'
#' @param conda_env (default = `"fastrerandomize_env"`) Name of the conda environment in which to place the backends.
#' @param conda (default = `auto`) The path to a conda executable. Using `"auto"` allows reticulate to attempt to automatically find an appropriate conda binary.

#' @return Invisibly returns NULL; this function is used for its side effects 
#' of creating and configuring a conda environment for `fastrerandomize`. 
#' This function requires an Internet connection.
#' You can find out a list of conda Python paths via: `Sys.which("python")`
#'
#' @examples
#' \dontrun{
#' # Create a conda environment named "fastrerandomize_env"
#' # and install the required Python packages (jax, numpy, etc.)
#' build_backend(conda_env = "fastrerandomize_env", conda = "auto")
#'
#' # If you want to specify a particular conda path:
#' # build_backend(conda_env = "fastrerandomize_env", conda = "/usr/local/bin/conda")
#' }
#'
#' @export
#' @md

build_backend <- function(conda_env = "fastrerandomize_env", conda = "auto"){
  # Create a new conda environment
  reticulate::conda_create(envname = conda_env,
                           conda = conda,
                           python_version = "3.13")
  
  os <- Sys.info()[["sysname"]]
  machine <- Sys.info()["machine"]
  msg <- function(...) message(sprintf(...))
  
  pip_install <- function(pkgs, ...) {
    reticulate::py_install(packages = pkgs, envname = conda_env, conda = conda, pip = TRUE, ...)
    TRUE
  }
  
  # Install numpy first (pinned version)
  pip_install("numpy")
  
  # --- Install JAX with appropriate backend ---
  install_jax <- function() {
    if (os == "Darwin" && machine == "arm64") {
      msg("Apple Silicon detected: installing JAX 0.5.0 with Metal support.")
      pip_install(c("jax==0.5.0", "jaxlib==0.5.0", "jax-metal==0.1.1"))
    } else if (identical(os, "Linux")) {
      # Read driver version as integer major (e.g., 580)
      drv <- try(suppressWarnings(system("nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1", intern=TRUE)), TRUE)
      drv_major <- suppressWarnings(as.integer(sub("^([0-9]+).*", "\\1", drv[1])))
      
      # Prefer CUDA 13 if the driver is new enough; otherwise CUDA 12; else CPU fallback
      if (!is.na(drv_major) && drv_major >= 580) {
        msg("Driver %s detected (>=580): installing JAX CUDA 13 wheels.", drv[1])
        tryCatch(pip_install('jax[cuda13]'), error = function(e) {
          msg("CUDA 13 wheels failed (%s); falling back to CUDA 12.", e$message)
          pip_install('jax[cuda12]')
        })
      } else if (!is.na(drv_major) && drv_major >= 525) {
        msg("Driver %s detected (>=525,<580): installing JAX CUDA 12 wheels.", drv[1])
        pip_install('jax[cuda12]')
      } else {
        msg("Driver %s too old for CUDA wheels; installing CPU-only JAX.", drv[1])
        pip_install('jax')
      }
    } else {
      msg("Installing CPU-only JAX.")
      pip_install('jax')
    }
  }
  
  # Install JAX first (so later deps don't pull a CPU variant)
  install_jax()
  
  # (Optional) neutralize LD_LIBRARY_PATH inside this env to prevent overrides
  if (os == "Linux") {
    try({
      actdir <- file.path(Sys.getenv("HOME"), "miniconda3/envs", conda_env, "etc", "conda", "activate.d")
      dir.create(actdir, recursive = TRUE, showWarnings = FALSE)
      writeLines("unset LD_LIBRARY_PATH", file.path(actdir, "00-unset-ld.sh"))
    }, silent = TRUE)
  }
  
  msg("Environment '%s' is ready.", conda_env)
}