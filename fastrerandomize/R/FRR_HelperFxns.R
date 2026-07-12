#' Print timestamped messages with optional quieting
#'
#' This function prints messages prefixed with the current timestamp in a standardized format.
#' Messages can be suppressed using the quiet parameter.
#'
#' @param text A character string containing the message to be printed.
#' @param quiet A logical value indicating whether to suppress output. Default is \code{FALSE}. 
#'
#' @return No return value, called for side effect of printing with timestamp. 
#'
#' @examples
#' # Print a basic message with timestamp
#' print2("Processing started")
#'
#' # Suppress output
#' print2("This won't show", quiet = TRUE)
#'
#' # Use in a loop
#' for(i in 1:3) {
#'   print2(sprintf("Processing item %d", i))
#' }
#'
#' @details
#' The function prepends the current timestamp in "YYYY-MM-DD HH:MM:SS" format
#' to the provided message. 
#'
#' @seealso
#' \code{Sys.time} for the underlying timestamp functionality. 
#'
#' @export
#' @md
print2 <- function(text, 
                   quiet = FALSE){
  if(!quiet){
    print( sprintf("[%s] %s" ,format(Sys.time(), "%Y-%m-%d %H:%M:%S"),text) )
  }
}

#' Check if 'Python' and 'JAX' are available
#'
#' This function checks if 'Python' and 'JAX' can be accessed via `reticulate`. If not,
#' it returns `NULL` and prints a message suggesting to run `build_backend()`.
#'
#' @param conda_env A character string specifying the name of the conda environment. 
#'   Default is `"fastrerandomize_env"`.
#' @param conda The path to a conda executable, or `"auto"`. Default is `"auto"`.
#'
#' @return Returns `TRUE` (invisibly) if both 'Python' and 'JAX' are available; otherwise returns `NULL`.
#'
#' @examples
#' \dontrun{
#'   check_jax_availability()
#' }
#'
#' @export
check_jax_availability <- function(conda_env = "fastrerandomize_env", 
                                   conda = "auto"){
  
  # Try to use the specified conda environment
  try_condaenv <- try(reticulate::use_condaenv(conda_env, 
                                               required = TRUE, 
                                               conda = conda), T)
  if("try-error" %in% class(try_condaenv)){
    message("conda environment is not available. Please install Python/conda and build the backend using ",
            "fastrerandomize::build_backend(conda_env = '", conda_env, "', conda = '", conda, "').")
    return(NULL)
  }
  
  # Check if Python is available
  if(!reticulate::py_available(initialize = TRUE)){
    message("Python is not available. Please install Python/conda and build the backend using ",
            "fastrerandomize::build_backend(conda_env = '", conda_env, "', conda = '", conda, "').")
    return(NULL)
  }
  
  # Check if 'JAX' is installed
  if(!reticulate::py_module_available("jax")){
    message("JAX is not installed. Please build the backend using ",
            "fastrerandomize::build_backend(conda_env = '", conda_env, "', conda = '", conda, "').")
    return(NULL)
  }
  
  # If we reach this point, both Python and JAX are accessible
  invisible(TRUE)
}

.is_metal_backend <- function(device_string = NULL) {
  if (is.null(device_string)) {
    device_string <- reticulate::py_str(fastrr_env$jax$devices()[[1]])
  }

  isTRUE(grepl("METAL", toupper(as.character(device_string)), fixed = TRUE))
}

.warn_if_large_exact <- function(n_combinations, threshold = 1e6) {
  if (is.finite(n_combinations) && n_combinations <= threshold) {
    return(invisible(FALSE))
  }

  warning(
    sprintf(
      paste0(
        "Exact randomization is requested, but that is %s combinations. ",
        "This may be infeasible in terms of memory/time. Consider Monte Carlo instead."
      ),
      format(n_combinations, big.mark = ",", scientific = FALSE)
    ),
    immediate. = TRUE
  )
  invisible(TRUE)
}

.fi_initial_bounds <- function(tau_obs) {
  tau_obs <- as.numeric(tau_obs)[1]
  bound_range <- max(3 * abs(tau_obs), 1)
  c(tau_obs - bound_range, tau_obs + bound_range)
}

.fi_step_multiplier <- function(c_initial) {
  if (length(c_initial) != 1L || !is.numeric(c_initial) ||
      !is.finite(c_initial) || c_initial <= 0) {
    stop("'c_initial' must be a finite positive number.", call. = FALSE)
  }
  as.numeric(c_initial) / 2
}

.fi_search_grid <- function(bounds, tau_obs, length.out = 100L) {
  bounds <- sort(as.numeric(bounds))
  fi_range <- abs(diff(bounds))
  expansion <- max(fi_range * 0.5, abs(as.numeric(tau_obs)[1]) * 0.5, 1)
  seq(bounds[1] - expansion, bounds[2] + expansion, length.out = length.out)
}

.fi_accepted_range <- function(tau_seq, tail_probabilities, alpha) {
  accepted <- is.finite(tail_probabilities) & tail_probabilities > alpha / 2
  if (!any(accepted)) {
    return(c(NA_real_, NA_real_))
  }
  range(tau_seq[accepted])
}

.resolve_jax_seed <- function(seed = NULL) {
  if (is.null(seed)) {
    return(sample.int(.Machine$integer.max, size = 1L))
  }

  if (length(seed) != 1L || !is.numeric(seed) || !is.finite(seed) ||
      seed < 0 || seed > .Machine$integer.max || seed != floor(seed)) {
    stop(
      sprintf("'seed' must be a whole number between 0 and %s.", .Machine$integer.max),
      call. = FALSE
    )
  }
  as.integer(seed)
}

output2output <- function(x, return_type = "R"){
  if (is.null(x)) { return(NULL) }
  if(return_type == "R"){ return( fastrr_env$np$array(x) )  }
  if(return_type == "jax"){ return( x ) }
}

.onUnload <- function(libpath) {
  # Clean up temp files
  temp_files <- list.files(tempdir(), pattern = "^fastrerandomize_tmp", full.names = TRUE)
  unlink(temp_files, recursive = TRUE)
}

fastrr_env <- new.env( parent = emptyenv() )
