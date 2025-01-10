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
#'   Default is `"fastrerandomize"`.
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
check_jax_availability <- function(conda_env = "fastrerandomize", 
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

output2output <- function(x, return_type = "R"){ 
  if(return_type == "R"){ return( fastrr_env$np$array(x) )  }
  if(return_type == "jax"){ return( x ) }
}

.onUnload <- function(libpath) {
  # Clean up temp files
  temp_files <- list.files(tempdir(), pattern = "^fastrerandomize_tmp", full.names = TRUE)
  unlink(temp_files, recursive = TRUE)
}

fastrr_env <- new.env( parent = emptyenv() )