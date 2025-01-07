#' Print timestamped messages with optional quieting
#'
#' This function prints messages prefixed with the current timestamp in a standardized format.
#' Messages can be suppressed using the quiet parameter.
#'
#' @param text A character string containing the message to be printed
#' @param quiet A logical value indicating whether to suppress output. Default is FALSE
#'
#' @return No return value, called for side effect of printing
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
#' \code{Sys.time} for the underlying timestamp functionality
#'
#' @export
#' @md
print2 <- function(text, 
                   quiet = FALSE){
  if(!quiet){
    print( sprintf("[%s] %s" ,format(Sys.time(), "%Y-%m-%d %H:%M:%S"),text) )
  }
}

output2output <- function(x, return_type = "R"){ 
  if(return_type == "R"){ return( fastrr_env$np$array(x) )  }
  if(return_type == "jax"){ return( x ) }
}

.onUnload <- function(libpath) {
  # Clean up temp files
  temp_files <- list.files(tempdir(), pattern = "^tmp", full.names = TRUE)
  unlink(temp_files, recursive = TRUE)
}

fastrr_env <- new.env( parent = emptyenv() )