#!/usr/bin/env Rscript
#' Initialize JAX
#'
#' @usage
#'
#' InitializeJax(conda_env, conda_env_required)
#'
#' @param conda_env A character string representing the conda environment to activate. A version of JAX should live in that environment.
#' @param conda_env_required A logical representing whether to force use the specified conda environment.
#'
#' @return This function initializes a JAX-containing conda environment as specified by `conda_env`. This function must be run before any others in `fastrerandomize`.
#'
#' @examples
#' # For a tutorial, see
#' # github.com/cjerzak/fastrerandomization-software
#'
#' @export
#' @md

InitializeJAX <- function(conda_env, conda_env_required){
  library(fastmatch);
  library(reticulate)
  reticulate::use_condaenv(condaenv = conda_env, required = conda_env_required)

  #print(   tf$config$list_physical_devices('GPU')  )
  #tfFloatType <- tf$float64
  #tf$keras$backend$set_floatx('float64') #default backend

  #tf setup - set seed
  #tf$random$set_seed(as.integer(round(runif(1,min=0,max=10^6),0))); print(  tf$version$VERSION )
  #tfp <- tf_probability();tfd <-tfp$distributions;tfb <-tfp$bijectors

  # setup other r packages
  Sys.sleep(2L)
  jax <<- reticulate::import("jax")
  jnp <<- reticulate::import("jax.numpy")
  np <<- reticulate::import("numpy")

  # enable 64 bit computations
  jax$config$update("jax_enable_x64", FALSE); jaxFloatType <<- jnp$float32 # use float64
  #jaxFloatType <- jnp$float32 # # use float32

  #jax$config$update('jax_platform_name', 'cpu')
  #optax <- reticulate::import("optax")
  #eq <- reticulate::import("equinox")
  #diffrax <- reticulate::import("diffrax")
  #oryx <- reticulate::import("oryx")
  #py_gc <- reticulate::import("gc")
  JaxKey <<- function(int_){ jax$random$PRNGKey(int_)}
  SoftPlus_r <<- function(x){ log(exp(x)+1) }

  LinearizeNestedList <<- function (NList, LinearizeDataFrames = FALSE, NameSep = "/",
                                   ForceNames = FALSE)
  {
    stopifnot(is.character(NameSep), length(NameSep) == 1)
    stopifnot(is.logical(LinearizeDataFrames), length(LinearizeDataFrames) ==
                1)
    stopifnot(is.logical(ForceNames), length(ForceNames) == 1)
    if (!is.list(NList))
      return(NList)
    if (is.null(names(NList)) | ForceNames == TRUE)
      names(NList) <- as.character(1:length(NList))
    if (is.data.frame(NList) & LinearizeDataFrames == FALSE)
      return(NList)
    if (is.data.frame(NList) & LinearizeDataFrames == TRUE)
      return(as.list(NList))
    A <- 1
    B <- length(NList)
    while (A <= B) {
      Element <- NList[[A]]
      EName <- names(NList)[A]
      if (is.list(Element)) {
        Before <- if (A == 1)
          NULL
        else NList[1:(A - 1)]
        After <- if (A == B)
          NULL
        else NList[(A + 1):B]
        if (is.data.frame(Element)) {
          if (LinearizeDataFrames == TRUE) {
            Jump <- length(Element)
            NList[[A]] <- NULL
            if (is.null(names(Element)) | ForceNames ==
                TRUE)
              names(Element) <- as.character(1:length(Element))
            Element <- as.list(Element)
            names(Element) <- paste(EName, names(Element),
                                    sep = NameSep)
            NList <- c(Before, Element, After)
          }
          Jump <- 1
        }
        else {
          NList[[A]] <- NULL
          if (is.null(names(Element)) | ForceNames == TRUE)
            names(Element) <- as.character(1:length(Element))
          Element <- LinearizeNestedList(Element, LinearizeDataFrames,
                                         NameSep, ForceNames)
          names(Element) <- paste(EName, names(Element),
                                  sep = NameSep)
          Jump <- length(Element)
          NList <- c(Before, Element, After)
        }
      }
      else {
        Jump <- 1
      }
      A <- A + Jump
      B <- length(NList)
    }
    return(NList)
  }
}
