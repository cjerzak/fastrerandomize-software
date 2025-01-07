#' An optional function to build the environment for fastrerandomize. Builds a conda environment in which jax and np are installed. Users can also create a conda environment where jax and np are installed themselves. 
#'
#' @param conda_env (default = `"fastrerandomize"`) Name of the conda environment in which to place the backends.
#' @param conda (default = `auto`) The path to a conda executable. Using `"auto"` allows reticulate to attempt to automatically find an appropriate conda binary.

#' @return Builds the computational environment for `fastrerandomize`. This function requires an Internet connection.
#' You may find out a list of conda Python paths via: `system("which python")`
#'
#' @examples
#' # For a tutorial, see
#' # github.com/cjerzak/fastrerandomize-software/
#'
#' @export
#' @md

build_backend <- function(conda_env = "fastrerandomize", conda = "auto"){
  # Create a new conda environment
  reticulate::conda_create(envname = conda_env,
                           conda = conda,
                           python_version = "3.11")
  
  # Install Python packages within the environment
  Packages2Install <- c("numpy==1.26.4",
                        "jax==0.4.26",
                        "jaxlib==0.4.26")
  
  # Install METAL where available 
  if( Sys.info()["machine"] == "arm64" & Sys.info()["sysname"] == "Darwin" ){
    Packages2Install <- c(Packages2Install,"jax-metal==0.1.0")
  }
  for(pack_ in Packages2Install){
      try_ <- try(reticulate::py_install(pack_, conda = conda, pip = TRUE, envname = conda_env), TRUE)
  }
  print("Done building fastrerandomize backend!")
}