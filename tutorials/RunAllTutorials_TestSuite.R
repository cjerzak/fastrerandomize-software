#!/usr/bin/env Rscript
{
  ##########################################
  # code for testing most functionalities of fastrerandomize on your hardware
  ##########################################
  tryTests <- try({
    # remote install latest version of the package
    # devtools::install_github(repo = "cjerzak/fastrerandomize-software/fastrerandomize")

    # local install for development team
    # install.packages("~/Documents/fastrerandomize-software/fastrerandomize",repos = NULL, type = "source",force = F)

    print("Starting ag experiment tutorial..."); setwd("~");
    t_ <- try(source("~/Documents/fastrerandomize-software/tutorials/AgExperiment_Tutorial.R"),T)
    if("try-error" %in% class(t_)){ stop("Failed at Ag experiment tutorial...") }

    Sys.sleep(1L);print("Starting geo-rerandomization tutorial..."); setwd("~");
    t_ <- try(source("~/Documents/fastrerandomize-software/tutorials/GeoRR_Tutorial.R"),T)
    if("try-error" %in% class(t_)){ stop("Failed at geo-rerandomization tutorial...") }
  }, T)

  if('try-error' %in% class(tryTests)){ print("At least one test failed..."); print( tryTests ) }
  if(!'try-error' %in% class(tryTests)){ print("All tests succeeded!") }
}
