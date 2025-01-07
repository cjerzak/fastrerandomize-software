{
  rm(list=ls()); options(error = NULL)
  # set path and specify package name
  setwd(sprintf("~/Documents/%s-software",
                package_name <- "fastrerandomize"))
  
  package_path <- sprintf("~/Documents/%s-software/%s",package_name,package_name)
  tools::add_datalist(package_path, force = TRUE, small.size = 1L)
  devtools::build_vignettes(package_path)
  devtools::document(package_path)
  
  # remove old PDF
  try(file.remove(sprintf("./%s.pdf",package_name)),T)
  
  # create new PDF
  system(sprintf("R CMD Rd2pdf %s",package_path))
  
  # install.packages( sprintf("~/Documents/%s-software/%s",package_name,package_name),repos = NULL, type = "source")
  # library( causalimages ); data(  CausalImagesTutorialData )
  log(sort( sapply(ls(),function(l_){object.size(eval(parse(text=l_)))})))
  
  # Check package to ensure it meets CRAN standards.
  # devtools::check( package_path )
}
