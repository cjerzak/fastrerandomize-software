# Geo-rerandomization tutorial
{
  options(error=NULL)
  
  # Install fastrerandomize and causalimages if you haven't already
  # devtools::install_github(repo = "cjerzak/fastrerandomize-software/fastrerandomize")
  # devtools::install_github(repo = "cjerzak/causalimages-software/causalimages")

  # Local install for development team
  # install.packages("~/Documents/causalimages-software/causalimages",repos = NULL, type = "source",force = F)
  # install.packages("~/Documents/fastrerandomize-software/fastrerandomize",repos = NULL, type = "source",force = F)

  # build backend you haven't ready:
  # causalimages::BuildBackend()

  # run code if downloading data for the first time
  download_folder <- "~/Downloads/UgandaAnalysis.zip"
  if( reDownloadRawData <- F  ){
    # specify uganda data URL
    uganda_data_url <- "https://dl.dropboxusercontent.com/s/xy8xvva4i46di9d/Public%20Replication%20Data%2C%20YOP%20Experiment.zip?dl=0"
    download_folder <- "~/Downloads/UgandaAnalysis.zip"

    # download into new directory
    download.file( uganda_data_url,  destfile = download_folder)

    # unzip and list files
    unzip(download_folder, exdir = "~/Downloads/UgandaAnalysis")
  }

  # set new wd
  setwd(sprintf('%s/Public Replication Data, YOP Experiment/',
                gsub(download_folder,pattern="\\.zip",replace="")))

  # see directory contents
  list.files()

  # images saved here
  list.files(  "./Uganda2000_processed"  )

  # individual-level data
  UgandaDataProcessed <- read.csv(  "./UgandaDataProcessed.csv"  )

  # unit-level covariates (many covariates are subject to missingness!)
  dim( UgandaDataProcessed )
  table( UgandaDataProcessed$age )

  # approximate longitude + latitude for units
  head(  cbind(UgandaDataProcessed$geo_long, UgandaDataProcessed$geo_lat) )

  # image keys of units (use for referencing satellite images)
  UgandaDataProcessed$geo_long_lat_key

  # an experimental outcome
  UgandaDataProcessed$Yobs

  # treatment variable
  UgandaDataProcessed$Wobs

  # information on keys linking to satellite images for all of Uganda
  # (not just experimental context, use for constructing transportability maps)
  UgandaGeoKeyMat <- read.csv(  "./UgandaGeoKeyMat.csv"  )

  # set outcome to an income index
  UgandaDataProcessed$Yobs <- UgandaDataProcessed$income_index_e_RECREATED

  # drop observations with NAs in key variables
  # (you can also use a multiple imputation strategy)
  UgandaDataProcessed <- UgandaDataProcessed[!is.na(UgandaDataProcessed$Yobs) &
                                               !is.na(UgandaDataProcessed$Wobs) &
                                               !is.na(UgandaDataProcessed$geo_lat) , ]

  # sanity checks
  {
    # write a function that reads in images as saved and process them into an array
    NBANDS <- 3L
    imageHeight <- imageWidth <- 351L #  pixel height/width
    acquireImageRep <- function(keys){
      # initialize an array shell to hold image slices
      array_shell <- array(NA, dim = c(1L, imageHeight, imageWidth, NBANDS))

      # iterate over keys:
      # -- images are referenced to keys
      # -- keys are referenced to units (to allow for duplicate images uses)
      array_ <- sapply(keys, function(key_) {
        # iterate over all image bands (NBANDS = 3 for RBG images)
        for (band_ in 1:NBANDS) {
          # place the image in the correct place in the array
          array_shell[,,,band_] <-
            as.matrix(
              #data.table::fread(input = 
              read.csv(
                    sprintf("./Uganda2000_processed/GeoKey%s_BAND%s.csv", key_, band_), header = FALSE)[-1,])
        }
        return(array_shell)
      }, simplify = "array")

      # return the array in the format c(nBatch, imageWidth, imageHeight, nChannels)
      # ensure that the dimensions are correctly ordered for further processing
      if(length(keys) > 1){ array_ <- aperm(array_[1,,,,], c(4, 1, 2, 3) ) }
      if(length(keys) == 1){
        array_ <- aperm(array_, c(1,5, 2, 3, 4))
        array_ <- array(array_, dim(array_)[-1])
      }

      return(array_)
    }

    # try out the function
    # note: some units are co-located in same area (hence, multiple observations per image key)
    ImageBatch <- acquireImageRep( UgandaDataProcessed$geo_long_lat_key[ check_indices <- c(1, 20, 50, 101)  ])
    acquireImageRep( UgandaDataProcessed$geo_long_lat_key[ check_indices[1] ]   )

    # sanity checks in the analysis of earth observation data are essential
    # check that images are centered around correct location
    causalimages::image2(  as.array(ImageBatch)[1,,,1] )
    UgandaDataProcessed$geo_long[check_indices[1]]
    UgandaDataProcessed$geo_lat[check_indices[1]]
    # check against google maps to confirm correctness
    # https://www.google.com/maps/place/1%C2%B018'16.4%22N+34%C2%B005'15.1%22E/@1.3111951,34.0518834,10145m/data=!3m1!1e3!4m4!3m3!8m2!3d1.3045556!4d34.0875278?entry=ttu

    # scramble data (important for reading into causalimages::WriteTfRecord
    # to ensure no systematic biases in data sequence with model training
    set.seed(144L); UgandaDataProcessed <- UgandaDataProcessed[sample(1:nrow(UgandaDataProcessed)),]
  }

  # subset data
  # UgandaDataProcessed <- UgandaDataProcessed[sample(1:24),]

  # Image heterogeneity example with tfrecords
  # write a tf records repository
  # whenever changes are made to the input data to AnalyzeImageHeterogeneity, WriteTfRecord() should be re-run
  # to ensure correct ordering of data
  tfrecord_loc <- "~/Downloads/GeoRerandomizeTutorial.tfrecord"
  if( reSaveTfRecords <- F ){
      causalimages::WriteTfRecord(  file = tfrecord_loc,
                                    uniqueImageKeys = unique(UgandaDataProcessed$geo_long_lat_key),
                                    acquireImageFxn = acquireImageRep, 
                                    conda_env = "jax_cpu", conda_env_required = T )
  }

  # get representations
  MyImageEmbeddings <- causalimages::GetImageRepresentations(
    file  = tfrecord_loc,
    imageKeysOfUnits = UgandaDataProcessed$geo_long_lat_key,
    nDepth_ImageRep = 1L,
    pretrainedModel = "clip-rsicd", 
    nWidth_ImageRep = 512L,
    batchSize = 2L,
    conda_env = "jax_cpu", conda_env_required = T
  )
  MyImageEmbeddings <- MyImageEmbeddings$ImageRepresentations

  UgandaDataProcessed$geo_lat_center
  # restart analysis here 
  # write.csv(data.frame("treated"=UgandaDataProcessed$treated, 
                        #"obsY"=UgandaDataProcessed$human_capital_index_e,
                        #"lat"=UgandaDataProcessed$geo_lat_center,"long"=UgandaDataProcessed$geo_long_center), 
                        #file = "~/Downloads/RCTData.csv")
  # write.csv(MyImageEmbeddings, file = "~/Downloads/MyImageEmbeddings.csv")
  # MyImageEmbeddings <- read.csv("~/Downloads/MyImageEmbeddings.csv")[,-1]
  # MyRCTData <- read.csv("~/Downloads/RCTData.csv")[,-1]
  
  # perform rerandomization
  candidate_randomizations_array <- fastrerandomize::generate_randomizations(
    n_units = nrow(MyImageEmbeddings),
    n_treated = nrow(MyImageEmbeddings) / 2,
    X = MyImageEmbeddings,
    randomization_accept_prob = 0.0001, 
    conda_env_required = F)
  dim( np$array( candidate_randomizations_array ) )

  fastrerandomize::print2("Done with Geo-rerandomization tutorial!")
}

