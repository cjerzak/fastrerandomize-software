{
##############################################################################
# Geo-rerandomization tutorial (advanced)
##############################################################################

options(error = NULL)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1. Install and Set Up
#    (You only need to do this once)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# devtools::install_github(repo = "cjerzak/fastrerandomize-software/fastrerandomize")
# devtools::install_github(repo = "cjerzak/causalimages-software/causalimages")

# Build backend(s) if needed
# causalimages::BuildBackend()
# fastrerandomize::build_backend()  

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2. Download and Load the Uganda Data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
download_folder <- "~/Downloads/UgandaAnalysis.zip"
if (reDownloadRawData <- FALSE) {
  # specify Uganda data URL
  uganda_data_url <- "https://dl.dropboxusercontent.com/s/xy8xvva4i46di9d/Public%20Replication%20Data%2C%20YOP%20Experiment.zip?dl=0"
  
  # download into new directory
  download.file(uganda_data_url, destfile = download_folder)
  
  # unzip and list files
  unzip(download_folder, exdir = "~/Downloads/UgandaAnalysis")
}

# set new working directory
setwd(sprintf('%s/Public Replication Data, YOP Experiment/',
              gsub(download_folder, pattern = "\\.zip", replace = "")))

# see directory contents
list.files()

# images saved here
list.files("./Uganda2000_processed")

# individual-level data
UgandaDataProcessed <- read.csv("./UgandaDataProcessed.csv")

# check dimension and a variable or two
dim(UgandaDataProcessed)
table(UgandaDataProcessed$age)

# approximate longitude + latitude for units
head(cbind(UgandaDataProcessed$geo_long, UgandaDataProcessed$geo_lat))

# image keys of units (use for referencing satellite images)
UgandaDataProcessed$geo_long_lat_key

# an experimental outcome
UgandaDataProcessed$Yobs

# treatment variable
UgandaDataProcessed$Wobs

# information on keys linking to satellite images for all of Uganda
UgandaGeoKeyMat <- read.csv("./UgandaGeoKeyMat.csv")

# set outcome to an income index
UgandaDataProcessed$Yobs <- UgandaDataProcessed$income_index_e_RECREATED

# drop observations with NAs in key variables
UgandaDataProcessed <- UgandaDataProcessed[
  !is.na(UgandaDataProcessed$Yobs) &
    !is.na(UgandaDataProcessed$Wobs) &
    !is.na(UgandaDataProcessed$geo_lat),
]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 3. Acquire Satellite Images (Example)
#    Function reads in images and converts them to arrays for analysis
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
NBANDS <- 3L
imageHeight <- imageWidth <- 351L

acquireImageRep <- function(keys){
  # initialize an array shell to hold image slices
  array_shell <- array(NA, dim = c(1L, imageHeight, imageWidth, NBANDS))
  
  # iterate over keys:
  array_ <- sapply(keys, function(key_) {
    # for each band
    for (band_ in seq_len(NBANDS)) {
      # place the image in the correct place
      array_shell[,,,band_] <-
        as.matrix(
          read.csv(
            sprintf("./Uganda2000_processed/GeoKey%s_BAND%s.csv", key_, band_),
            header = FALSE
          )[-1, ]
        )
    }
    array_shell
  }, simplify = "array")
  
  # reorder dims for a batch
  if (length(keys) > 1) {
    array_ <- aperm(array_[1,,,,], c(4, 1, 2, 3))
  } else {
    array_ <- aperm(array_, c(1, 5, 2, 3, 4))
    array_ <- array(array_, dim(array_)[-1])
  }
  
  array_
}

# Test the function on a few entries
check_indices <- c(1, 20, 50, 101)
ImageBatch <- acquireImageRep(UgandaDataProcessed$geo_long_lat_key[check_indices])
acquireImageRep(UgandaDataProcessed$geo_long_lat_key[ check_indices[1] ])

# quick visual check
causalimages::image2(as.array(ImageBatch)[1,,,1])
UgandaDataProcessed$geo_long[check_indices[1]]
UgandaDataProcessed$geo_lat[check_indices[1]]

# scramble data (important for reading into WriteTfRecord)
set.seed(144L)
UgandaDataProcessed <- UgandaDataProcessed[sample(nrow(UgandaDataProcessed)), ]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 4. Create TFRecord of Images (Example)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if (resave_files <- FALSE) {
  tfrecord_loc <- "~/Downloads/GeoRerandomizeTutorial.tfrecord"
  causalimages::WriteTfRecord(
    file = tfrecord_loc,
    uniqueImageKeys   = unique(UgandaDataProcessed$geo_long_lat_key),
    acquireImageFxn   = acquireImageRep,
    conda_env         = "CausalImagesEnv",
    conda_env_required = TRUE
  )
  
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # 5. Extract Image Representations (Embeddings)
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  MyImageEmbeddings <- causalimages::GetImageRepresentations(
    file              = tfrecord_loc,
    imageKeysOfUnits  = UgandaDataProcessed$geo_long_lat_key,
    nDepth_ImageRep   = 1L,
    pretrainedModel   = "clip-rsicd",
    nWidth_ImageRep   = 512L,
    batchSize         = 2L,
    conda_env         = "CausalImagesEnv",
    conda_env_required = TRUE
  )
  MyImageEmbeddings <- MyImageEmbeddings$ImageRepresentations
  write.csv(MyImageEmbeddings, 
            file = "~/Downloads/GeoRerandomizeTutorial.csv")
}
MyImageEmbeddings <- read.csv(file = "~/Downloads/GeoRerandomizeTutorial.csv")[,-1]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 6. Perform Geo-rerandomization using fastrerandomize
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Here we generate a set of acceptable randomizations based on image embeddings
# randomization_accept_prob determines what fraction pass a balance test.
CandidateRandomizations <- fastrerandomize::generate_randomizations(
  n_units       = nrow(MyImageEmbeddings),
  n_treated     = round(nrow(MyImageEmbeddings) / 2),
  X             = MyImageEmbeddings,
  max_draws     = 10^5, 
  batch_size   = 10^3,
  randomization_accept_prob = 0.001,
  conda_env_required       = FALSE
)

# S3 usage: print, summary, and plot
cat("\n--- Using the S3 object returned by 'generate_randomizations' ---\n")
print(CandidateRandomizations)
summary(CandidateRandomizations)
plot(CandidateRandomizations)  

fastrerandomize::print2("Done with Geo-rerandomization tutorial!")
}
