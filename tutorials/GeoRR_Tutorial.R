{
##############################################################################
# Geo-rerandomization tutorial (advanced)
##############################################################################

options(error = NULL)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Install and Set Up
#    (You only need to do this once)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# devtools::install_github(repo = "cjerzak/fastrerandomize-software/fastrerandomize")
# devtools::install_github(repo = "cjerzak/causalimages-software/causalimages")

# Build backend(s) if needed
# causalimages::BuildBackend()
# fastrerandomize::build_backend()  
  
data(YOPData, package = "fastrerandomize")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Perform Geo-rerandomization using fastrerandomize
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Here we generate a set of acceptable randomizations based on image embeddings
# randomization_accept_prob determines what fraction pass a balance test.
CandidateRandomizations <- fastrerandomize::generate_randomizations(
  n_units       = nrow(YOPData$ImageEmbeddings),
  n_treated     = round(nrow(YOPData$ImageEmbeddings) / 2),
  X             = YOPData$ImageEmbeddings,
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

