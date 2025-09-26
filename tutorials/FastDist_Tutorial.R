{
##############################################################################
# Fast distance calcs, tutorial (advanced)
##############################################################################

options(error = NULL)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Install and Set Up
#    (You only need to do this once)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# devtools::install_github(repo = "cjerzak/fastrerandomize-software/fastrerandomize")

# Build backend(s) if needed
# causalimages::BuildBackend()
# fastrerandomize::build_backend()  

# Simple Euclidean within-matrix distances (returns an n x n matrix)
X <- matrix(rnorm(50 * 8), 50, 8)
D <- fast_distance(X, metric = "euclidean")

# Cosine distance between two sets
A <- matrix(rnorm(100 * 16), 100, 16)
B <- matrix(rnorm(120 * 16), 120, 16)
Dcos <- fast_distance(A, B, metric = "cosine")

# Minkowski with p = 3 and feature weights
w <- runif(ncol(A))
Dm3 <- fast_distance(A, B, metric = "minkowski", p = 3, weights = w)

# Mahalanobis (diagonal approx, fast & robust)
Dmah_diag <- fast_distance(X, metric = "mahalanobis", approximate_inv = TRUE)

# Mahalanobis with full inverse (computed internally)
Dmah_full <- fast_distance(X, metric = "mahalanobis", approximate_inv = FALSE)

# Return a base R 'dist' object
D_dist <- fast_distance(X, metric = "euclidean", as_dist = TRUE)
  
fastrerandomize::print2("Done with Geo-rerandomization tutorial!")
}

