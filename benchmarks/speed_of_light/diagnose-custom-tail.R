#!/usr/bin/env Rscript
# Separate diagnostic for a CUDA custom-threshold tail failure.
# Run from the study root with the same environment as validate.R.
.libPaths(c(file.path(getwd(), "rlib"), .libPaths()))
library(fastrerandomize)
fastrerandomize:::.ensure_backend("fastrerandomize_env", TRUE)
e <- fastrerandomize:::fastrr_env
set.seed(309)
X <- matrix(rnorm(30), 10)
custom <- function(X_, S_, D_, W_, n0, n1, approximate_inv) {
  e$jnp$add(e$VectorizedFastHotel2T2(X_, S_, D_, W_, n0, n1, FALSE),
            e$VectorizedFastHotel2T2(X_, S_, D_, W_, n0, n1, TRUE))
}
result <- generate_randomizations_mc(
  10, 4, X, 0.25, threshold_func = custom, max_draws = 257L,
  batch_size = 64L, seed = 19L, verbose = FALSE
)
print(dim(fastrerandomize:::.to_r(result$candidate_randomizations)))
