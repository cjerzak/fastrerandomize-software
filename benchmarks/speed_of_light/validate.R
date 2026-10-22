#!/usr/bin/env Rscript
# Run against the private installation and require a functioning JAX backend,
# so missing CUDA cannot silently turn integration coverage into skipped tests.
.libPaths(c(file.path(getwd(), "rlib"), .libPaths()))
library(fastrerandomize)
stopifnot(isTRUE(check_jax_availability("fastrerandomize_env")))
fastrerandomize:::.ensure_backend("fastrerandomize_env", TRUE)
cat("Device:", fastrerandomize:::.backend_device(), "\n")
filter <- Sys.getenv("SOL_TEST_FILTER", "")
results <- testthat::test_local(
  "source/fastrerandomize", filter = if (nzchar(filter)) filter else NULL,
  reporter = "summary", stop_on_failure = TRUE
)
summary <- as.data.frame(results)
print(colSums(summary[, c("failed", "skipped", "error", "warning", "passed")]))
jsonlite::write_json(summary[, c("file", "test", "nb", "failed", "skipped", "error", "warning", "passed")],
                     file.path("results", paste0("tests_", Sys.getenv("JAX_PLATFORMS"), ".json")),
                     pretty = TRUE, auto_unbox = TRUE)
