#!/usr/bin/env Rscript
# The production package is installed into a private library from the untouched
# source revision. All measured JIT functions come directly from its namespace.
args <- commandArgs(trailingOnly = TRUE)
mode <- if (length(args)) args[[1]] else "sweep"
root <- normalizePath(Sys.getenv("SOL_ROOT", "."))
out <- file.path(root, "results", Sys.getenv("SOL_OUTPUT_TAG", mode))
dir.create(out, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(file.path(root, "rlib"), .libPaths()))
library(fastrerandomize)
env_name <- Sys.getenv("SOL_CONDA_ENV", "preference_fm_cuda_20261009")
stopifnot(isTRUE(fastrerandomize:::.ensure_backend(env_name, TRUE)))
e <- fastrerandomize:::fastrr_env
stopifnot(e$counter_keys_ok, e$jax$devices()[[1]]$platform == "gpu")
h <- reticulate::import_from_path("timing", path = file.path(root, "benchmarks", "speed_of_light"), convert = FALSE)
h$save_metadata(out)
jnp <- e$jnp

# Match the default production caller on both the baseline and patched package.
production_design <- function(X) {
  fn <- fastrerandomize:::.balance_design
  if ("need_full_inverse" %in% names(formals(fn))) {
    fn(X, TRUE, need_full_inverse = FALSE)
  } else {
    fn(X, TRUE)
  }
}

inputs <- function(n, p, batch, draws = batch) {
  set.seed(20261022L)
  X <- matrix(rnorm(n * p), n, p)
  # Kernel-only timing excludes preparation and uses the package's actual inputs.
  Xw <- production_design(X)$Xw
  Xj <- jnp$array(Xw, dtype = jnp$float32)
  base <- jnp$array(c(rep(1L, n %/% 2L), rep(0L, n - n %/% 2L)), dtype = jnp$int32)
  key <- e$jax$random$PRNGKey(77L)
  lane <- jnp$arange(as.integer(batch), dtype = jnp$uint32)
  n0 <- jnp$float32(n - n %/% 2L)
  n1 <- jnp$float32(n %/% 2L)
  n1i <- jnp$int32(n %/% 2L)
  maxj <- jnp$uint32(draws)
  keys <- e$CounterKeysJit(key, lane)
  W <- e$jax$jit(e$BatchPermutation)(keys, base)
  h$sync(W)
  list(X = X, Xw = Xw, Xj = Xj, W = W, key = key, keys = keys,
       base = base, n0 = n0, n1 = n1, n1i = n1i, lane = lane,
       balance_args = list(Xj, W, n0, n1),
       batch_args = list(key, lane, jnp$uint32(0), list(Xj), base, n0, n1, n1i, maxj),
       all_args = list(key, lane, jnp$arange(0, as.numeric(draws), as.numeric(batch), dtype = jnp$uint32),
                       list(Xj), base, n0, n1, n1i, maxj))
}

verify <- function(z, scores, n, p, batch) {
  wr <- fastrerandomize:::.to_r(z$W)
  sr <- as.numeric(fastrerandomize:::.to_r(scores))
  stopifnot(all(wr %in% c(0L, 1L)), all(rowSums(wr) == n %/% 2L))
  ix <- unique(as.integer(seq(1, batch, length.out = min(64, batch))))
  s <- wr[ix, , drop = FALSE] %*% z$Xw
  d <- s * (1 / (n %/% 2L) + 1 / (n - n %/% 2L)) -
    matrix(colSums(z$Xw) / (n - n %/% 2L), nrow(s), p, byrow = TRUE)
  ref <- (n %/% 2L) * (n - n %/% 2L) / n * rowSums(d^2)
  rel <- max(abs(sr[ix] - ref) / pmax(abs(ref), 1e-12))
  stopifnot(rel < 1e-4)
  list(binary = TRUE, fixed_treatment_count = TRUE,
       checked_scores = length(ix), max_relative_error_float64 = rel)
}

if (mode == "sweep") {
  cases <- rbind(c(1024L, 16L, 4096L), c(1024L, 256L, 4096L),
                 c(1024L, 1024L, 4096L), c(1024L, 2048L, 4096L),
                 c(4096L, 1024L, 1024L), c(4096L, 1024L, 4096L),
                 c(4096L, 2048L, 4096L))
  for (i in seq_len(nrow(cases))) {
    n <- cases[i, 1L]; p <- cases[i, 2L]; batch <- cases[i, 3L]
    label <- sprintf("n%d_p%d_b%d", n, p, batch)
    cat("CASE", label, "\n")
    z <- inputs(n, p, batch)
    s <- h$measure(paste0(label, "_balance"), e$WhitenedT2Jit, z$balance_args, out, 30L)
    checked <- verify(z, s, n, p, batch)
    jsonlite::write_json(checked, file.path(out, paste0(label, "_validation.json")), auto_unbox = TRUE, pretty = TRUE)
    scored <- h$measure(paste0(label, "_batch"), e$MCScorers$batch, z$batch_args, out, 30L)
    # timing.py is imported with convert=FALSE; Python list indexing is zero-based.
    stopifnot(as.numeric(fastrerandomize:::.to_r(scored[[1L]])) == 0)
    rm(z, s, scored)
    gc()
  }
} else if (mode %in% c("profile", "pipeline", "api", "focus", "hostprep")) {
  n <- as.integer(Sys.getenv("SOL_N", "4096"))
  p <- as.integer(Sys.getenv("SOL_P", "1024"))
  batch <- as.integer(Sys.getenv("SOL_BATCH", "4096"))
  draws <- as.integer(Sys.getenv("SOL_DRAWS", "1048576"))
  label <- sprintf("n%d_p%d_b%d_d%d", n, p, batch, draws)
  z <- inputs(n, p, batch, draws)
  if (mode == "hostprep") {
    Rprof(file.path(out, "preparation.Rprof"), interval = 0.01)
    start <- proc.time()[["elapsed"]]
    original <- production_design(z$X)
    elapsed <- proc.time()[["elapsed"]] - start
    Rprof(NULL)
    diagonal_only <- function(X) {
      Xc <- sweep(X, 2L, colMeans(X))
      var_pop <- colMeans(Xc^2)
      inv <- ifelse(var_pop > 0, 1 / var_pop, 0)
      list(Xw = sweep(Xc, 2L, sqrt(inv), "*"), S_inv_approx = inv)
    }
    d <- diagonal_only(z$X)
    stopifnot(identical(original$Xw, d$Xw), identical(original$S_inv_approx, d$S_inv_approx))
    times <- replicate(5L, system.time(production_design(z$X))[["elapsed"]])
    prof <- summaryRprof(file.path(out, "preparation.Rprof"))
    result <- list(profiled_seconds = elapsed, samples_seconds = times,
                   median_seconds = median(times),
                   shortcut_available = "need_full_inverse" %in% names(formals(fastrerandomize:::.balance_design)),
                   identical_Xw = TRUE, identical_inverse_variance = TRUE,
                   profile_by_self = head(prof$by.self, 15L))
    jsonlite::write_json(result, file.path(out, paste0(label, ".json")), auto_unbox = TRUE, pretty = TRUE, digits = 16)
    print(result)
  } else if (mode == "focus") {
    s <- h$measure(paste0(label, "_balance"), e$WhitenedT2Jit, z$balance_args, out, 100L, 300L)
    checked <- verify(z, s, n, p, batch)
    jsonlite::write_json(checked, file.path(out, paste0(label, "_validation.json")), auto_unbox = TRUE, pretty = TRUE)
    scored <- h$measure(paste0(label, "_batch"), e$MCScorers$batch, z$batch_args, out, 100L, 150L)
    stopifnot(as.numeric(fastrerandomize:::.to_r(scored[[1L]])) == 0)
    stopifnot(max(abs(as.numeric(fastrerandomize:::.to_r(scored[[0L]])) -
                         as.numeric(fastrerandomize:::.to_r(s)))) < 1e-3)
  } else if (mode == "profile") {
    target <- Sys.getenv("SOL_TARGET", "batch")
    fn <- switch(target, batch = e$MCScorers$batch, balance = e$WhitenedT2Jit, all = e$MCScorers$all)
    a <- switch(target, batch = z$batch_args, balance = z$balance_args, all = z$all_args)
    h$profile(paste0(label, "_", target), fn, a, out, 3L)
  } else if (mode == "pipeline") {
    h$measure(paste0(label, "_all"), e$MCScorers$all, z$all_args, out, 15L, 3L)
  } else {
    call_args <- list(n_units = n, n_treated = n %/% 2L, X = z$X,
                      randomization_accept_prob = 0.001, max_draws = draws,
                      batch_size = batch, approximate_inv = TRUE, seed = 77L,
                      verbose = FALSE, conda_env = env_name, return_type = "R")
    run <- function() do.call(fastrerandomize::generate_randomizations, call_args)
    start <- proc.time()[["elapsed"]]
    first <- run()
    cold <- proc.time()[["elapsed"]] - start
    times <- replicate(7L, {
      start <- proc.time()[["elapsed"]]
      result <- run()
      elapsed <- proc.time()[["elapsed"]] - start
      stopifnot(nrow(result$randomizations) == ceiling(draws * 0.001),
                all(rowSums(result$randomizations) == n %/% 2L),
                identical(result$randomizations, first$randomizations),
                identical(result$balance, first$balance))
      elapsed
    })
    # Save values outside the timers for exact cross-revision comparison,
    # including keys (the high-level S3 wrapper does not expose them).
    mc_args <- call_args[names(call_args) != "return_type"]
    direct <- do.call(fastrerandomize::generate_randomizations_mc, mc_args)
    direct <- lapply(direct, fastrerandomize:::.to_r)
    stopifnot(identical(direct$candidate_randomizations, first$randomizations),
              identical(as.numeric(direct$M_candidate_randomizations), as.numeric(first$balance)))
    saveRDS(direct, file.path(out, paste0(label, "_outputs.rds")), compress = "xz")
    Xc <- sweep(z$X, 2L, colMeans(z$X))
    variances <- colMeans(Xc^2)
    check_ix <- seq_len(min(64L, nrow(first$randomizations)))
    reference <- vapply(check_ix, function(i) {
      w <- first$randomizations[i, ]
      d <- colMeans(z$X[w == 1L, , drop = FALSE]) - colMeans(z$X[w == 0L, , drop = FALSE])
      (n / 4) * sum(d^2 / variances)
    }, numeric(1))
    err <- max(abs(first$balance[check_ix] - reference) / abs(reference))
    stopifnot(err < 1e-4, !is.unsorted(first$balance))
    result <- list(n = n, p = p, batch = batch, draws = draws,
                   first_api_call_after_backend_setup_seconds = cold, samples_seconds = times,
                   median_seconds = median(times),
                   accepted = nrow(first$randomizations),
                   max_relative_error_float64 = err, repeat_identical = TRUE,
                   R = R.version.string, blas = extSoftVersion()[["BLAS"]])
    jsonlite::write_json(result, file.path(out, paste0(label, ".json")), pretty = TRUE, auto_unbox = TRUE, digits = 16)
    print(result)
  }
} else stop("Unknown mode: ", mode)
