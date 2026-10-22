#' Draws a random sample of acceptable randomizations from all possible complete randomizations using Monte Carlo sampling
#' 
#' This function performs sampling with replacement to generate randomizations in a memory-efficient way.
#' It processes randomizations in batches to avoid memory issues and filters them based on covariate balance.
#' The function uses JAX for fast computation and memory management.
#'
#' @param n_units An integer specifying the total number of experimental units. 
#' @param n_treated An integer specifying the number of units to be assigned to treatment. 
#' @param X A numeric matrix of covariates used for balance checking. Cannot be NULL.
#' @param randomization_accept_prob A numeric value between 0 and 1 specifying the probability threshold for accepting randomizations based on balance. Default is 1
#' @param threshold_func A traceable JAX function that computes a balance measure for each randomization in a batch. It is called as \code{threshold_func(X, S_inv, S_inv_approx, W, n0, n1, approximate_inv)}, where \code{X} is the covariate matrix, \code{S_inv} the inverse sample covariance, \code{S_inv_approx} the reciprocal covariate variances, \code{W} a batch of 0/1 assignments (one per row), \code{n0}/\code{n1} the group sizes, and \code{approximate_inv} the logical argument below; it must return one value per row of \code{W}. Default (\code{NULL}) uses Hotelling's T-squared statistic.
#' @param max_draws An integer specifying the maximum number of randomizations to draw.
#' @param batch_size An integer specifying how many randomizations to process at once, or \code{NULL} (default) to size batches automatically from \code{n_units}. Lower values use less memory but may be slower. Results do not depend on the batch size.
#' @param approximate_inv A logical value indicating whether to use an approximate inverse 
#'   (diagonal of the covariance matrix) instead of the full matrix inverse when computing 
#'   balance metrics. This can speed up computations for high-dimensional covariates.
#'   Default is \code{TRUE}.
#' @param verbose A logical value indicating whether to print detailed information about batch processing progress, and GPU memory usage. Default is \code{FALSE}. 
#' @param conda_env A character string specifying the name of the conda environment to use 
#'   via \code{reticulate}. Default is \code{"fastrerandomize_env"}.
#' @param conda_env_required A logical indicating whether the specified conda environment 
#'   must be strictly used. If \code{TRUE}, an error is thrown if the environment is not found. 
#'   Default is \code{TRUE}.
#' @param seed Optional whole-number seed for the JAX random stream. If \code{NULL},
#'   a seed is drawn from R's full positive integer range, so \code{set.seed()} can
#'   still be used to reproduce the draw. Default is \code{NULL}.
#' @details
#' The function works by:
#' 1. Deriving the key of draw \code{i} as \code{jax.random.split(key, max_draws)[i]}
#'    inside each batch, without materializing all keys.
#' 2. Permuting the base assignment for each key and computing its balance measure.
#' 3. Keeping the \code{ceiling(max_draws * randomization_accept_prob)} most balanced draws
#'    and regenerating their assignments from their keys.
#'
#' All batches run inside one compiled JAX program (on Metal, or when
#' \code{options(fastrerandomize.mc_fused = FALSE)} is set, batches are dispatched from R).
#' Covariates are centered and scaled or whitened in double precision before
#' they are sent to the device; constant covariates are dropped with a warning.
#' Every generated assignment is checked to contain exactly \code{n_treated} treated units.
#'
#' @return The function returns a \emph{list} with three elements:
#' \code{candidate_randomizations}: an int8 array of randomization vectors;
#' \code{keys_candidate_randomizations}: their JAX random keys;
#' \code{M_candidate_randomizations}: an array of their balance measures. 
#' @examples
#' 
#' \dontrun{
#' # Generate synthetic data 
#' X <- matrix(rnorm(100*5), 100, 5) # 5 covariates
#' 
#' # Generate 1000 randomizations for 100 units with 50 treated
#' rand_less_strict <- generate_randomizations_mc(
#'                n_units = 100, 
#'                n_treated = 50, 
#'                X = X, 
#'                randomization_accept_prob=0.01, 
#'                max_draws = 100000,
#'                batch_size = 1000)
#'
#' # Use a stricter balance criterion
#' rand_more_strict <- generate_randomizations_mc(
#'                n_units = 100, 
#'                n_treated = 50, 
#'                X = X, 
#'                randomization_accept_prob=0.001, 
#'                max_draws = 1000000,
#'                batch_size = 1000)
#' }
#'
#' @seealso
#' \code{\link{generate_randomizations}} for full randomization generation function. 
#' \code{\link{generate_randomizations_exact}} for the exact version. 
#' 
#' @import reticulate
#' @export
#' @md
generate_randomizations_mc <- function(n_units, 
                                       n_treated,
                                       X,
                                       randomization_accept_prob = 1,
                                       threshold_func = NULL, 
                                       max_draws = 100000, 
                                       batch_size = NULL, 
                                       approximate_inv = TRUE,
                                       verbose = TRUE,
                                       conda_env = "fastrerandomize_env", 
                                       conda_env_required = TRUE,
                                       seed = NULL
                                      ){
  if (length(n_units) != 1L || !is.numeric(n_units) || !is.finite(n_units) ||
      n_units < 2 || n_units != floor(n_units)) {
    stop("'n_units' must be a whole number of at least 2.", call. = FALSE)
  }
  if (length(n_treated) != 1L || !is.numeric(n_treated) || !is.finite(n_treated) ||
      n_treated < 1 || n_treated >= n_units || n_treated != floor(n_treated)) {
    stop("'n_treated' must be a whole number between 1 and n_units - 1.", call. = FALSE)
  }
  if (length(randomization_accept_prob) != 1L ||
      !is.numeric(randomization_accept_prob) ||
      !is.finite(randomization_accept_prob) ||
      randomization_accept_prob < 0 || randomization_accept_prob > 1) {
    stop("'randomization_accept_prob' must be between 0 and 1.", call. = FALSE)
  }
  if (length(max_draws) != 1L || !is.numeric(max_draws) || !is.finite(max_draws) ||
      max_draws < 1 || max_draws > .Machine$integer.max || max_draws != floor(max_draws)) {
    stop("'max_draws' must be a positive whole number in R's integer range.", call. = FALSE)
  }
  if (!is.null(batch_size) &&
      (length(batch_size) != 1L || !is.numeric(batch_size) || !is.finite(batch_size) ||
       batch_size < 1 || batch_size > .Machine$integer.max || batch_size != floor(batch_size))) {
    stop("'batch_size' must be a positive whole number in R's integer range.", call. = FALSE)
  }
  n_units <- as.integer(n_units)
  n_treated <- as.integer(n_treated)
  max_draws <- as.integer(max_draws)
  X <- .prepare_X(X, n_units)
  seed <- .resolve_jax_seed(seed)

  if (is.null(.ensure_backend(conda_env, conda_env_required))) { return(NULL) }
  jnp <- fastrr_env$jnp

  batch_size <- .resolve_batch_size(batch_size, n_units, max_draws)
  num_batches <- as.integer(ceiling(max_draws / batch_size))

  # Determine the number of permutations to accept based on the acceptance probability
  float_num_to_accept <- max_draws * randomization_accept_prob
  if (float_num_to_accept < 1){
    warning(
      "max_draws * randomization_accept_prob is below 1; returning the single best sampled randomization.",
      call. = FALSE
    )
  }
  num_to_accept <- as.integer(max(ceiling(float_num_to_accept), 1)) # Ensure at least one

  # Balance inputs are built in double precision on the host
  default_threshold <- .is_default_threshold(threshold_func)
  design <- .balance_design(X, approximate_inv, need_full_inverse = !default_threshold)
  if (default_threshold) {
    scorers <- fastrr_env$MCScorers
    balance_jit <- fastrr_env$WhitenedT2Jit
    design_jax <- list(jnp$array(design$Xw, dtype = jnp$float32))
  } else {
    balance_fn <- function(W_, design_, n0, n1) {
      jnp$reshape(threshold_func(
        design_[[1]], design_[[2]], design_[[3]], W_$astype(jnp$int8),
        n0$astype(jnp$int32), n1$astype(jnp$int32), approximate_inv
      ), -1L)$astype(jnp$float32)
    }
    scorers <- .mc_scorers(balance_fn)
    balance_jit <- fastrr_env$jax$jit(function(design_, W_, n0, n1) balance_fn(W_, design_, n0, n1))
    design_jax <- list(jnp$array(X, dtype = jnp$float32),
                       jnp$array(design$S_inv, dtype = jnp$float32),
                       jnp$array(design$S_inv_approx, dtype = jnp$float32))
  }

  # Base vector: 1s for treated, 0s for control (int32 sort payload)
  base_vector_jax <- jnp$array(c(rep(1L, n_treated), rep(0L, n_units - n_treated)),
                               dtype = jnp$int32)
  key <- fastrr_env$jax$random$PRNGKey(seed)
  n0_jax <- jnp$float32(n_units - n_treated)
  n1_jax <- jnp$float32(n_treated)
  n1_int <- jnp$int32(n_treated)
  lane <- jnp$arange(batch_size, dtype = jnp$uint32)
  max_draws_jax <- jnp$uint32(max_draws)
  fused <- fastrr_env$counter_keys_ok &&
    isTRUE(getOption("fastrerandomize.mc_fused", !.is_metal_backend()))
  # Older PRNG schemes: materialize jax.random.split(key, max_draws) instead
  all_keys <- if (fastrr_env$counter_keys_ok) NULL else fastrr_env$jax$random$split(key, max_draws)

  if (verbose){
    message(sprintf("Scoring %d draws in %d batch(es) of %d (%s).", max_draws, num_batches,
                    batch_size, if (fused) "one compiled program" else "dispatched from R"))
  }
  t0 <- Sys.time()

  if (fused) {
    scored <- scorers$all(key, lane,
                          jnp$arange(0, as.numeric(num_batches) * batch_size, batch_size,
                                     dtype = jnp$uint32),
                          design_jax, base_vector_jax, n0_jax, n1_jax, n1_int, max_draws_jax)
    order_all <- scored[[1]]
    M_sorted <- scored[[2]]
    n_bad <- .to_r(scored[[3]])
  } else {
    batches <- lapply(seq_len(num_batches) - 1L, function(b_) {
      if (verbose) { message(paste0("On batch ", b_ + 1L, " of ", num_batches)) }
      start <- jnp$uint32(as.numeric(b_) * batch_size)
      if (is.null(all_keys)) {
        return(scorers$batch(key, lane, start, design_jax, base_vector_jax,
                             n0_jax, n1_jax, n1_int, max_draws_jax))
      }
      idx <- jnp$add(lane, start)
      scorers$batch_from_keys(jnp$take(all_keys, idx, axis = 0L, mode = "clip"), idx,
                              design_jax, base_vector_jax, n0_jax, n1_jax, n1_int,
                              max_draws_jax)
    })
    sorted <- fastrr_env$SortBalance(jnp$concatenate(lapply(batches, `[[`, 1L), 0L))
    order_all <- sorted[[1]]
    M_sorted <- sorted[[2]]
    n_bad <- sum(vapply(batches, function(b_) as.numeric(.to_r(b_[[2]])), numeric(1)))
  }
  .assert_valid_rows(n_bad, n_treated)

  # Keep the num_to_accept most balanced draws (ties broken by draw index)
  keep_range <- jnp$arange(num_to_accept)
  indices_to_keep <- jnp$take(order_all, keep_range)
  top_M_results <- jnp$take(M_sorted, keep_range)

  # (Re-)generate the accepted permutations from their keys, one batch at a
  # time, and check that they reproduce their balance scores
  balance_design_arg <- if (default_threshold) design_jax[[1]] else design_jax
  regenerated <- lapply(seq(0L, num_to_accept - 1L, by = batch_size), function(s_) {
    chunk <- jnp$arange(as.integer(s_), as.integer(min(s_ + batch_size, num_to_accept)))
    idx <- jnp$take(indices_to_keep, chunk)
    keys <- if (is.null(all_keys)) fastrr_env$CounterKeysJit(key, idx) else jnp$take(all_keys, idx, axis = 0L)
    regen <- fastrr_env$MCRegenerate(keys, base_vector_jax, n1_int)
    M_regen <- as.numeric(.to_r(balance_jit(balance_design_arg, regen[[1]], n0_jax, n1_jax)))
    M_chunk <- as.numeric(.to_r(jnp$take(top_M_results, chunk)))
    list(keys, regen[[1]], as.numeric(.to_r(regen[[2]])),
         sum(abs(M_regen - M_chunk) > 1e-4 * abs(M_chunk) + 1e-6))
  })
  .assert_valid_rows(sum(vapply(regenerated, `[[`, numeric(1), 3L)), n_treated)
  .assert_reproducible(sum(vapply(regenerated, `[[`, numeric(1), 4L)), num_to_accept)
  top_keys <- jnp$concatenate(lapply(regenerated, `[[`, 1L), 0L)
  top_perms <- jnp$concatenate(lapply(regenerated, `[[`, 2L), 0L)

  if(verbose){ message(sprintf("MC Loop Time (s): %.4f", as.numeric(difftime(Sys.time(), t0, units = "secs"))))}

  return(list(
              "candidate_randomizations" = top_perms,
              "keys_candidate_randomizations" = top_keys,
              "M_candidate_randomizations"= top_M_results
              ))
}
