initialize_jax <- function(conda_env = "fastrerandomize_env", 
                           conda_env_required = TRUE) {
  # Load reticulate (Declared in Imports: in DESCRIPTION)
  reticulate::use_condaenv(condaenv = conda_env, required = conda_env_required)
  
  # Import Python packages once, storing them in fastrr_env
  if (!exists("jax", envir = fastrr_env, inherits = FALSE)) {
    fastrr_env$jax <- reticulate::import("jax")
    fastrr_env$jnp <- reticulate::import("jax.numpy")
    fastrr_env$np  <- reticulate::import("numpy")
  }
  
  # Disable 64-bit computations
  fastrr_env$jax$config$update("jax_enable_x64", FALSE)
  fastrr_env$jaxFloatType <- fastrr_env$jnp$float32
  
  # Setup core JAX functions once and store them in fastrr_env.
  if (!exists("VectorizedFastHotel2T2", envir = fastrr_env, inherits = FALSE)) {
    fastrr_env$InsertOnes <- fastrr_env$jax$jit(function(treat_indices_, zeros_) {
      zeros_$at[treat_indices_]$add(1L)
    })
    fastrr_env$InsertOnesVectorized <- fastrr_env$jax$jit(
      fastrr_env$jax$vmap(function(treat_indices_, zeros_) {
        fastrr_env$InsertOnes(treat_indices_, zeros_)
      }, list(1L, NULL))
    )

    fast_diff_in_means_jax_impl <- function(y_, w_, n0, n1) {
      my1 <- fastrr_env$jnp$divide(
        fastrr_env$jnp$sum(fastrr_env$jnp$multiply(y_, w_)), n1
      )
      my0 <- fastrr_env$jnp$divide(
        fastrr_env$jnp$sum(
          fastrr_env$jnp$multiply(y_, fastrr_env$jnp$subtract(1., w_))
        ),
        n0
      )
      fastrr_env$jnp$subtract(my1, my0)
    }
    fastrr_env$FastDiffInMeans <- fastrr_env$jax$jit(fast_diff_in_means_jax_impl)
    fastrr_env$W_VectorizedFastDiffInMeans <- fastrr_env$jax$jit(
      fastrr_env$jax$vmap(
        function(y_, w_, n0, n1) {
          fast_diff_in_means_jax_impl(y_, w_, n0, n1)
        },
        in_axes = list(NULL, 0L, NULL, NULL)
      )
    )

    fastrr_env$RowBroadcast <- fastrr_env$jax$vmap(function(mat, vec) {
      fastrr_env$jnp$multiply(mat, vec)
    }, in_axes = list(1L, NULL))

    fastrr_env$FastHotel2T2 <- function(
        samp_, samp_cov_inv, samp_cov_inv_approx, w_, n0, n1,
        approximate_inv = FALSE) {
      xbar1 <- fastrr_env$jnp$divide(
        fastrr_env$jnp$sum(
          fastrr_env$RowBroadcast(samp_, w_), 1L, keepdims = TRUE
        ),
        n1
      )
      xbar2 <- fastrr_env$jnp$divide(
        fastrr_env$jnp$sum(
          fastrr_env$RowBroadcast(
            samp_, fastrr_env$jnp$subtract(1., w_)
          ),
          1L,
          keepdims = TRUE
        ),
        n0
      )
      xbar_diff <- fastrr_env$jnp$subtract(xbar1, xbar2)
      cov_inv_times_xbar_diff <- fastrr_env$jax$lax$cond(
        pred = approximate_inv,
        true_fun = function() {
          fastrr_env$jnp$multiply(
            fastrr_env$jnp$expand_dims(samp_cov_inv_approx, 1L),
            xbar_diff
          )
        },
        false_fun = function() {
          fastrr_env$jnp$matmul(samp_cov_inv, xbar_diff)
        }
      )
      fastrr_env$jnp$multiply(
        (n0 * n1) / (n0 + n1),
        fastrr_env$jnp$matmul(
          fastrr_env$jnp$transpose(xbar_diff),
          cov_inv_times_xbar_diff
        )
      )
    }

    fastrr_env$VectorizedFastHotel2T2 <- fastrr_env$jax$jit(
      fastrr_env$jax$vmap(function(
          samp_, samp_cov_inv_, samp_cov_inv_approx_, w_, n0, n1,
          approximate_inv = FALSE) {
        fastrr_env$FastHotel2T2(
          samp_, samp_cov_inv_, samp_cov_inv_approx_, w_,
          n0, n1, approximate_inv
        )
      }, in_axes = list(NULL, NULL, NULL, 0L, NULL, NULL, NULL))
    )
  }

  # ------------------------------------------------------------------
  # GEMM-form kernels. Balance is a squared norm of whitened mean
  # differences (one matmul per batch), Monte Carlo keys are derived per
  # draw index, and matmuls use HIGHEST precision so TF32 (NVIDIA Ampere+)
  # does not undo float32 accuracy.
  # ------------------------------------------------------------------
  if (!exists("WhitenedT2", envir = fastrr_env, inherits = FALSE)) {
    jnp <- fastrr_env$jnp
    HIGHEST <- fastrr_env$jax$lax$Precision$HIGHEST
    fastrr_env$jax_extend_random <- tryCatch(
      reticulate::import("jax.extend.random"), error = function(e) NULL
    )

    # Rows idx of jax.random.split(key, N) under the partitionable threefry
    # scheme (default since JAX 0.5.0): key_i = threefry2x32(key, (0, i)).
    fastrr_env$CounterKeys <- function(key_, idx_) {
      lo_ <- idx_$astype(jnp$uint32)
      bits_ <- fastrr_env$jax_extend_random$threefry2x32_p$bind(
        jnp$broadcast_to(jnp$take(key_, 0L), lo_$shape),
        jnp$broadcast_to(jnp$take(key_, 1L), lo_$shape),
        jnp$zeros_like(lo_), lo_
      )
      jnp$stack(list(bits_[[1]], bits_[[2]]), axis = 1L)
    }
    fastrr_env$CounterKeysJit <- fastrr_env$jax$jit(fastrr_env$CounterKeys)
    fastrr_env$counter_keys_ok <- isTRUE(tryCatch({
      key0_ <- fastrr_env$jax$random$PRNGKey(0L)
      .to_r(jnp$array_equal(
        fastrr_env$CounterKeysJit(key0_, jnp$arange(16L, dtype = jnp$uint32)),
        fastrr_env$jax$random$split(key0_, 16L)
      ))
    }, error = function(e) FALSE))

    # Permutations use an int32 payload: 1-byte sort payloads are miscomputed
    # on some GPU backends (observed on ROCm for n >= 256).
    fastrr_env$BatchPermutation <- fastrr_env$jax$vmap(function(key_, base_) {
      fastrr_env$jax$random$permutation(key = key_, x = base_, axis = 0L,
                                        independent = TRUE)
    }, in_axes = list(0L, NULL))

    fastrr_env$WhitenedT2 <- function(Xw_, W_, n0, n1) {
      s_ <- jnp$matmul(W_$astype(jnp$float32), Xw_, precision = HIGHEST)
      diff_ <- jnp$subtract(jnp$multiply(s_, 1 / n1 + 1 / n0),
                            jnp$divide(jnp$sum(Xw_, 0L), n0))
      jnp$multiply(n0 * n1 / (n0 + n1), jnp$sum(jnp$square(diff_), 1L))
    }
    fastrr_env$WhitenedT2Jit <- fastrr_env$jax$jit(fastrr_env$WhitenedT2)

    fastrr_env$MCScorers <- .mc_scorers(function(W_, design_, n0, n1) {
      fastrr_env$WhitenedT2(design_[[1]], W_, n0, n1)
    })

    fastrr_env$MCRegenerate <- fastrr_env$jax$jit(function(keys_, base_, n1_int) {
      W_ <- fastrr_env$BatchPermutation(keys_, base_)
      list(W_$astype(jnp$int8),
           jnp$sum(jnp$not_equal(jnp$sum(W_, 1L), n1_int)))
    })

    fastrr_env$SortBalance <- fastrr_env$jax$jit(function(M_) {
      order_ <- jnp$argsort(M_)
      list(order_, jnp$take(M_, order_))
    })

    # Per-candidate sums w . (Y - mean(Y)), w . obsW and sum(w) for the
    # closed-form fiducial search.
    fastrr_env$CandidateMoments <- fastrr_env$jax$jit(function(cand_, Yc_, obsW_) {
      rhs_ <- jnp$stack(list(Yc_, obsW_, jnp$ones_like(Yc_)), axis = 1L)
      jnp$matmul(cand_$astype(jnp$float32), rhs_, precision = HIGHEST)
    })
    fastrr_env$IsBinary <- fastrr_env$jax$jit(function(cand_) {
      jnp$all(jnp$logical_or(jnp$equal(cand_, 0L), jnp$equal(cand_, 1L)))
    })

    # ||a||^2 + ||b||^2 - 2 a.b on pre-centered inputs; self-distances
    # (row_ids_ == column index) are zeroed when zero_self_ is TRUE.
    fastrr_env$PairwiseSqEuclidGemm <- fastrr_env$jax$jit(function(A_, B_, row_ids_, zero_self_) {
      d2_ <- jnp$subtract(
        jnp$add(jnp$sum(jnp$square(A_), 1L, keepdims = TRUE), jnp$sum(jnp$square(B_), 1L)),
        jnp$multiply(2, jnp$matmul(A_, jnp$transpose(B_), precision = HIGHEST))
      )
      self_ <- jnp$logical_and(
        zero_self_,
        jnp$equal(jnp$expand_dims(row_ids_, 1L), jnp$arange(B_$shape[[1]]))
      )
      jnp$where(self_, 0, jnp$maximum(d2_, 0))
    })
  }

  # ------------------------------------------------------------------
  # Pairwise distance kernels (JAX; float32, broadcasted, jitted)
  # ------------------------------------------------------------------
  {
    if (!exists("PairwiseSqEuclidean", envir = fastrr_env, inherits = FALSE)) {
      fastrr_env$PairwiseSqEuclidean <- fastrr_env$jax$jit(function(A_, B_) {
        diff_ <- fastrr_env$jnp$subtract(
          fastrr_env$jnp$expand_dims(A_, 1L),
          fastrr_env$jnp$expand_dims(B_, 0L)
        )
        fastrr_env$jnp$sum(fastrr_env$jnp$square(diff_), 2L)
      })
    }
    if (!exists("PairwiseEuclidean", envir = fastrr_env, inherits = FALSE)) {
      fastrr_env$PairwiseEuclidean <- fastrr_env$jax$jit(function(A_, B_) {
        fastrr_env$jnp$sqrt(fastrr_env$PairwiseSqEuclidean(A_, B_) + 1e-12)
      })
    }
    if (!exists("PairwiseManhattan", envir = fastrr_env, inherits = FALSE)) {
      fastrr_env$PairwiseManhattan <- fastrr_env$jax$jit(function(A_, B_) {
        diff_ <- fastrr_env$jnp$subtract(
          fastrr_env$jnp$expand_dims(A_, 1L),
          fastrr_env$jnp$expand_dims(B_, 0L)
        )
        fastrr_env$jnp$sum(fastrr_env$jnp$abs(diff_), 2L)
      })
    }
    if (!exists("PairwiseChebyshev", envir = fastrr_env, inherits = FALSE)) {
      fastrr_env$PairwiseChebyshev <- fastrr_env$jax$jit(function(A_, B_) {
        diff_ <- fastrr_env$jnp$subtract(
          fastrr_env$jnp$expand_dims(A_, 1L),
          fastrr_env$jnp$expand_dims(B_, 0L)
        )
        fastrr_env$jnp$max(fastrr_env$jnp$abs(diff_), 2L)
      })
    }
    if (!exists("PairwiseCosine", envir = fastrr_env, inherits = FALSE)) {
      fastrr_env$PairwiseCosine <- fastrr_env$jax$jit(function(A_, B_) {
        eps <- fastrr_env$jnp$array(1e-12, dtype = fastrr_env$jnp$float32)
        # Normalize rows
        A_norms <- fastrr_env$jnp$clip(
          fastrr_env$jnp$linalg$norm(A_, ord = 2L, axis = 1L, keepdims = TRUE),
          eps, 1e30
        )
        B_norms <- fastrr_env$jnp$clip(
          fastrr_env$jnp$linalg$norm(B_, ord = 2L, axis = 1L, keepdims = TRUE),
          eps, 1e30
        )
        A_n <- fastrr_env$jnp$divide(A_, A_norms)
        B_n <- fastrr_env$jnp$divide(B_, B_norms)
        sims <- fastrr_env$jnp$matmul(A_n, fastrr_env$jnp$transpose(B_n))
        fastrr_env$jnp$subtract(1.0, fastrr_env$jnp$clip(sims, -1.0, 1.0))
      })
    }
    if (!exists("PairwiseMinkowski", envir = fastrr_env, inherits = FALSE)) {
      fastrr_env$PairwiseMinkowski <- fastrr_env$jax$jit(function(A_, B_, p_) {
        diff_ <- fastrr_env$jnp$subtract(
          fastrr_env$jnp$expand_dims(A_, 1L),
          fastrr_env$jnp$expand_dims(B_, 0L)
        )
        fastrr_env$jnp$power(
          fastrr_env$jnp$sum(
            fastrr_env$jnp$power(fastrr_env$jnp$abs(diff_), p_), 2L),
          fastrr_env$jnp$divide(1.0, p_)
        )
      })
    }
    if (!exists("PairwiseWeightedMinkowski", envir = fastrr_env, inherits = FALSE)) {
      fastrr_env$PairwiseWeightedMinkowski <- fastrr_env$jax$jit(function(A_, B_, w_, p_) {
        # w_ length p; broadcast to (1,1,p)
        diff_ <- fastrr_env$jnp$subtract(
          fastrr_env$jnp$expand_dims(A_, 1L),
          fastrr_env$jnp$expand_dims(B_, 0L)
        )
        powered <- fastrr_env$jnp$power(fastrr_env$jnp$abs(diff_), p_)
        w_b <- fastrr_env$jnp$expand_dims(fastrr_env$jnp$expand_dims(w_, 0L), 0L)
        fastrr_env$jnp$power(
          fastrr_env$jnp$sum(fastrr_env$jnp$multiply(powered, w_b), 2L),
          fastrr_env$jnp$divide(1.0, p_)
        )
      })
    }
    if (!exists("PairwiseMahalanobisFull", envir = fastrr_env, inherits = FALSE)) {
      fastrr_env$PairwiseMahalanobisFull <- fastrr_env$jax$jit(function(A_, B_, S_inv_) {
        diff_ <- fastrr_env$jnp$subtract(
          fastrr_env$jnp$expand_dims(A_, 1L),
          fastrr_env$jnp$expand_dims(B_, 0L)
        )
        tmp <- fastrr_env$jnp$matmul(diff_, S_inv_)
        fastrr_env$jnp$sum(fastrr_env$jnp$multiply(tmp, diff_), 2L)  # squared Mahalanobis
      })
    }
    if (!exists("PairwiseMahalanobisDiag", envir = fastrr_env, inherits = FALSE)) {
      fastrr_env$PairwiseMahalanobisDiag <- fastrr_env$jax$jit(function(A_, B_, diag_inv_) {
        diff_ <- fastrr_env$jnp$subtract(
          fastrr_env$jnp$expand_dims(A_, 1L),
          fastrr_env$jnp$expand_dims(B_, 0L)
        )
        fastrr_env$jnp$sum(
          fastrr_env$jnp$multiply(fastrr_env$jnp$square(diff_), diag_inv_), 2L
        )
      })
    }
  }
}

# Jitted Monte Carlo scorers for a balance function balance_(W_, design_, n0, n1).
# Draw i uses key i of jax.random.split(key, max_draws), so results do not
# depend on the batch size; draws past max_draws (batch padding) score Inf.
#   batch: one batch of keys derived from start_ + lane_ (R-loop mode)
#   batch_from_keys: one batch of pre-split keys (fallback for older PRNGs)
#   all: every batch via lax.map, returning the stable order of the scores,
#        the sorted scores, and the count of rows without n1 treated units
.mc_scorers <- function(balance_) {
  jnp <- fastrr_env$jnp
  score_keys <- function(keys_, idx_, design_, base_, n0, n1, n1_int, max_draws_) {
    W_ <- fastrr_env$BatchPermutation(keys_, base_)
    valid_ <- jnp$less(idx_, max_draws_)
    n_bad <- jnp$sum(jnp$logical_and(valid_, jnp$not_equal(jnp$sum(W_, 1L), n1_int)))
    list(jnp$where(valid_, balance_(W_, design_, n0, n1), jnp$inf), n_bad)
  }
  score_start <- function(key_, lane_, start_, design_, base_, n0, n1, n1_int, max_draws_) {
    idx_ <- jnp$add(lane_, start_)
    score_keys(fastrr_env$CounterKeys(key_, idx_), idx_, design_, base_,
               n0, n1, n1_int, max_draws_)
  }
  list(
    batch = fastrr_env$jax$jit(score_start),
    batch_from_keys = fastrr_env$jax$jit(score_keys),
    all = fastrr_env$jax$jit(function(key_, lane_, starts_, design_, base_,
                                      n0, n1, n1_int, max_draws_) {
      out_ <- fastrr_env$jax$lax$map(function(start_) {
        score_start(key_, lane_, start_, design_, base_, n0, n1, n1_int, max_draws_)
      }, starts_)
      M_ <- jnp$reshape(out_[[1]], -1L)
      order_ <- jnp$argsort(M_)
      list(order_, jnp$take(M_, order_), jnp$sum(out_[[2]]))
    })
  )
}
