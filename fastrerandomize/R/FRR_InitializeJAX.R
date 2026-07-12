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

    fastrr_env$get_stat_vec_at_tau_pseudo <- fastrr_env$jax$jit(function(
        treatment_pseudo, obsY_array, obsW_array, tau_pseudo, n0_array, n1_array) {
      Y0_under_null <- fastrr_env$jnp$subtract(
        obsY_array, fastrr_env$jnp$multiply(obsW_array, tau_pseudo)
      )
      Y1_under_null_pseudo <- fastrr_env$jnp$add(
        Y0_under_null,
        fastrr_env$jnp$multiply(treatment_pseudo, tau_pseudo)
      )
      Yobs_pseudo <- fastrr_env$jnp$add(
        fastrr_env$jnp$multiply(Y1_under_null_pseudo, treatment_pseudo),
        fastrr_env$jnp$multiply(
          Y0_under_null,
          fastrr_env$jnp$subtract(1., treatment_pseudo)
        )
      )
      fastrr_env$FastDiffInMeans(
        Yobs_pseudo, treatment_pseudo, n0_array, n1_array
      )
    })
    fastrr_env$vec1_get_stat_vec_at_tau_pseudo <- fastrr_env$jax$jit(
      fastrr_env$jax$vmap(function(
          treatment_pseudo, obsY_array, obsW_array, tau_pseudo, n0_array, n1_array) {
        fastrr_env$get_stat_vec_at_tau_pseudo(
          treatment_pseudo, obsY_array, obsW_array, tau_pseudo,
          n0_array, n1_array
        )
      }, in_axes = list(0L, NULL, NULL, NULL, NULL, NULL))
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
          a_min = eps, a_max = 1e30
        )
        B_norms <- fastrr_env$jnp$clip(
          fastrr_env$jnp$linalg$norm(B_, ord = 2L, axis = 1L, keepdims = TRUE),
          a_min = eps, a_max = 1e30
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
