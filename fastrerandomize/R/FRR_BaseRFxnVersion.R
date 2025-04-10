#' Compute Hotelling's T^2 in base R
#'
#' This function provides a base R implementation of Hotelling's T^2
#' balance measure, renamed with `_R` for clarity that it is the R-based
#' analog to the JAX version in fastrerandomize.
#'
#' @param X A numeric matrix of covariates (observations in rows).
#' @param W A 0/1 treatment assignment vector of the same length as nrow(X).
#'
#' @return A numeric scalar: the Hotelling's T^2 for that assignment. 
#'
#' @export
hotellingT2_R <- function(X, W) {
  # T^2 = (n0 * n1 / (n0 + n1)) * (xbar1 - xbar0)^T * S_inv * (xbar1 - xbar0)
  n <- length(W)
  n1 <- sum(W)
  n0 <- n - n1
  if (n1 == 0 || n0 == 0) return(NA_real_)  # invalid scenario
  
  xbar_treat <- colMeans(X[W == 1, , drop = FALSE])
  xbar_control <- colMeans(X[W == 0, , drop = FALSE])
  diff_vec <- (xbar_treat - xbar_control)
  
  # covariance (pooled) – we just use cov(X)
  S <- cov(X)
  Sinv <- tryCatch(solve(S), error = function(e) NULL)
  if (is.null(Sinv)) {
    # fallback: diagonal approximation if solve fails
    Sinv <- diag(1 / diag(S), ncol(S))
  }
  
  out <- (n0 * n1 / (n0 + n1)) * c(t(diff_vec) %*% Sinv %*% diff_vec)
  out
}


#' Generate randomizations in base R, filtering by Hotelling's T^2 acceptance
#'
#' Base R function to either do exact enumeration or Monte Carlo random permutations,
#' then keep the fraction whose T^2 is below the acceptance cutoff.
#'
#' @param n_units Integer, total number of units.
#' @param n_treated Integer, number of units to be assigned to treatment.
#' @param X Covariate matrix (n_units x p).
#' @param accept_prob Numeric in [0, 1]: keep the fraction of randomizations 
#'   that have the lowest T^2 up to this quantile.
#' @param random_type Either "exact" or "monte_carlo".
#' @param max_draws If `random_type="monte_carlo"`, how many permutations to sample.
#' @param batch_size If `random_type="monte_carlo"`, how many permutations to handle per chunk.
#'
#' @return A list with:
#'   \itemize{
#'     \item \code{randomizations}: a matrix (rows = accepted assignments).
#'     \item \code{balance}: numeric vector of T^2 values for each accepted assignment.
#'   }
#'
#' @export
generate_randomizations_R <- function(n_units, n_treated, X, accept_prob, random_type,
                                            max_draws, batch_size) {
  
  # Safety checks for exact enumeration size
  if (random_type == "exact") {
    n_comb_total <- choose(n_units, n_treated)
    if (n_comb_total > 1e6) {
      warning(
        sprintf("Exact randomization is requested, but that is %s combinations. 
                 This may be infeasible in terms of memory/time. 
                 Consider Monte Carlo instead.", 
                format(n_comb_total, big.mark=",")), 
        immediate. = TRUE
      )
    }
  }
  
  if (random_type == "exact") {
    # ---------- EXACT RANDOMIZATIONS ----------
    cidx <- combn(n_units, n_treated)
    n_comb <- ncol(cidx)
    
    assignment_mat <- matrix(0, nrow = n_comb, ncol = n_units)
    for (i in seq_len(n_comb)) {
      assignment_mat[i, cidx[, i]] <- 1
    }
    # Compute T^2 for each row
    T2vals <- apply(assignment_mat, 1, function(w) hotellingT2_R(X, w))
    # Drop NA
    keep_idx <- which(!is.na(T2vals))
    assignment_mat <- assignment_mat[keep_idx, , drop = FALSE]
    T2vals <- T2vals[keep_idx]
    
    # acceptance threshold
    cutoff <- quantile(T2vals, probs = accept_prob)
    keep_final <- (T2vals < cutoff)
    assignment_mat_accepted <- assignment_mat[keep_final, , drop = FALSE]
    T2vals_accepted <- T2vals[keep_final]
    
  } else {
    # ---------- MONTE CARLO RANDOMIZATIONS ----------
    base_assign <- c(rep(1, n_treated), rep(0, n_units - n_treated))
    
    batch_count <- ceiling(max_draws / batch_size)
    all_assign <- list()
    all_T2 <- numeric(0)
    
    cur_draw <- 0
    for (b in seq_len(batch_count)) {
      ndraws_here <- min(batch_size, max_draws - cur_draw)
      cur_draw <- cur_draw + ndraws_here
      
      perms <- matrix(nrow = ndraws_here, ncol = n_units)
      for (j in seq_len(ndraws_here)) {
        perms[j, ] <- sample(base_assign)
      }
      T2vals_batch <- apply(perms, 1, function(w) hotellingT2_R(X, w))
      
      all_assign[[b]] <- perms
      all_T2 <- c(all_T2, T2vals_batch)
    }
    assignment_mat <- do.call(rbind, all_assign)
    
    keep_idx <- which(!is.na(all_T2))
    assignment_mat <- assignment_mat[keep_idx, , drop = FALSE]
    all_T2 <- all_T2[keep_idx]
    
    cutoff <- quantile(all_T2, probs = accept_prob)
    keep_final <- (all_T2 < cutoff)
    assignment_mat_accepted <- assignment_mat[keep_final, , drop = FALSE]
    T2vals_accepted <- all_T2[keep_final]
  }
  
  list(randomizations = assignment_mat_accepted, balance = T2vals_accepted)
}


#' Simple difference in means in base R
#'
#' @param Y Numeric outcome vector.
#' @param W 0/1 treatment assignment vector.
#'
#' @return Scalar difference in means: mean(Y|W=1) - mean(Y|W=0).
#'
#' @export
diff_in_means_R <- function(Y, W) {
  mean(Y[W == 1]) - mean(Y[W == 0])
}


#' Compute potential outcome difference in means for a single assignment
#'   under a hypothesized tau in base R
#'
#' @param Wprime A 0/1 assignment vector for which to compute the diff in means.
#' @param obsY Observed outcome vector.
#' @param obsW Observed assignment vector.
#' @param tau The hypothesized true effect for the shift in outcomes under treatment.
#'
#' @return Scalar difference in means for the assignment Wprime.
#'
#' @export
compute_diff_at_tau_for_oneW_R <- function(Wprime, obsY, obsW, tau) {
  # Y0_under_null = obsY - obsW*tau
  Y0 <- obsY - obsW * tau
  # Then Y' under Wprime means we add tau for those assigned in Wprime
  Yprime <- Y0
  Yprime[Wprime == 1] <- Y0[Wprime == 1] + tau
  diff_in_means_R(Yprime, Wprime)
}


#' Fiducial interval logic in base R, for randomization test
#'
#' @param obsW Observed assignment (0/1).
#' @param obsY Observed outcome.
#' @param allW Matrix of candidate random assignments (rows = assignments).
#' @param tau_obs Observed difference in means with obsW, obsY.
#' @param alpha Significance level (default 0.05).
#' @param c_initial A numeric step scale (default 2).
#' @param n_search_attempts Number of bracket search attempts (default 500).
#'
#' @return 2-element numeric vector [lower, upper] or [NA, NA] if none accepted.
#'
#' @export
find_fiducial_interval_R <- function(obsW, obsY, allW, tau_obs, alpha = 0.05, 
                                           c_initial = 2, n_search_attempts = 500) {
  
  # Attempt random bracket approach
  lowerBound_est <- tau_obs - 3 * tau_obs
  upperBound_est <- tau_obs + 3 * tau_obs
  
  z_alpha <- qnorm(1 - alpha)
  k <- 2 / (z_alpha * (2 * pi)^(-1/2) * exp(-z_alpha^2 / 2))
  
  n_allW <- nrow(allW)
  
  for (step_t in seq_len(n_search_attempts)) {
    idx <- sample.int(n_allW, 1)
    Wprime <- allW[idx, ]
    
    # ~~~~~ update lowerBound ~~~~~
    Y0_lower <- obsY - obsW * lowerBound_est
    Yprime_lower <- Y0_lower
    Yprime_lower[Wprime == 1] <- Y0_lower[Wprime == 1] + lowerBound_est
    
    tau_at_step_lower <- diff_in_means_R(Yprime_lower, Wprime)
    delta <- tau_obs - tau_at_step_lower
    
    if (tau_at_step_lower < tau_obs) {
      lowerBound_est <- lowerBound_est + k * delta * (alpha / 2) / step_t
    } else {
      lowerBound_est <- lowerBound_est - k * (-delta) * (1 - alpha / 2) / step_t
    }
    
    # ~~~~~ update upperBound ~~~~~
    Y0_upper <- obsY - obsW * upperBound_est
    Yprime_upper <- Y0_upper
    Yprime_upper[Wprime == 1] <- Y0_upper[Wprime == 1] + upperBound_est
    
    tau_at_step_upper <- diff_in_means_R(Yprime_upper, Wprime)
    delta2 <- tau_at_step_upper - tau_obs
    
    if (tau_at_step_upper > tau_obs) {
      upperBound_est <- upperBound_est - k * delta2 * (alpha / 2) / step_t
    } else {
      upperBound_est <- upperBound_est + k * (-delta2) * (1 - alpha / 2) / step_t
    }
  }
  
  # Now do a grid search from (lowerBound_est - 1) to (upperBound_est * 2)
  grid_lower <- lowerBound_est - 1
  grid_upper <- upperBound_est * 2
  tau_seq <- seq(grid_lower, grid_upper, length.out = 100)
  
  accepted <- logical(length(tau_seq))
  for (i in seq_along(tau_seq)) {
    tau_pseudo <- tau_seq[i]
    diffs_pseudo <- apply(allW, 1, function(wp) {
      compute_diff_at_tau_for_oneW_R(wp, obsY, obsW, tau_pseudo)
    })
    frac_ge <- mean(diffs_pseudo >= tau_obs)
    frac_le <- mean(diffs_pseudo <= tau_obs)
    
    accepted[i] <- (min(frac_ge, frac_le) > alpha / 2)
  }
  
  if (!any(accepted)) {
    return(c(NA, NA))
  }
  
  c(min(tau_seq[accepted]), max(tau_seq[accepted]))
}


#' Base R randomization test: difference in means + optional fiducial interval
#'
#' @param obsW Observed assignment (0/1).
#' @param obsY Observed outcome vector.
#' @param allW Matrix of candidate random assignments (rows = assignments).
#' @param findFI Logical, whether to compute fiducial interval as well.
#' @param alpha Significance level (default 0.05).
#'
#' @return A list with p_value, tau_obs, and (optionally) FI if `findFI=TRUE`.
#'
#' @export
randomization_test_R <- function(obsW, obsY, allW, findFI = FALSE, alpha = 0.05) {
  tau_obs <- diff_in_means_R(obsY, obsW)
  
  diffs <- apply(allW, 1, function(w) diff_in_means_R(obsY, w))
  pval <- mean(abs(diffs) >= abs(tau_obs))
  
  FI <- NULL
  if (findFI) {
    FI <- find_fiducial_interval_R(obsW, obsY, allW, tau_obs, alpha = alpha)
  }
  
  list(p_value = pval, tau_obs = tau_obs, FI = FI)
}
