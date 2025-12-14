#' Diagnostic map from observed (or targeted) balance to precision and stringency
#'
#' Implements the calculations in Theorem 1 and Appendix D of the paper involving: 
#' (1) Realized RMSE from an observed Mahalanobis distance M (or SMDs);
#' (2) Ex-ante RMSE when accepting assignments with M < a (equivalently, with acceptance probability q under complete randomization);
#' (3) largest acceptance probability q that attains a user-specified precision goal,provided via an RMSE target or via a power target (alpha, 1-beta, |tau|).
#'
#' @param smd Optional numeric vector of standardized mean differences; if supplied,
#'   M is computed as sum(smd^2), and d = length(smd).
#' @param M Optional scalar Mahalanobis distance M; if provided without `smd`,
#'   you must also supply `d` (the number of covariates used in M).
#' @param d Optional integer number of covariates (needed if supplying only `M`).
#' @param n_T Integer, number of treated units.
#' @param n_C Integer, number of control units.
#' @param sigma Optional outcome noise SD (sigma). If `NULL`, absolute RMSEs cannot be
#'   formed; dimensionless "per-sigma" factors are still returned.
#' @param R2 Optional model R^2 for Y ~ X under the linear potential-outcomes model.
#'   Must lie in [0,1). If `NULL`, RMSEs that require R^2 are returned as NA, but the
#'   "per-sigma" formulas that do not need R^2 are still shown when possible.
#' @param rmse_goal Optional numeric target for RMSE (same units as Y). If supplied
#'   (with sigma and R2), the largest q achieving this ex-ante goal is returned.
#' @param tau Optional effect size |tau| (same units as Y) to back out an RMSE goal
#'   via a normal approximation to power.
#' @param alpha Size of a two-sided test (default 0.05).
#' @param power Desired power 1 - beta (default 0.80). Used only if `tau` is given.
#' @param two_sided Logical; if FALSE, uses a one-sided z-threshold for power inversion.
#' @param q_min Lower bound for numerical search over q (default 1e-9).
#' @param q_tol Absolute tolerance for q root-finding (default 1e-10).
#'
#' @details
#' Realized (conditional) RMSE: with standardized/whitened X and typical
#' orientation,
#' \deqn{ \mathrm{RMSE}_{\text{realized}}
#'   \approx \sqrt{\;\sigma^2\!\left(\frac{1}{n_T}+\frac{1}{n_C}\right)
#'   + \frac{\sigma_{\text{Prog}}^2}{d}\, M\;}
#'   \;=\; \sigma\,\sqrt{\left(\frac{1}{n_T}+\frac{1}{n_C}\right)
#'   + \frac{R^2}{1-R^2}\,\frac{M}{d}}\,,}
#' and the conservative upper bound replaces \eqn{\sigma_{\text{Prog}}^2/d} by \eqn{\sigma_{\text{Prog}}^2}.
#'
#' Ex-ante (design-stage) RMSE under thresholding:
#' with acceptance rule M <= a (acceptance probability q),
#' \deqn{ \mathbb{E}[\mathrm{MSE}\mid M\le a]
#'   = \left(\frac{1}{n_T}+\frac{1}{n_C}\right)\!\left(\sigma^2 + v_a(d)\,\sigma_{\text{Prog}}^2\right),}
#' where \eqn{v_a(d) = \Pr(\chi^2_{d+2}\le c)/\Pr(\chi^2_d\le c)} and
#' \eqn{c = a / (\tfrac{1}{n_T}+\tfrac{1}{n_C})}. Since
#' \eqn{q = \Pr(\chi^2_d \le c)}, we can parameterize by q:
#' \eqn{v(q;d) = \Pr(\chi^2_{d+2}\le \chi^2_{d;q})/q}, with
#' \eqn{\chi^2_{d;q}} the q-th quantile.
#'
#' Power inversion (Appendix D): for two-sided size \eqn{\alpha}{alpha}
#' and power \eqn{1-\beta}{1-beta}, a normal approximation suggests the RMSE goal
#' \eqn{|\tau| / (z_{1-\alpha/2}+z_{1-\beta})}{|tau| / (z_(1-alpha/2)+z_(1-beta))}.
#'
#' @return A list of class `"fastrerandomize_diagnostic"` with elements:
#' \itemize{
#'   \item \code{inputs}: Echo of parsed inputs and derived quantities (M, d, S=1/n_T+1/n_C).
#'   \item \code{realized}: \code{rmse_factor} (dimensionless, per sigma),
#'         \code{rmse}, and conservative \code{rmse_upper_factor}, \code{rmse_upper}.
#'   \item \code{power_check}: If \code{tau}, \code{alpha}, \code{power}, \code{sigma}, \code{R2}
#'         are given, includes \code{z_needed}, \code{z_realized}, and \code{already_sufficient}.
#'   \item \code{recommendation}: If a target is supplied (via \code{rmse_goal} or \code{tau}),
#'         returns \code{q_star}, \code{a_star}, \code{v_star}, \code{rmse_exante},
#'         \code{expected_M_accepted}, and \code{expected_draws_per_accept = 1/q_star}.
#' }
#'
#' @examples
#' # Example 1: observed SMDs, realized precision only (dimensionless factors)
#' smd <- c(0.10, -0.05, 0.08, 0.02)  # standardized mean differences
#' out1 <- diagnose_rerandomization(smd = smd, n_T = 100, n_C = 100)
#' print(out1)
#'
#' # Example 2: same, but supply sigma and R^2 for absolute RMSE
#' out2 <- diagnose_rerandomization(smd = smd, n_T = 100, n_C = 100, sigma = 1.2, R2 = 0.4)
#'
#' # Example 3: choose q to hit a power target (two-sided alpha=.05, 80% power, |tau|=0.2)
#' out3 <- diagnose_rerandomization(smd = smd, n_T = 100, n_C = 100, sigma = 1.2, R2 = 0.4,
#'                 tau = 0.2, alpha = 0.05, power = 0.80)
#'                 
#' # Analyze rerandomization recommendation given contextual factors
#' out3$recommendation
#'
#' # Example 4: choose q to hit an absolute RMSE goal directly
#' out4 <- diagnose_rerandomization(M = sum(smd^2), d = length(smd), n_T = 100, n_C = 100,
#'                 sigma = 1.2, R2 = 0.4, rmse_goal = 0.25)
#'
#' @export
diagnose_rerandomization <- function(
    smd = NULL,
    M = NULL,
    d = NULL,
    n_T,
    n_C,
    sigma = NULL,
    R2 = NULL,
    rmse_goal = NULL,
    tau = NULL,
    alpha = 0.05,
    power = 0.80,
    two_sided = TRUE,
    q_min = 1e-9,
    q_tol = 1e-10
){
  # --- basic checks & derived quantities ---
  stopifnot(is.numeric(n_T), is.numeric(n_C), n_T > 0, n_C > 0)
  S <- (1 / n_T) + (1 / n_C)  # shorthand from the paper
  if (!is.null(smd)) {
    smd <- as.numeric(smd)
    M <- sum(smd^2)
    d <- length(smd)
  } else {
    stopifnot(!is.null(M))
    if (is.null(d) || d <= 0) stop("If supplying only M, please also supply d (number of covariates).")
  }
  M <- as.numeric(M); d <- as.integer(d)
  
  # helpers for chi-square ratio v(q; d) and related mappings
  v_from_q <- function(q, df) {
    cval <- stats::qchisq(p = q, df = df)
    num  <- stats::pchisq(q = cval, df = df + 2L)
    num / q
  }
  a_from_q <- function(q, df, S) {
    cval <- stats::qchisq(p = q, df = df)
    cval * S  # since c = a / S
  }
  
  # sigma_Prog^2 / sigma^2 = R2 / (1 - R2); guard edges
  rat_X_over_eps <- function(R2) {
    if (is.null(R2)) return(NA_real_)
    if (R2 < 0 || R2 >= 1) stop("R2 must lie in [0, 1).")
    if (R2 == 0) return(0)  # no linear predictability
    R2 / (1 - R2)
  }
  rho <- rat_X_over_eps(R2)
  
  # --- Realized precision (eq. (1)) ---
  # Per-sigma "dimensionless" factor is useful even if sigma is unknown
  rmse_factor_realized <- sqrt(S + if (!is.na(rho)) (rho * M / d) else NA_real_)
  rmse_factor_upper    <- sqrt(S + if (!is.na(rho)) (rho * M)    else NA_real_)
  rmse_realized <- if (!is.null(sigma) && !is.na(rmse_factor_realized)) sigma * rmse_factor_realized else NA_real_
  rmse_upper    <- if (!is.null(sigma) && !is.na(rmse_factor_upper))    sigma * rmse_factor_upper    else NA_real_
  
  realized <- list(
    rmse_factor = rmse_factor_realized,
    rmse        = rmse_realized,
    rmse_upper_factor = rmse_factor_upper,
    rmse_upper        = rmse_upper
  )
  
  # --- If tau & (sigma,R2) are present, show "already sufficient?" diagnostic ---
  power_check <- NULL
  if (!is.null(tau) && !is.null(sigma) && !is.na(rmse_factor_realized) && !is.na(rho)) {
    z_alpha <- if (two_sided) stats::qnorm(1 - alpha / 2) else stats::qnorm(1 - alpha)
    z_beta  <- stats::qnorm(power)
    z_needed <- z_alpha + z_beta
    z_realized <- abs(tau) / (sigma * rmse_factor_realized)
    power_check <- list(
      z_needed = z_needed,
      z_realized = z_realized,
      already_sufficient = (z_realized >= z_needed)
    )
  }
  
  # --- Choose q to hit a target (rmse_goal OR power/|tau|) ---
  reason <- recommendation <- NULL
  have_target <- FALSE
  
  # If a power target is given via tau, alpha, power, convert to an RMSE goal
  if (!is.null(tau)) {
    if (is.null(sigma) || is.null(R2)) {
      warning("tau supplied but sigma and/or R2 missing; cannot solve for q. Provide sigma and R2.")
    } else {
      z_alpha <- if (two_sided) stats::qnorm(1 - alpha / 2) else stats::qnorm(1 - alpha)
      z_beta  <- stats::qnorm(power)
      rmse_goal <- abs(tau) / (z_alpha + z_beta)
      have_target <- TRUE
    }
  } else if (!is.null(rmse_goal)) {
    if (is.null(sigma) || is.null(R2)) {
      warning("rmse_goal supplied but sigma and/or R2 missing; cannot solve for q. Provide sigma and R2.")
    } else {
      have_target <- TRUE
    }
  }

  # Solve for minimal q such that ex-ante RMSE(q) <= rmse_goal
  if (have_target) {
    # required v* from S * (sigma^2 + v*sigma_Prog^2) <= rmse_goal^2
    # -> v* <= (rmse_goal^2 / (S * sigma^2) - 1) / (sigma_Prog^2 / sigma^2) = (...)
    v_target <- (rmse_goal^2 / (S * sigma^2) - 1)
    if (is.na(rho) || rho == 0) {
      # No linear predictability: sigma_Prog^2 = 0, so v doesn't matter.
      # Feasible iff rmse_goal^2 >= S*sigma^2
      feasible <- (rmse_goal^2 >= S * sigma^2)
      if (!feasible) {
        recommendation <- list(
          feasible = FALSE,
          reason = "Target RMSE is below the irreducible noise floor sqrt(S)*sigma; no stringency can attain it when R2=0."
        )
      } else {
        recommendation <- list(
          feasible = TRUE,
          q_star = 1.0,
          a_star = a_from_q(1.0, d, S),
          v_star = 1.0,
          rmse_exante = sqrt(S * sigma^2),  # since v irrelevant
          expected_M_accepted = S * d * 1.0,
          expected_draws_per_accept = 1.0
        )
      }
    } else {
      feasible <- (rmse_goal^2 >= S * sigma^2)
      if (!feasible) {
        recommendation <- list(
          feasible = FALSE,
          reason = "Target RMSE is below the irreducible noise floor sqrt(S)*sigma; even infinite stringency (q->0, v->0) cannot attain it when R2>0."
        )
      } else {
        v_target <- v_target / rho
        
        if (v_target <= 0) {
          # Asymptotically stringent (q -> 0) meets the target.
          q_star <- q_min
          reason <- "Asymptotically stringent (q -> 0) meets the target."
        } else if (v_target >= 1) {
          # Complete randomization already suffices.
          q_star <- 1.0
          reason <- "Complete randomization already suffices given target RMSE/power goals."
        } else {
          # Monotone root-find in q: solve v(q; d) - v_target = 0
          f <- function(q) v_from_q(q, d) - v_target
          # Check monotonic bracket
          f_lo <- f(q_min)
          f_hi <- f(1 - 1e-15)
          if (is.nan(f_lo) || is.nan(f_hi)){
            stop("Numerical issue evaluating v(q;d); try a different q_min.")
          }
          if (f_lo > 0) {
            # Even extremely small q has v>v_target -> push toward q_min
            q_star <- q_min
            reason <- "Even extremely small q has v>v_target -> pushing to q_min."
          } else if (f_hi < 0) {
            # Even q near 1 achieves v<v_target -> q=1
            q_star <- 1.0
            reason <- "Even q near 1 achieves v<v_target; setting -> q=1."
          } else {
            q_star <- tryCatch(
              uniroot(f, interval = c(q_min, 1 - 1e-15), tol = q_tol)$root,
              error = function(e) NA_real_
            )
            reason <- "q_star is found via uniroot."
          }
        }
        
        if (is.na(q_star)) {
          recommendation <- list(
            feasible = FALSE,
            reason = "Could not bracket a solution for q numerically."
          )
        } else {
          v_star <- v_from_q(q_star, d)
          a_star <- a_from_q(q_star, d, S)
          rmse_exante <- sqrt(S * (sigma^2 + (v_star * (rho * sigma^2))))
          recommendation <- list(
            feasible = TRUE,
            q_star = q_star,
            a_star = a_star,
            v_star = v_star,
            rmse_exante = rmse_exante,
            expected_M_accepted = S * d * v_star, # eq. (3)
            expected_draws_per_accept = 1 / q_star,
            reason = reason
          )
        }
      }
    }
  }
  
  out <- structure(
    list(
      inputs = list(
        n_T = n_T, n_C = n_C,
        S = S, d = d, M = M,
        sigma = sigma, R2 = R2,
        rmse_goal = rmse_goal, tau = tau,
        alpha = alpha, power = power, two_sided = two_sided
      ),
      realized = realized,
      power_check = power_check,
      recommendation = recommendation
    ),
    class = "fastrerandomize_diagnostic"
  )
  out
}



#' @export
print.fastrerandomize_diagnostic <- function(x, ...) {
  cat("diagnose_rerandomization(): diagnostic map\n")
  cat(sprintf("  n_T = %s, n_C = %s, d = %s\n", x$inputs$n_T, x$inputs$n_C, x$inputs$d))
  cat(sprintf("  M (sum SMD^2) = %.6g\n", x$inputs$M))
  if (!is.null(x$inputs$R2)) cat(sprintf("  sigma = %s, R^2 = %s\n", x$inputs$sigma, x$inputs$R2))
  cat("\nRealized precision (from observed M):\n")
  cat(sprintf("  RMSE (per sigma): %.6g", x$realized$rmse_factor)); 
  if (!is.na(x$realized$rmse)) cat(sprintf("   =>  RMSE = %.6g", x$realized$rmse))
  cat("\n")
  cat(sprintf("  Conservative upper bound (per sigma): %.6g", x$realized$rmse_upper_factor))
  if (!is.na(x$realized$rmse_upper)) cat(sprintf("   =>  upper RMSE = %.6g", x$realized$rmse_upper))
  cat("\n")
  
  if (!is.null(x$power_check)) {
    cat("\nPower check (with realized RMSE):\n")
    cat(sprintf("  z_needed = %.3f; z_realized = %.3f; already sufficient? %s\n",
                x$power_check$z_needed, x$power_check$z_realized,
                ifelse(x$power_check$already_sufficient, "YES", "NO")))
  }
  
  if (!is.null(x$recommendation)) {
    cat("\nTargeted stringency recommendation:\n")
    if (isFALSE(x$recommendation$feasible)) {
      cat("  (No feasible q)  ", x$recommendation$reason, "\n", sep = "")
    } else {
      cat(sprintf("  q* (randomization_accept_prob) = %.6g\n", x$recommendation$q_star))
      cat(sprintf("  threshold a* (on M)           = %.6g\n", x$recommendation$a_star))
      cat(sprintf("  v*                             = %.6g\n", x$recommendation$v_star))
      cat(sprintf("  ex-ante RMSE at q*             = %.6g\n", x$recommendation$rmse_exante))
      cat(sprintf("  E[M | M <= a*]                 = %.6g\n", x$recommendation$expected_M_accepted))
      cat(sprintf("  expected draws per accept      = %.6g (~1/q*)\n", x$recommendation$expected_draws_per_accept))
      cat(sprintf("  reason      = %s \n", x$recommendation$reason))
    }
  }
  invisible(x)
}

#' @export
summary.fastrerandomize_diagnostic <- function(object, ...) {
  print(object, ...)
  invisible(object)
}

# usage example 
if(FALSE){
  # Observed SMDs; want realized RMSE and conservative upper bound
  out <- diagnose_rerandomization(
    smd = observed_smds,   # or M = sum(observed_smds^2), d = length(observed_smds)
    n_T = n_treated,
    n_C = n_control,
    sigma = sigma_est,     # optional
    R2    = R2_est         # optional
  )
  out$realized$rmse            # plug-in realized RMSE (if sigma & R2 supplied)
  out$realized$rmse_upper      # conservative upper bound
  
  # Power planning: largest q that achieves 1-β power at size α for |τ|
  plan <- diagnose_rerandomization(
    smd = observed_smds,
    n_T = n_treated, n_C = n_control,
    sigma = sigma_est, R2 = R2_est,
    tau = abs(target_tau), alpha = 0.05, power = 0.80
  )
  plan$recommendation$q_star         # <-- randomization_accept_prob to use
  plan$recommendation$a_star         # Mahalanobis threshold a
  plan$recommendation$expected_M_accepted
  plan$recommendation$expected_draws_per_accept
}
  
  