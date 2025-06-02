r_files <- list.files(file.path('fastrerandomize','R'), full.names = TRUE)
for(f in r_files) source(f)

# helper for approximate equality
approx_equal <- function(x, y, tol = 1e-8) {
  if (is.na(x) || is.na(y)) stop('NA comparison')
  abs(x - y) < tol
}

# hotellingT2_R
X <- matrix(c(1,2,3,4,5,6,7,8), nrow = 4)
W <- c(1,0,1,0)
stopifnot(approx_equal(hotellingT2_R(X, W), 1.2))

# diff_in_means_R
Y <- 1:4
stopifnot(diff_in_means_R(Y, W) == -1)

# compute_diff_at_tau_for_oneW_R
Wprime <- c(0,1,0,1)
obsY <- c(1,2,3,4)
obsW <- c(1,0,1,0)
stopifnot(compute_diff_at_tau_for_oneW_R(Wprime, obsY, obsW, 1) == 3)

# randomization_test_R
obsW2 <- c(1,1,0,0)
obsY2 <- c(2,2,1,0)
comb <- t(combn(4,2))
allW <- matrix(0, nrow = nrow(comb), ncol = 4)
for(i in seq_len(nrow(comb))) allW[i, comb[i,]] <- 1
res <- randomization_test_R(obsW2, obsY2, allW, findFI = FALSE)
stopifnot(approx_equal(res$p_value, 1/3))
stopifnot(res$tau_obs == 1.5)

# generate_randomizations_R (monte carlo)
set.seed(1)
X2 <- matrix(rnorm(8), 4, 2)
res2 <- generate_randomizations_R(4, 2, X2, 1, 'monte_carlo', max_draws = 6, batch_size = 2)
stopifnot(ncol(res2$randomizations) == 4)
stopifnot(length(res2$balance) == nrow(res2$randomizations))

cat('All tests passed\n')
