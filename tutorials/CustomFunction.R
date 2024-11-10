#!/usr/bin/env Rscript
# This tutorial shows how to define a custom threshold function for randomization
{
    # Load fastrerandomize
    library(fastrerandomize)
    # Initialize JAX
    fastrerandomize::InitializeJAX(conda_env = "jax_cpu", conda_env_required = T)

    # Generate synthetic data
    covariates <- jnp$array(matrix(rnorm(100),nrow=10))
    treatment_assignment <- jnp$array(matrix(sample(c(0,1), size = 10, replace = T)))
    n0 <- jnp$array(5)
    n1 <- jnp$array(5)

    # Define custom threshold function
    # The inputs to this function should be covariates, treatment assignment, n0, n1, of the same shape of what is above

    my_threshold_func <- function(covariates, treatment_assignment, n0, n1){
    # Create boolean masks
        mask_c <- treatment_assignment == 0
        mask_t <- treatment_assignment == 1
        xbar_c <- jnp$mean(jnp$where(mask_c$reshape(c(-1L,1L)), covariates, 0L), axis=0L)
        xbar_t <- jnp$mean(jnp$where(mask_t$reshape(c(-1L,1L)), covariates, 0L), axis=0L)
        
        dist_ <- jnp$linalg$norm(xbar_t - xbar_c, axis=0L)
        return(dist_)
    }

    # Vectorize the threshold function, the vectorization should be over the randomization
    my_threshold_func_vec <- jax$vmap(my_threshold_func, in_axes = list(NULL, 0L, NULL, NULL))

    # Example usage of GenerateRandomizations with custom threshold function
    rand <- GenerateRandomizations(10L, 5L, X = covariates, randomization_accept_prob = 0.5, threshold_func = my_threshold_func_vec)
}