#!/usr/bin/env Rscript
# This tutorial shows how to define a custom threshold function for randomization
{
    # Load fastrerandomize
    library(fastrerandomize); options( error = NULL )

    # Generate synthetic data
    covariates <- matrix(rnorm(100),nrow=10)
    
    # Example usage of GenerateRandomizations with custom threshold function
    CandidateRandomizations <- fastrerandomize::generate_randomizations(n_units = 10L, 
                                                     n_treated = 5L, 
                                                     X = covariates, 
                                                     randomization_accept_prob = 0.5)
}
