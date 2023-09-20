#!/usr/bin/env Rscript
#' Initialize JAX
#'
#' @usage
#'
#' InitializeJAXFxns()
#'
#' @param

#' @return A list consiting of \itemize{
#'   \item
#' }
#'
#' @section References:
#' \itemize{
#' \item
#' }
#'
#' @examples
#' # For tutorials, see
#' # github.com/cjerzak/fastrerandomization-software
#'
#' @export
#' @md

InitializeJAXFxns <- function(){
  FastHotel2T2 <<- jax$jit( function(samp_, t_, n0, n1){
    # set up calc
    RowBroadcast <- jax$vmap(function(mat, vec){
      jnp$multiply(mat, vec)},
      in_axes = list(1L, NULL))
    xbar1 <- jnp$divide(jnp$sum(RowBroadcast(samp_,t_),1L,keepdims = T), n1)
    xbar2 <- jnp$divide(jnp$sum(RowBroadcast(samp_,jnp$subtract(1.,t_)),1L,keepdims = T), n0)
    CovPooled <- jnp$cov(samp_, rowvar = F)
    CovWts <- jnp$add(jnp$reciprocal(n0), jnp$reciprocal(n1))
    xbar_diff <- jnp$subtract(xbar1, xbar2)

    # perform calc
    Tstat <- jnp$matmul(
      jnp$matmul(jnp$transpose(xbar_diff),
                 jnp$linalg$inv( jnp$multiply(CovPooled,CovWts) )),
      xbar_diff)
  })

  VectorizedFastHotel2T2 <<- jax$jit( jax$vmap(function(samp_, t_, n0, n1){
    FastHotel2T2(samp_, t_, n0, n1)},
    in_axes = list(NULL, 0L, NULL, NULL)) )

  if(T == F){
    # tests
    Compositional::hotel2T2()
    samp_ <- jnp$array(matrix(rnorm(10*100),nrow=100))
    t_ <- jnp$array(sample(c(0,1), size = 100, replace =T))
    t_full <- jnp$array(matrix(rbinom(100*34,size = 1, prob = 0.5),nrow = 100))
    FastHotel2T2(samp_ = samp_, t_ = t_, n0 = n0, n1 = n1)
    VectorizedFastHotel2T2(samp_, t_full, n0, n1)
  }

  FastDiffInMeans <<- jax$jit( function(y_,t_, n0, n1){
    my1 <- jnp$divide(jnp$sum(jnp$multiply(y_, t_)), n1)
    my0 <- jnp$divide(jnp$sum(jnp$multiply(y_, jnp$subtract(1.,t_))), n0)
    return( diff10 <- jnp$subtract(my1, my0) )
  })

  VectorizedFastDiffInMeans <<- jax$jit( jax$vmap(function(y_, t_, n0, n1){
    FastDiffInMeans(y_, t_, n0, n1)},
    in_axes = list(NULL, 0L, NULL, NULL)) )

  if(T == F){
    # tests
    y_ <- jnp$array(rnorm(10))
    t_ <-  jnp$array(rbinom(10,size=1, prob = 0.5))
    n0 <- jnp$array(10.)
    n1 <- jnp$array(3.)
    t_mat <- jnp$array( matrix(rbinom(100*10, size = 1, prob = 0.5),nrow = 100) )
    FastDiffInMeans(y_, t_, n0, n1)
    np$array( VectorizedFastDiffInMeans(y_, t_mat, n0, n1) )
  }
}
