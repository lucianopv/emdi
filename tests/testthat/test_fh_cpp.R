# Reference: exact transcription of A.reml (R/estim_sigmau2.R), the dense oracle.
ref_reml_loglik <- function(s2, direct, X, vardir) {
  m  <- length(direct)
  V  <- s2 * diag(m) + diag(as.numeric(vardir))
  Vi <- solve(V)
  Q  <- solve(t(X) %*% Vi %*% X)
  P  <- Vi - Vi %*% X %*% Q %*% t(X) %*% Vi
  ee <- eigen(V)$values
  -(m / 2) * log(2 * pi) - 0.5 * sum(log(ee)) -
    0.5 * log(det(t(X) %*% Vi %*% X)) -
    0.5 * as.numeric(t(direct) %*% P %*% direct)
}

test_that("fh_reml_loglik_cpp matches the dense A.reml formula on a sigma2 grid", {
  data("eusilcA_smpAgg")
  fixed  <- Mean ~ Cash
  X      <- model.matrix(fixed, eusilcA_smpAgg)
  direct <- eusilcA_smpAgg$Mean
  vardir <- eusilcA_smpAgg$Var_Mean
  upper  <- var(direct)
  for (s2 in seq(1e-6 * upper, upper, length.out = 25)) {
    expect_equal(
      fh_reml_loglik_cpp(s2, direct, X, as.numeric(vardir)),
      ref_reml_loglik(s2, direct, X, vardir),
      tolerance = 1e-9, info = paste("sigma2 =", s2)
    )
  }
})
