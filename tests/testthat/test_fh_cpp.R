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

test_that("fh_estsigmau2_reml_cpp matches R Reml() (optimize) to optimizer tolerance", {
  data("eusilcA_smpAgg")
  fixed  <- Mean ~ Cash
  X      <- model.matrix(fixed, eusilcA_smpAgg)
  direct <- eusilcA_smpAgg$Mean
  vardir <- eusilcA_smpAgg$Var_Mean
  upper  <- var(direct)
  tol    <- .Machine$double.eps^0.25

  r_s2   <- Reml(interval = c(0, upper), vardir = vardir, x = X,
                 direct = direct, areanumber = length(direct))
  cpp_s2 <- fh_estsigmau2_reml_cpp(direct, X, as.numeric(vardir), 0, upper, tol)

  # optimize() locates the maximiser to ~tol; require agreement well inside that.
  expect_equal(cpp_s2, as.numeric(r_s2), tolerance = 1e-6)
})

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

test_that("fh_eblup_core_cpp matches dense EBLUP algebra at fixed sigma2", {
  data("eusilcA_smpAgg")
  fixed  <- Mean ~ Cash
  X      <- model.matrix(fixed, eusilcA_smpAgg)
  direct <- eusilcA_smpAgg$Mean
  vardir <- as.numeric(eusilcA_smpAgg$Var_Mean)
  m      <- length(direct)
  s2     <- 0.5 * var(direct)

  # Dense reference (eblup_FH internal block)
  V    <- s2 * diag(m) + diag(vardir)
  Vi   <- solve(V)
  Qr   <- solve(t(X) %*% Vi %*% X)
  br   <- Qr %*% t(X) %*% Vi %*% direct
  resr <- direct - c(X %*% br)
  ur   <- s2 * (Vi %*% resr)

  core <- fh_eblup_core_cpp(s2, direct, X, vardir)
  expect_equal(as.numeric(core$beta_hat), as.numeric(br), tolerance = 1e-9)
  expect_equal(unname(core$Q), unname(Qr), tolerance = 1e-9)
  expect_equal(as.numeric(core$u_hat),    as.numeric(ur), tolerance = 1e-9)
})

test_that("engine switch + wrapper_estsigmau2 cpp/r parity (reml, no correlation)", {
  expect_true(is.function(.fh_use_cpp))
  expect_false(withr::with_options(list(emdi.fh_engine = "r"),   .fh_use_cpp()))
  expect_true( withr::with_options(list(emdi.fh_engine = "cpp"), .fh_use_cpp()))
  expect_true(withr::with_options(list(emdi.fh_engine = "INVALID"), .fh_use_cpp()))
  expect_true(withr::with_options(list(emdi.fh_engine = 42L),       .fh_use_cpp()))

  data("eusilcA_smpAgg")
  fr <- framework_FH(
    combined_data = eusilcA_smpAgg, fixed = Mean ~ Cash, vardir = "Var_Mean",
    domains = "Domain", transformation = "no", eff_smpsize = NULL,
    correlation = "no", corMatrix = NULL, Ci = NULL, tol = 0.0001, maxit = 100
  )
  interval <- c(0, var(fr$direct))
  s2_r   <- withr::with_options(list(emdi.fh_engine = "r"),
              wrapper_estsigmau2(fr, method = "reml", interval = interval))
  s2_cpp <- withr::with_options(list(emdi.fh_engine = "cpp"),
              wrapper_estsigmau2(fr, method = "reml", interval = interval))
  expect_equal(as.numeric(s2_cpp), as.numeric(s2_r), tolerance = 1e-6)
})
