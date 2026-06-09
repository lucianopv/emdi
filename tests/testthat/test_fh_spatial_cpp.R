make_spatial_fr <- function() {
  data("eusilcA_popAgg"); data("eusilcA_smpAgg"); data("eusilcA_prox")
  combined <- combine_data(eusilcA_popAgg, "Domain", eusilcA_smpAgg, "Domain")
  framework_FH(combined_data = combined, fixed = Mean ~ cash + self_empl,
               vardir = "Var_Mean", domains = "Domain", transformation = "no",
               eff_smpsize = NULL, correlation = "spatial",
               corMatrix = as.matrix(eusilcA_prox), Ci = NULL, tol = 1e-4, maxit = 100)
}

test_that("fh_sreml_cpp matches R SREML (sigmau2, rho, convergence)", {
  fr <- make_spatial_fr()
  W <- as.matrix(fr$W); direct <- fr$direct; X <- fr$model_X
  vardir <- as.numeric(fr$vardir); m <- fr$m
  r   <- SREML(direct = direct, X = X, vardir = vardir, areanumber = m,
               W = W, maxit = 100, tol = 1e-4)
  cpp <- fh_sreml_cpp(direct, X, vardir, W, 100L, 1e-4)
  expect_equal(cpp$sigmau2, r$sigmau2, tolerance = 1e-6)
  expect_equal(cpp$rho,     r$rho,     tolerance = 1e-6)
  expect_equal(cpp$convergence, r$convergence)
})

test_that("fh_sreml_cpp matches R SREML on non-convergence (maxit = 1)", {
  fr <- make_spatial_fr()
  W <- as.matrix(fr$W); direct <- fr$direct; X <- fr$model_X
  vardir <- as.numeric(fr$vardir); m <- fr$m
  r   <- SREML(direct = direct, X = X, vardir = vardir, areanumber = m,
               W = W, maxit = 1, tol = 1e-4)
  cpp <- fh_sreml_cpp(direct, X, vardir, W, 1L, 1e-4)
  expect_equal(cpp$convergence, r$convergence)   # both FALSE after 1 iteration
  expect_equal(cpp$sigmau2, r$sigmau2, tolerance = 1e-6)
  expect_equal(cpp$rho,     r$rho,     tolerance = 1e-6)
})

test_that("wrapper_estsigmau2 cpp==r for reml spatial", {
  fr <- make_spatial_fr()
  s_r   <- withr::with_options(list(emdi.fh_engine = "r"),
             wrapper_estsigmau2(fr, method = "reml", interval = c(0, var(fr$direct))))
  s_cpp <- withr::with_options(list(emdi.fh_engine = "cpp"),
             wrapper_estsigmau2(fr, method = "reml", interval = c(0, var(fr$direct))))
  expect_equal(s_cpp$sigmau2, s_r$sigmau2, tolerance = 1e-6)
  expect_equal(s_cpp$rho,     s_r$rho,     tolerance = 1e-6)
  expect_equal(s_cpp$convergence, s_r$convergence)
})
