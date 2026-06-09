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

test_that("fh_eblup_sfh_cpp matches eblup_SFH numeric core at fixed (sigmau2, rho)", {
  fr <- make_spatial_fr()
  data("eusilcA_popAgg"); data("eusilcA_smpAgg")
  combined <- combine_data(eusilcA_popAgg, "Domain", eusilcA_smpAgg, "Domain")
  s2 <- list(sigmau2 = 0.5 * var(fr$direct), rho = 0.4, convergence = TRUE)
  e_r <- withr::with_options(list(emdi.fh_engine = "r"), eblup_SFH(fr, s2, combined))
  cpp <- fh_eblup_sfh_cpp(s2$sigmau2, s2$rho, fr$direct, fr$model_X,
                          as.numeric(fr$vardir), as.matrix(fr$W))
  expect_equal(as.numeric(cpp$beta_hat), as.numeric(e_r$coefficients$coefficients),
               tolerance = 1e-8)
  expect_equal(unname(as.matrix(cpp$Q)), unname(as.matrix(e_r$beta_vcov)),
               tolerance = 1e-7)   # beta_vcov accumulates 3 inversions; 1e-8 is borderline cross-LAPACK
  expect_equal(as.numeric(cpp$u_hat), as.numeric(e_r$random_effects), tolerance = 1e-8)
})

test_that("eblup_SFH cpp engine equals r engine (point + coef names)", {
  fr <- make_spatial_fr()
  data("eusilcA_popAgg"); data("eusilcA_smpAgg")
  combined <- combine_data(eusilcA_popAgg, "Domain", eusilcA_smpAgg, "Domain")
  s2 <- list(sigmau2 = 0.5 * var(fr$direct), rho = 0.4, convergence = TRUE)
  e_r   <- withr::with_options(list(emdi.fh_engine = "r"),   eblup_SFH(fr, s2, combined))
  e_cpp <- withr::with_options(list(emdi.fh_engine = "cpp"), eblup_SFH(fr, s2, combined))
  expect_equal(e_cpp$eblup_data$FH, e_r$eblup_data$FH, tolerance = 1e-7)
  expect_equal(as.numeric(e_cpp$random_effects), as.numeric(e_r$random_effects), tolerance = 1e-7)
  expect_equal(rownames(e_cpp$coefficients), rownames(e_r$coefficients))  # arma name guard
})
