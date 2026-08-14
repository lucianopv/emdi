# C++ fast path for the Jiang jackknife MSE (fh_jackknife_cpp).
#
# The R implementation runs m delete-one refits, each rebuilding a framework_FH
# and round-tripping through wrapper_estsigmau2() and eblup_FH(). The kernel
# does the whole loop in C++ on (direct, X, vardir), reusing the Plan-1
# fh_estsigmau2_reml_cpp and the diagonal EBLUP algebra.
#
# The loop is RNG-free, so unlike the bootstrap kernels it is deterministic and
# parity with R is exact to linear-algebra accuracy, not merely statistical.

jk_data <- local({
  load("FH/eusilcA_popAgg.RData")
  load("FH/eusilcA_smpAgg.RData")
  combine_data(
    pop_data = eusilcA_popAgg, pop_domains = "Domain",
    smp_data = eusilcA_smpAgg, smp_domains = "Domain"
  )
})

jk_fixed <- MTMED ~ cash + age_ben

# Reference: transcription of jiang_jackknife()'s numeric core, driven purely by
# (direct, X, vardir) so it is independent of the framework plumbing.
ref_jackknife <- function(direct, X, vardir, sigmau2, fh_full, lower, upper) {
  m <- length(direct)
  tol <- .Machine$double.eps^0.25
  g1 <- vardir * (1 - vardir / (sigmau2 + vardir))

  dg1 <- matrix(0, m, m)
  deb <- matrix(0, m, m)
  for (j in seq_len(m)) {
    keep <- setdiff(seq_len(m), j)
    s2j <- stats::optimize(
      f = function(s2) -fh_reml_loglik_cpp(s2, direct[keep], X[keep, , drop = FALSE],
                                           vardir[keep]),
      interval = c(lower, upper), tol = tol
    )$minimum

    dg1[, j] <- vardir * (1 - vardir / (s2j + vardir)) - g1

    core <- fh_eblup_core_cpp(s2j, direct, X, vardir)
    fh_j <- as.numeric(X %*% core$beta_hat) + as.numeric(core$u_hat)
    deb[, j] <- fh_j - fh_full
  }

  cc <- (m - 1) / m
  g1 - cc * rowSums(dg1) + cc * rowSums(deb^2)
}

test_that("fh_jackknife_cpp matches the R jackknife numeric core", {
  fr <- framework_FH(
    combined_data = jk_data, fixed = jk_fixed, vardir = "Var_MTMED",
    domains = "Domain", transformation = "no", correlation = "no",
    corMatrix = NULL, eff_smpsize = "n", Ci = NULL, tol = NULL, maxit = NULL
  )
  direct <- as.numeric(fr$direct)
  X <- fr$model_X
  vardir <- as.numeric(fr$vardir)
  lower <- 0; upper <- 1e7
  tol <- .Machine$double.eps^0.25

  s2 <- fh_estsigmau2_reml_cpp(direct, X, vardir, lower, upper, tol)
  core <- fh_eblup_core_cpp(s2, direct, X, vardir)
  fh_full <- as.numeric(X %*% core$beta_hat) + as.numeric(core$u_hat)

  want <- ref_jackknife(direct, X, vardir, s2, fh_full, lower, upper)
  got <- fh_jackknife_cpp(direct, X, vardir, s2, fh_full, lower, upper, tol)

  expect_equal(as.numeric(got$mse), want, tolerance = 1e-6)
  expect_length(got$jack_sigmau2, length(direct))
  expect_true(all(is.finite(got$jack_sigmau2)))
})

test_that("fh_jackknife_cpp is invariant to the thread count", {
  nthr <- emdi_cores(4L)
  skip_if(nthr < 2L, "needs at least 2 cores to be meaningful")
  fr <- framework_FH(
    combined_data = jk_data, fixed = jk_fixed, vardir = "Var_MTMED",
    domains = "Domain", transformation = "no", correlation = "no",
    corMatrix = NULL, eff_smpsize = "n", Ci = NULL, tol = NULL, maxit = NULL
  )
  direct <- as.numeric(fr$direct); X <- fr$model_X
  vardir <- as.numeric(fr$vardir); tol <- .Machine$double.eps^0.25
  s2 <- fh_estsigmau2_reml_cpp(direct, X, vardir, 0, 1e7, tol)
  core <- fh_eblup_core_cpp(s2, direct, X, vardir)
  fh_full <- as.numeric(X %*% core$beta_hat) + as.numeric(core$u_hat)

  a <- fh_jackknife_cpp(direct, X, vardir, s2, fh_full, 0, 1e7, tol, threads = 1L)
  b <- fh_jackknife_cpp(direct, X, vardir, s2, fh_full, 0, 1e7, tol, threads = nthr)

  expect_equal(as.numeric(a$mse), as.numeric(b$mse), tolerance = 1e-12)
})

test_that("fh() jackknife: cpp engine reproduces the r engine end to end", {
  run <- function(engine) withr::with_options(
    list(emdi.fh_engine = engine),
    suppressMessages(fh(
      fixed = jk_fixed, vardir = "Var_MTMED", combined_data = jk_data,
      domains = "Domain", method = "reml", interval = c(0, 1e7),
      transformation = "arcsin", backtransformation = "naive",
      eff_smpsize = "n", MSE = TRUE, mse_type = "jackknife"
    ))
  )
  f_r <- run("r"); f_cpp <- run("cpp")

  expect_equal(f_cpp$MSE$FH, f_r$MSE$FH, tolerance = 1e-6)
  expect_equal(f_cpp$ind$FH, f_r$ind$FH, tolerance = 1e-6)
  expect_equal(f_cpp$MSE$Out, f_r$MSE$Out)
})

test_that("the cpp jackknife path is taken for reml and skipped for ml", {
  # framework_FH() call count is a direct probe of which path ran: the cpp
  # kernel does the whole delete-one loop internally and rebuilds no framework,
  # so only fh()'s own single framework construction remains.
  count_fw <- function(expr) {
    n <- 0L
    orig <- framework_FH
    testthat::with_mocked_bindings(
      force(expr),
      framework_FH = function(...) { n <<- n + 1L; orig(...) },
      .package = "emdi"
    )
    n
  }
  runner <- function(meth, engine) withr::with_options(
    list(emdi.fh_engine = engine),
    suppressMessages(fh(
      fixed = jk_fixed, vardir = "Var_MTMED", combined_data = jk_data,
      domains = "Domain", method = meth, interval = c(0, 1e7),
      transformation = "arcsin", backtransformation = "naive",
      eff_smpsize = "n", MSE = TRUE, mse_type = "jackknife"
    ))
  )

  expect_equal(count_fw(runner("reml", "cpp")), 1L)   # loop entirely in C++
  expect_gt(count_fw(runner("reml", "r")), 1L)        # R loop rebuilds per domain
  # ml uses a different sigmau2 estimator, so it must stay on the R loop even
  # when the cpp engine is selected.
  expect_gt(count_fw(runner("ml", "cpp")), 1L)
})

# ---------------------------------------------------------------------------
# cpus core-budget argument (Task 7)
# ---------------------------------------------------------------------------

test_that("fh() accepts cpus and returns identical results at 1 and 2", {
  run <- function(n) suppressMessages(fh(
    fixed = jk_fixed, vardir = "Var_MTMED", combined_data = jk_data,
    domains = "Domain", method = "reml", interval = c(0, 1e7),
    transformation = "arcsin", backtransformation = "naive",
    eff_smpsize = "n", MSE = TRUE, mse_type = "jackknife", cpus = n))
  a <- run(1L); b <- run(2L)
  expect_equal(a$MSE$FH, b$MSE$FH, tolerance = 1e-12)
  expect_equal(a$ind$FH, b$ind$FH, tolerance = 1e-12)
})

# The identity check above cannot prove the budget reaches the kernel: the
# jackknife loop is RNG-free, so 1 and 2 threads give identical results
# whether or not the thread count is actually forwarded. This reads the value
# fh_jackknife_cpp() is handed via the real call chain -- arcsin_mse() calls
# wrapper_MSE() directly (Trap 1 from the task brief: FH.R's own two
# wrapper_MSE() call sites are NOT on the arcsin+jackknife path).
test_that("fh() hands the resolved cpus budget to the jackknife kernel", {
  budget <- emdi_cores(3L)
  skip_if(budget < 2L, "machine reports a single core")
  seen <- integer(0)
  orig <- fh_jackknife_cpp          # capture BEFORE mocking, or this recurses
  testthat::with_mocked_bindings(
    {
      invisible(suppressMessages(fh(
        fixed = jk_fixed, vardir = "Var_MTMED", combined_data = jk_data,
        domains = "Domain", method = "reml", interval = c(0, 1e7),
        transformation = "arcsin", backtransformation = "naive",
        eff_smpsize = "n", MSE = TRUE, mse_type = "jackknife", cpus = 3L)))
    },
    fh_jackknife_cpp = function(..., threads = 1L) {
      seen <<- c(seen, as.integer(threads))
      orig(..., threads = threads)
    },
    .package = "emdi"
  )
  expect_identical(seen, budget)
})
