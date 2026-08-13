# C++ B-loop for the logit parametric bootstrap, mirroring fh_boot_arcsin_cpp.
#
# This is where the payoff is: logit_bc costs one integrate() per in-sample
# domain, and inside the bootstrap that multiplies by B -- the same structure
# that took arcsin to 130x.
#
# Parity is against an exact R mirror of the loop rather than against
# boot_logit() itself, because the two draw independent RNG streams (the kernel
# takes pre-generated matrices, as fh_boot_arcsin_cpp does). Given identical
# draws, the two must agree to linear-algebra accuracy.

logit_inv_r <- function(l) exp(l) / (1 + exp(l))

# Exact R transcription of the kernel's per-iteration arithmetic.
mirror_boot_logit <- function(sigmau2, vardir, beta, X, predX, is_in,
                              v_boot, e_boot, eblup_corr, bc,
                              lower, upper) {
  M <- nrow(predX); m <- nrow(X); B <- ncol(v_boot)
  tol <- .Machine$double.eps^0.25
  in_idx <- which(is_in == 1L)
  Xbeta <- as.numeric(predX %*% beta)

  est <- tru <- matrix(NA_real_, M, B)
  for (b in seq_len(B)) {
    vb <- v_boot[, b]; eb <- e_boot[, b]
    tru[, b] <- logit_inv_r(Xbeta + vb)

    ystar <- Xbeta[in_idx] + vb[in_idx] + eb
    s2b <- fh_estsigmau2_reml_cpp(ystar, X, vardir, lower, upper, tol)

    vi <- 1 / (s2b + vardir)
    XtViX <- t(X) %*% (X * vi); XtViy <- t(X) %*% (ystar * vi)
    bb <- solve(XtViX, XtViy)
    uh <- s2b * (vi * as.numeric(ystar - X %*% bb))

    est_trans <- as.numeric(predX %*% bb)
    est_trans[in_idx] <- as.numeric(X %*% bb) + uh

    var_in <- s2b * (vardir / (s2b + vardir))
    sd_all <- rep(0, M); sd_all[in_idx] <- sqrt(var_in)

    if (bc) {
      ev <- logit_inv_r(est_trans)                       # OOS / zero-var
      ev[in_idx] <- fh_logit_integral_cpp(est_trans[in_idx], sd_all[in_idx])
      est[, b] <- ev
    } else {
      est[, b] <- logit_inv_r(est_trans)
    }
  }

  d <- est - tru
  list(mse = rowMeans(d^2),
       Li = eblup_corr + apply(d, 1, quantile, 0.025, type = 7),
       Ui = eblup_corr + apply(d, 1, quantile, 0.975, type = 7))
}

make_case <- function(seed = 3, m = 24L, M = 30L, B = 25L) {
  set.seed(seed)
  X <- cbind(1, rnorm(m)); predX <- cbind(1, rnorm(M))
  list(sigmau2 = 0.4, vardir = runif(m, 0.02, 0.3), beta = c(0.2, 0.5),
       X = X, predX = predX,
       is_in = c(rep(1L, m), rep(0L, M - m)),
       v_boot = matrix(rnorm(M * B, 0, sqrt(0.4)), M, B),
       e_boot = matrix(rnorm(m * B), m, B) * sqrt(runif(m, 0.02, 0.3)),
       eblup_corr = runif(M, 0.2, 0.7), lower = 0, upper = 10)
}

test_that("fh_boot_logit_cpp matches the exact R mirror (bc and naive)", {
  d <- make_case()
  for (bc in c(TRUE, FALSE)) {
    got <- fh_boot_logit_cpp(d$sigmau2, d$vardir, d$beta, d$X, d$predX,
                             d$is_in, d$v_boot, d$e_boot, d$eblup_corr,
                             bc, d$lower, d$upper)
    ref <- mirror_boot_logit(d$sigmau2, d$vardir, d$beta, d$X, d$predX,
                             d$is_in, d$v_boot, d$e_boot, d$eblup_corr,
                             bc, d$lower, d$upper)
    lbl <- paste("bc =", bc)
    expect_equal(as.numeric(got$mse), ref$mse, tolerance = 1e-7, info = lbl)
    expect_equal(as.numeric(got$Li), ref$Li, tolerance = 1e-7, info = lbl)
    expect_equal(as.numeric(got$Ui), ref$Ui, tolerance = 1e-7, info = lbl)
  }
})

test_that("fh_boot_logit_cpp is invariant to the thread count", {
  d <- make_case(seed = 9)
  nthr <- emdi_cores(4L)
  skip_if(nthr < 2L, "needs at least 2 cores to be meaningful")

  a <- fh_boot_logit_cpp(d$sigmau2, d$vardir, d$beta, d$X, d$predX, d$is_in,
                         d$v_boot, d$e_boot, d$eblup_corr, TRUE,
                         d$lower, d$upper, threads = 1L)
  b <- fh_boot_logit_cpp(d$sigmau2, d$vardir, d$beta, d$X, d$predX, d$is_in,
                         d$v_boot, d$e_boot, d$eblup_corr, TRUE,
                         d$lower, d$upper, threads = nthr)
  expect_equal(as.numeric(a$mse), as.numeric(b$mse), tolerance = 1e-12)
  expect_equal(as.numeric(a$Li), as.numeric(b$Li), tolerance = 1e-12)
})

test_that("the results are probabilities and the interval brackets them", {
  d <- make_case(seed = 21)
  got <- fh_boot_logit_cpp(d$sigmau2, d$vardir, d$beta, d$X, d$predX, d$is_in,
                           d$v_boot, d$e_boot, d$eblup_corr, TRUE,
                           d$lower, d$upper)
  expect_true(all(is.finite(got$mse)) && all(got$mse >= 0))
  expect_true(all(got$Li <= got$Ui))
})

test_that("boot_logit routes through the kernel, and works where R cannot", {
  skip_on_cran()
  data("eusilcA_popAgg", package = "emdi2")
  data("eusilcA_smpAgg", package = "emdi2")
  cd <- combine_data(eusilcA_popAgg, "Domain", eusilcA_smpAgg, "Domain")

  run <- function(engine) withr::with_options(
    list(emdi.fh_engine = engine), {
      set.seed(1)
      suppressMessages(fh(
        fixed = MTMED ~ eqsize + cash + self_empl, vardir = "Var_MTMED",
        combined_data = cd, domains = "Domain", method = "reml",
        interval = c(0, 1e7), transformation = "logit",
        backtransformation = "bc", eff_smpsize = "n",
        MSE = TRUE, mse_type = "boot", B = 10
      ))
    })

  # The R path dies in logit_bc()'s integrate() -- exp(l)/(1+exp(l)) is NaN at
  # the mu +/- 50*sd bounds once a replicate's posterior sd exceeds ~14.2, and
  # sigmau2 is re-estimated every replicate so one bad draw is enough.
  expect_error(run("r"), "non-finite")

  # The kernel uses the stable sigmoid and clips to +/- 8, so it completes.
  res <- run("cpp")
  expect_true(all(is.finite(res$MSE$FH[res$MSE$Out == 0])))
  expect_true(all(res$MSE$FH[res$MSE$Out == 0] >= 0))

  # Dispatch probe: value identity cannot show the kernel was reached, because
  # an unwired path would just fall back to R -- which here errors, so this is
  # belt and braces rather than the only signal.
  called <- FALSE
  orig <- fh_boot_logit_cpp
  testthat::with_mocked_bindings(
    invisible(run("cpp")),
    fh_boot_logit_cpp = function(...) { called <<- TRUE; orig(...) },
    .package = "emdi2"
  )
  expect_true(called)
})
