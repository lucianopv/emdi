# The bc back-transformation of Hadam et al. (2023, eq. 5): E[sin^2(t)] for t ~ N(mu, s^2), over the WHOLE
# real line. Oracle: numerical integration over mu +/- 12 s (the full normal mass), not over [0, pi/2].
bc_integrand <- function(x, mean, sd) sin(x)^2 * stats::dnorm(x, mean = mean, sd = sd)
bc_ref <- function(mu, s) {
  stats::integrate(bc_integrand, mu - 12 * s, mu + 12 * s, mu, s, subdivisions = 1000L, rel.tol = 1e-10)$value
}
# What emdi integrated before the fix (only [0, pi/2]); used to show low rates do not move.
bc_ref_truncated <- function(mu, s) {
  lo <- max(0, mu - 8 * s); hi <- min(pi/2, mu + 8 * s)
  if (hi <= lo) return(0.0)
  stats::integrate(bc_integrand, lo, hi, mu, s, subdivisions = 1000L)$value
}

test_that("bc follows Hadam et al. eq. 5: the expectation over the whole real line", {
  got <- fh_bc_integral_cpp(c(pi/2, 1.5, asin(sqrt(0.97))), c(0.03, 0.1, 0.1))
  expect_equal(as.numeric(got), c(0.999101, 0.985195, 0.960693), tolerance = 1e-6)
  r <- withr::with_options(list(emdi.fh_engine = "r"),
                           arcsin_bc(list(M = 1L, obs_dom = TRUE), pi/2, 0.03^2))
  expect_equal(as.numeric(r), 0.999101, tolerance = 1e-6)
  mus    <- seq(-0.2, pi/2 + 0.2, length.out = 8)
  sigmas <- c(1e-4, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8)
  grid <- expand.grid(mu = mus, s = sigmas)
  expect_equal(as.numeric(fh_bc_integral_cpp(grid$mu, grid$s)), mapply(bc_ref, grid$mu, grid$s),
               tolerance = 1e-6)
})

test_that("tiny sigma gives sin^2(mu) for any mu, also just past pi/2", {
  mu  <- c(pi/2 + 0.05, -0.03, 0.3)
  got <- fh_bc_integral_cpp(mu, rep(1e-12, 3))
  expect_equal(as.numeric(got), sin(mu)^2, tolerance = 1e-12)
})

test_that("low rates are unchanged by the whole-line expectation", {
  mu <- asin(sqrt(0.05))
  expect_equal(as.numeric(fh_bc_integral_cpp(mu, 0.03)), bc_ref_truncated(mu, 0.03), tolerance = 1e-6)
})

test_that("arcsin_bc cpp engine equals r engine on a realistic mu/var vector", {
  set.seed(7)
  M  <- 40
  obs <- rep(TRUE, M); obs[1:5] <- FALSE          # 5 OOS
  fr <- list(M = M, obs_dom = obs)
  mu  <- runif(M, 0.05, 1.5)
  # variances >= 1e-3 (sigma >= ~0.03): R integrate() is an accurate oracle here.
  v   <- rep(NA_real_, M); v[obs] <- runif(sum(obs), 1e-3, 0.05)
  bc_r   <- withr::with_options(list(emdi.fh_engine = "r"),   arcsin_bc(fr, mu, v))
  bc_cpp <- withr::with_options(list(emdi.fh_engine = "cpp"), arcsin_bc(fr, mu, v))
  expect_equal(as.numeric(bc_cpp), as.numeric(bc_r), tolerance = 1e-6)
  # OOS entries are sin(mu)^2 in both
  expect_equal(bc_cpp[!obs], sin(mu[!obs])^2, tolerance = 1e-12)
})

# Exact R mirror of the intended C++ algorithm: same pre-generated draws, the
# closed-form bc integral (via fh_bc_integral_cpp), diagonal REML + EBLUP.
boot_arcsin_mirror <- function(sigmau2, vardir, beta, X, predX, is_in,
                               v_boot, e_boot, eblup_corr, bc, interval) {
  M <- nrow(predX); m <- nrow(X); B <- ncol(v_boot)
  in_idx <- which(is_in == 1L)
  Xbeta  <- as.numeric(predX %*% beta)
  est <- true <- matrix(NA_real_, M, B)
  for (b in seq_len(B)) {
    vb <- v_boot[, b]; eb <- e_boot[, b]
    true[, b] <- sin(Xbeta + vb)^2               # the bootstrap truth of Hadam et al. (2023, p. 14): unclipped
    ystar <- Xbeta[in_idx] + vb[in_idx] + eb
    s2b <- fh_estsigmau2_reml_cpp(ystar, X, as.numeric(vardir),
                                  interval[1], interval[2], .Machine$double.eps^0.25)
    vi <- 1/(s2b + vardir)
    Q  <- solve(t(vi * X) %*% X)
    bb <- as.numeric(Q %*% (t(X) %*% (vi * ystar)))
    uh <- s2b * (vi * (ystar - as.numeric(X %*% bb)))
    est_trans <- numeric(M)
    est_trans[in_idx] <- as.numeric(X %*% bb) + uh
    est_trans[-in_idx] <- as.numeric(predX[-in_idx, , drop = FALSE] %*% bb)
    var_in <- s2b * (vardir/(s2b + vardir))
    eb_val <- numeric(M)
    for (d in seq_len(M)) {
      if (is_in[d] == 1L) {
        if (bc) eb_val[d] <- fh_bc_integral_cpp(est_trans[d],
                                                sqrt(var_in[match(d, in_idx)]))
        else    eb_val[d] <- sin(est_trans[d])^2
      } else if (bc) {
        eb_val[d] <- fh_bc_integral_cpp(est_trans[d], sqrt(s2b))   # out of sample: eq. 5 with gamma = 0
      } else eb_val[d] <- sin(est_trans[d])^2
    }
    est[, b] <- eb_val
  }
  mse <- rowMeans((est - true)^2)
  Li <- Ui <- numeric(M)
  for (d in seq_len(M)) {
    q <- stats::quantile(est[d, ] - true[d, ], c(0.025, 0.975), names = FALSE)
    Li[d] <- eblup_corr[d] + q[1]; Ui[d] <- eblup_corr[d] + q[2]
  }
  list(mse = mse, Li = Li, Ui = Ui)
}

test_that("fh_boot_arcsin_cpp matches the exact R mirror (bc and naive)", {
  set.seed(11)
  m <- 25; M <- 30; p <- 2; B <- 40
  X     <- cbind(1, rnorm(m))
  predX <- cbind(1, rnorm(M))
  is_in <- c(rep(1L, m), rep(0L, M - m))
  vardir <- runif(m, 1e-3, 0.02)
  beta   <- c(1.3, 0.3)          # x'beta spans pi/2, so a clipped truth would differ
  sigmau2 <- 0.01
  eblup_corr <- runif(M, 0.1, 0.6)
  interval <- c(0, 0.5)
  v_boot <- matrix(rnorm(M*B, 0, sqrt(sigmau2)), M, B)
  e_boot <- matrix(rnorm(m*B), m, B) * sqrt(vardir)

  for (bc in c(TRUE, FALSE)) {
    ref <- boot_arcsin_mirror(sigmau2, vardir, beta, X, predX, is_in,
                              v_boot, e_boot, eblup_corr, bc, interval)
    got <- fh_boot_arcsin_cpp(sigmau2, vardir, beta, X, predX, is_in,
                              v_boot, e_boot, eblup_corr, bc,
                              interval[1], interval[2])
    expect_equal(as.numeric(got$mse), ref$mse, tolerance = 1e-7,
                 info = paste("bc =", bc))
    expect_equal(as.numeric(got$Li), ref$Li, tolerance = 1e-7)
    expect_equal(as.numeric(got$Ui), ref$Ui, tolerance = 1e-7)
  }
})

test_that("fh_boot_arcsin_cpp is thread-count invariant", {
  nthr <- emdi_cores(4L)
  skip_if(nthr < 2L, "needs at least 2 cores to be meaningful")
  set.seed(13)
  m <- 30; M <- 36; B <- 60
  X     <- cbind(1, rnorm(m)); predX <- cbind(1, rnorm(M))
  is_in <- c(rep(1L, m), rep(0L, M - m))
  vardir <- runif(m, 1e-3, 0.02); beta <- c(0.5, 0.2)
  s2 <- 0.012; eblup_corr <- runif(M, 0.1, 0.6); interval <- c(0, 0.5)
  v_boot <- matrix(rnorm(M*B, 0, sqrt(s2)), M, B)
  e_boot <- matrix(rnorm(m*B), m, B) * sqrt(vardir)
  a <- fh_boot_arcsin_cpp(s2, vardir, beta, X, predX, is_in, v_boot, e_boot,
                          eblup_corr, TRUE, interval[1], interval[2], threads = 1L)
  b <- fh_boot_arcsin_cpp(s2, vardir, beta, X, predX, is_in, v_boot, e_boot,
                          eblup_corr, TRUE, interval[1], interval[2], threads = nthr)
  expect_equal(as.numeric(a$mse), as.numeric(b$mse), tolerance = 1e-12)
  expect_equal(as.numeric(a$Li),  as.numeric(b$Li),  tolerance = 1e-12)
  expect_equal(as.numeric(a$Ui), as.numeric(b$Ui), tolerance = 1e-12)
})

test_that("fh() arcsin+boot: cpp deterministic; point matches R; MSE MC-equivalent", {
  data("eusilcA_popAgg"); data("eusilcA_smpAgg")
  combined <- combine_data(eusilcA_popAgg, "Domain", eusilcA_smpAgg, "Domain")
  f0 <- function(engine, B = 100) withr::with_options(
    list(emdi.fh_engine = engine),
    fh(MTMED ~ cash + self_empl, vardir = "Var_MTMED", combined_data = combined,
       domains = "Domain", method = "reml", transformation = "arcsin",
       backtransformation = "bc", eff_smpsize = "n", MSE = TRUE,
       mse_type = "boot", B = c(B, 0), seed = 123))

  c1 <- f0("cpp"); c2 <- f0("cpp")
  expect_equal(c1$MSE$FH, c2$MSE$FH, tolerance = 1e-10)   # deterministic given seed

  r1 <- f0("r")
  expect_equal(c1$ind$FH, r1$ind$FH, tolerance = 1e-6)    # point back-transform deterministic

  # Bootstrap MSE: cpp and r use INDEPENDENT RNG streams, so per-domain MSE has
  # ~sqrt(2/B) MC noise (~15-20% at B=100). Assert (a) structural agreement
  # (correlation) and (b) domain-MEAN MSE agreement (averaging cancels MC noise;
  # the closed-form integral matches integrate() so there's no systematic bias).
  # With the paper's unclipped truth the domain MSEs are more alike than they were under the clipped one (whose
  # outliers near 1 propped up the correlation), so at B = 100 MC noise alone keeps the correlation near 0.86
  # (0.94 at B = 400, median per-domain ratio 1.008 at both). Systematic agreement is checked directly.
  ok <- is.finite(c1$MSE$FH) & is.finite(r1$MSE$FH) & r1$MSE$FH > 0
  expect_gt(stats::cor(c1$MSE$FH[ok], r1$MSE$FH[ok]), 0.80)
  expect_true(abs(stats::median(c1$MSE$FH[ok] / r1$MSE$FH[ok]) - 1) < 0.10)
  expect_lt(abs(mean(c1$MSE$FH[ok]) - mean(r1$MSE$FH[ok])) / mean(r1$MSE$FH[ok]), 0.10)
  expect_true(all(c1$MSE$FH[ok] > 0))
})

# ---------------------------------------------------------------------------
# cpus core-budget argument (Task 7) -- Trap 2 from the task brief:
# boot_arcsin_2() is NOT reached through wrapper_MSE(). Its real chain is
# fh() -> backtransformed() -> arcsin_bt() -> arcsin_mse() -> boot_arcsin_2()
# -> fh_boot_arcsin_cpp(), and none of the three intermediates forwarded
# `threads` before this task. A value-identity test cannot distinguish a
# correctly-wired budget from one silently dropped back to 1 (the kernel is
# deterministic given the same pre-generated draws either way), so this reads
# the thread count the kernel actually receives.
# ---------------------------------------------------------------------------

test_that("fh() hands the resolved cpus budget to the arcsin bootstrap kernel", {
  budget <- emdi_cores(3L)
  skip_if(budget < 2L, "machine reports a single core")
  data("eusilcA_popAgg"); data("eusilcA_smpAgg")
  combined <- combine_data(eusilcA_popAgg, "Domain", eusilcA_smpAgg, "Domain")

  seen <- integer(0)
  orig <- fh_boot_arcsin_cpp        # capture BEFORE mocking, or this recurses
  testthat::with_mocked_bindings(
    {
      invisible(suppressMessages(fh(
        MTMED ~ cash + self_empl, vardir = "Var_MTMED", combined_data = combined,
        domains = "Domain", method = "reml", transformation = "arcsin",
        backtransformation = "bc", eff_smpsize = "n", MSE = TRUE,
        mse_type = "boot", B = c(5, 0), seed = 123, cpus = 3L)))
    },
    fh_boot_arcsin_cpp = function(..., threads = 1L) {
      seen <<- c(seen, as.integer(threads))
      orig(..., threads = threads)
    },
    .package = "emdi"
  )
  expect_identical(seen, budget)
})

test_that("out-of-sample areas under bc take eq. 5 with gamma = 0", {
  data("eusilcA_popAgg"); data("eusilcA_smpAgg")
  combined <- combine_data(eusilcA_popAgg, "Domain", eusilcA_smpAgg, "Domain")
  oos <- 1:5
  combined$MTMED[oos] <- NA; combined$Var_MTMED[oos] <- NA; combined$n[oos] <- NA
  f0 <- function(engine, bt) suppressMessages(withr::with_options(list(emdi.fh_engine = engine),
    fh(MTMED ~ cash + self_empl, vardir = "Var_MTMED", combined_data = combined, domains = "Domain",
       method = "reml", transformation = "arcsin", backtransformation = bt, eff_smpsize = "n", MSE = FALSE)))
  c1 <- f0("cpp", "bc"); r1 <- f0("r", "bc")
  xb <- as.numeric(stats::model.matrix(~ cash + self_empl, combined[oos, ]) %*% c1$model$coefficients$coefficients)
  out <- c1$ind$Out == 1
  expect_equal(sum(out), length(oos))
  expect_equal(c1$ind$FH[out], (1 - exp(-2 * c1$model$variance) * cos(2 * xb)) / 2, tolerance = 1e-8)
  expect_equal(c1$ind$FH, r1$ind$FH, tolerance = 1e-6)
  nv <- f0("cpp", "naive")
  expect_equal(nv$ind$FH[out], sin(pmin(pmax(xb, 0), pi/2))^2, tolerance = 1e-8)   # naive: unchanged
})

test_that("fh() returns the bootstrap intervals of Hadam et al. eq. 7 for arcsin", {
  data("eusilcA_popAgg"); data("eusilcA_smpAgg")
  combined <- combine_data(eusilcA_popAgg, "Domain", eusilcA_smpAgg, "Domain")
  f0 <- function(...) suppressMessages(fh(MTMED ~ cash + self_empl, vardir = "Var_MTMED", combined_data = combined,
                                          domains = "Domain", method = "reml", eff_smpsize = "n", MSE = TRUE, ...))
  a <- f0(transformation = "arcsin", backtransformation = "bc", mse_type = "boot", B = c(30, 0), seed = 7)
  b <- f0(transformation = "arcsin", backtransformation = "bc", mse_type = "boot", B = c(30, 0), seed = 7)
  expect_s3_class(a$CI, "data.frame")
  expect_named(a$CI, c("Domain", "FH_Lower", "FH_Upper"))
  expect_equal(nrow(a$CI), nrow(a$ind))
  expect_equal(as.character(a$CI$Domain), as.character(a$ind$Domain))
  expect_true(all(a$CI$FH_Lower <= a$CI$FH_Upper))
  expect_identical(a$CI, b$CI)                                          # deterministic given the seed
  expect_null(f0(transformation = "arcsin", backtransformation = "naive", mse_type = "jackknife")$CI)
  expect_null(f0(transformation = "no", mse_type = "analytical")$CI)   # (a log fit here hangs on MTMED's 0: fork issue #9)
})
