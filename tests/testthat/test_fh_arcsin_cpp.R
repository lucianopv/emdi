# Oracle: the exact integrand boot_arcsin_2 / arcsin_bc integrate numerically.
# Use tight integration bounds [max(0, mu-8s), min(pi/2, mu+8s)] so that
# integrate() does not miss the sharp spike when sigma is very small.
bc_integrand <- function(x, mean, sd) sin(x)^2 * stats::dnorm(x, mean = mean, sd = sd)
bc_ref <- function(mu, s) {
  lo <- max(0,    mu - 8 * s)
  hi <- min(pi/2, mu + 8 * s)
  if (hi <= lo) return(0.0)
  stats::integrate(bc_integrand, lo, hi, mu, s, subdivisions = 1000L)$value
}

test_that("fh_bc_integral_cpp matches R integrate() across a mu/sigma grid", {
  mus    <- c(0.01, 0.2, 0.5, 0.785, 1.2, 1.56, -0.1, 1.7)  # incl. outside [0,pi/2]
  sigmas <- c(1e-4, 1e-3, 0.01, 0.05, 0.1, 0.3, 0.7, 1.0)
  grid <- expand.grid(mu = mus, s = sigmas)
  ref  <- mapply(bc_ref, grid$mu, grid$s)
  got  <- fh_bc_integral_cpp(grid$mu, grid$s)
  expect_equal(as.numeric(got), as.numeric(ref), tolerance = 1e-6)
})

test_that("fh_bc_integral_cpp small-sigma guard returns the correct limit", {
  # sigma -> 0: integral -> sin^2(mu) for mu in (0, pi/2); 0 if mu outside [0,pi/2].
  mu  <- c(0.3, -0.2, 1.7)
  exp <- c(sin(0.3)^2, 0, 0)
  got <- fh_bc_integral_cpp(mu, rep(1e-13, 3))
  expect_equal(as.numeric(got), exp, tolerance = 1e-12)
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
    tt <- pmin(pmax(Xbeta + vb, 0), pi/2)
    true[, b] <- sin(tt)^2
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
  beta   <- c(0.6, 0.1)
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
                          eblup_corr, TRUE, interval[1], interval[2], threads = 4L)
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
  ok <- is.finite(c1$MSE$FH) & is.finite(r1$MSE$FH) & r1$MSE$FH > 0
  expect_gt(stats::cor(c1$MSE$FH[ok], r1$MSE$FH[ok]), 0.95)
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
    .package = "emdi2"
  )
  expect_identical(seen, emdi_cores(3L))
})
