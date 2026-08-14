# C++ kernel for the logit back-transformation integral.
#
# Upstream's logit_bc() (R/back_transformation.R) computes, per in-sample
# domain, E[logit^-1(theta)] with theta ~ N(mu, var), via integrate() over
# mu +/- 50*sd -- carrying the author's own comment "# Can this be vectorized?"
# on the loop. fh_logit_integral_cpp is the GL-64 answer, mirroring
# fh_bc_integral_cpp for arcsin.
#
# IMPORTANT: unlike the arcsin case, this is a SPEED port, not a correctness
# one. arcsin_bc integrated over a fixed [0, pi/2], which let integrate() miss
# a narrow spike and underflow (see NEWS). logit_bc's bounds are already
# centred on the spike, so upstream avoided that trap. The parity bar is
# therefore strict: if the kernel disagrees with integrate(), the kernel is
# wrong.

logit_inv_r <- function(l) exp(l) / (1 + exp(l))

# Direct transcription of logit_bc's integral for one domain.
ref_logit_one <- function(mu, sd) {
  integrand <- function(x, mean, sd) logit_inv_r(x) * dnorm(x, mean, sd)
  integrate(integrand, lower = mu - 50 * sd, upper = mu + 50 * sd,
            mu, sd)$value
}

test_that("fh_logit_integral_cpp matches integrate() across a mu/sd grid", {
  mus <- c(-4, -2, -0.5, 0, 0.5, 2, 4)
  sds <- c(0.01, 0.05, 0.2, 0.5, 1, 2)
  grid <- expand.grid(mu = mus, sd = sds)

  got <- fh_logit_integral_cpp(grid$mu, grid$sd)
  want <- mapply(ref_logit_one, grid$mu, grid$sd)

  expect_equal(as.numeric(got), as.numeric(want), tolerance = 1e-8)
})

test_that("the sd -> 0 limit is logit_inverse(mu)", {
  mus <- c(-3, -1, 0, 1, 3)
  got <- fh_logit_integral_cpp(mus, rep(0, length(mus)))
  expect_equal(as.numeric(got), logit_inv_r(mus), tolerance = 1e-12)
})

test_that("the kernel stays finite where R's logit_inverse overflows", {
  # R's exp(l)/(1+exp(l)) is Inf/Inf = NaN for l beyond ~710. The kernel uses
  # the numerically stable sigmoid, so it is defined on the whole line. It
  # agrees with R wherever R is finite (covered by the grid test above); this
  # pins that it does not itself produce NaN/Inf at the extremes.
  expect_true(is.nan(logit_inv_r(800)))            # R's form breaks
  got <- fh_logit_integral_cpp(c(-800, 0, 800), c(1, 1, 1))
  expect_true(all(is.finite(got)))
  expect_lt(got[1], 1e-8)          # deep in the lower tail -> ~0
  expect_gt(got[3], 1 - 1e-8)      # deep in the upper tail -> ~1
})

test_that("the result is a probability for every input", {
  set.seed(11)
  mu <- runif(200, -6, 6); sd <- runif(200, 1e-3, 3)
  got <- fh_logit_integral_cpp(mu, sd)
  expect_true(all(got >= 0 & got <= 1))
})

test_that("length mismatch is rejected", {
  expect_error(fh_logit_integral_cpp(c(0, 1), c(1)), "length")
})

# --- engine parity, end to end ------------------------------------------------
# logit_bc() now routes through the kernel under the default engine. These pin
# that flipping the engine changes nothing observable, which is the property
# the whole engine-switch design rests on.

# Package data and upstream's own formula, matching
# test_fh_backtransformation.R. The 15-domain FH/*.RData fixtures lack eqsize,
# and their MTMED reaches exactly 1, where logit is infinite.
logit_data <- local({
  data("eusilcA_popAgg", package = "emdi")
  data("eusilcA_smpAgg", package = "emdi")
  combine_data(pop_data = eusilcA_popAgg, pop_domains = "Domain",
               smp_data = eusilcA_smpAgg, smp_domains = "Domain")
})

test_that("logit_bc gives the same answer under both engines", {
  fr <- framework_FH(
    combined_data = logit_data, fixed = MTMED ~ eqsize + cash + self_empl,
    vardir = "Var_MTMED", domains = "Domain", transformation = "logit",
    correlation = "no", corMatrix = NULL, eff_smpsize = "n",
    Ci = NULL, tol = NULL, maxit = NULL
  )
  set.seed(4)
  mu <- rnorm(fr$M, 0, 1.5)
  var <- runif(fr$M, 1e-4, 0.5)

  r   <- withr::with_options(list(emdi.fh_engine = "r"),
                             logit_bc(fr$M, mu, var, fr$obs_dom))
  cpp <- withr::with_options(list(emdi.fh_engine = "cpp"),
                             logit_bc(fr$M, mu, var, fr$obs_dom))

  expect_equal(as.numeric(cpp), as.numeric(r), tolerance = 1e-8)
  expect_length(cpp, fr$M)
})

test_that("the kernel computes what integrate() cannot: upstream overflow bug", {
  # logit_bc() integrates over mu +/- 50*sd using
  # logit_inverse(l) = exp(l)/(1 + exp(l)). That form is Inf/Inf = NaN once
  # exp(l) overflows (l beyond ~710), so whenever the posterior sd exceeds
  # ~14.2 the integrand is NaN at the endpoints and integrate() aborts with
  # "non-finite function value". Reachable in practice: a model with few
  # covariates leaves enough residual variance to get there.
  #
  # The kernel avoids it twice over -- the numerically stable sigmoid, and
  # clipping to +/- 8 where the mass actually is.
  mu <- 0; sd <- 15
  f <- function(x, mean, sd) (exp(x) / (1 + exp(x))) * dnorm(x, mean, sd)
  expect_error(integrate(f, mu - 50 * sd, mu + 50 * sd, mu, sd),
               "non-finite")

  got <- fh_logit_integral_cpp(mu, sd)
  expect_true(is.finite(got))
  expect_true(got >= 0 && got <= 1)
})

test_that("fh() logit + bc: cpp engine reproduces the r engine", {
  skip_on_cran()
  # Three covariates keep the posterior sd small enough that the R path's
  # integrate() does not hit the overflow above, so the two engines can be
  # compared at all. This is upstream's own formula from
  # test_fh_backtransformation.R.
  run <- function(engine) withr::with_options(
    list(emdi.fh_engine = engine),
    suppressMessages(fh(
      fixed = MTMED ~ eqsize + cash + self_empl, vardir = "Var_MTMED",
      combined_data = logit_data, domains = "Domain", method = "reml",
      interval = c(0, 1e7), transformation = "logit",
      backtransformation = "bc", eff_smpsize = "n", MSE = FALSE
    ))
  )
  f_r <- try(run("r"), silent = TRUE)
  skip_if(inherits(f_r, "try-error"),
          "R path hits the logit_bc overflow on this fit; parity untestable")
  f_cpp <- run("cpp")
  expect_equal(f_cpp$ind$FH, f_r$ind$FH, tolerance = 1e-8)
})
